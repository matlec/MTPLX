# SPDX-License-Identifier: Apache-2.0
"""Metal kernels of the Inkling 2 step engine (``decode.py``).

A decode step on M1 is bound by the host: MLX spends a few microseconds of
CPU per graph node, and the reference layer visit builds ~90 nodes. These
kernels let one visit run as ~13 nodes: each writes its outputs in the shape
the next one reads, so no reshape, cast or slice nodes sit in between.

Every kernel takes ``T`` <= 16 token rows of one sequence (decode is
``T == 1``; DSpark verify is ``T == block``). Rounding follows the reference
graph in ``model.py``.
"""

from __future__ import annotations

import mlx.core as mx

_kernels: dict = {}


def _kernel(name, inputs, outputs, source, header="", contiguous=True):
    k = _kernels.get(name)
    if k is None:
        k = mx.fast.metal_kernel(
            name=name,
            input_names=list(inputs),
            output_names=list(outputs),
            source=source,
            header=header,
            ensure_row_contiguous=contiguous,
        )
        _kernels[name] = k
    return k


_consts: dict = {}


def const(key, make):
    """A cached, evaluated constant (one upload, no per-step graph node)."""
    out = _consts.get(key)
    if out is None:
        out = make()
        mx.eval(out)
        _consts[key] = out
    return out


_HEADER = """
// Block-wide sum over a threadgroup of nsg simdgroups; every thread gets it.
inline float tg_sum(float v, threadgroup float* scratch, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint i = 0; i < nsg; ++i) t += scratch[i];
    return t;
}
// Same over a 128-thread threadgroup (4 simdgroups).
inline float tg_sum4(float v, threadgroup float* red, uint lane, uint sg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return red[0] + red[1] + red[2] + red[3];
}
"""

# --------------------------------------------------------------------------
# Attention residuals: depth mixing of unit-RMS block sources
# --------------------------------------------------------------------------
#
# MODE 0 (pre):   h -> hn = unit_rms(h); mix (blocks, hn) with Q queries;
#                 NORM: out_q = rms_norm(mix_q, w_q) (BF16), else raw FP32.
#                 With NB > 0 also mixes the loop bank (NB blocks + bank
#                 partial) with 4 message queries -> msg [4, T, H] FP32.
#                 FIRST: layer 0, no mixing: out = rms_norm(h, w).
# MODE 1 (post):  a (o_proj out) -> short conv with state -> residual
#                 (BOUNDARY: a alone) -> mix (blocks, unit_rms(residual))
#                 with one query -> out (norm as above).


_RESMIX_TG = 1024


def _resmix_source(mode: int, ns: int, q: int, nb: int, norm: bool, first: bool, boundary: bool):
    """Sources stream from memory in each pass; only the normalized partial
    row lives in threadgroup memory, so register use stays small."""
    srcs = [f"b{i}" for i in range(ns)]
    bank = [f"m{i}" for i in range(max(nb, 0))]
    L = []
    a = L.append
    a("""
    constexpr uint TG = RESMIX_TG;
    constexpr uint NSG = TG / 32;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.x;
    const uint T = threadgroups_per_grid.x;
    threadgroup float scratch[32];
    threadgroup float red[48][NSG];
    threadgroup float pnt[H];
    const size_t row = (size_t)t * H;
""")
    if mode == 0:
        a("""
    float ss = 0.0f;
    for (uint e = tid; e < H; e += TG) {
        const float v = float(h[row + e]);
        ss += v * v;
    }
    ss = tg_sum(ss, scratch, lane, sg, NSG);
    const float hinv = metal::precise::rsqrt(ss / H + 1e-6f);
    for (uint e = tid; e < H; e += TG) {
        const float v = float(h[row + e]) * hinv;
        pnt[e] = v;
        hn[row + e] = v;
    }
""")
        if first:
            a("""
    for (uint e = tid; e < H; e += TG) {
        out0[row + e] = bfloat(float(bfloat(float(h[row + e]) * hinv)) * float(nw[e]));
    }
""")
            return "".join(L)
    else:
        a(f"""
    float ss = 0.0f;
    for (uint e = tid; e < H; e += TG) {{
        float acc = 0.0f;
        for (int j = 0; j < KC; ++j) {{
            const int p = int(t) + j;
            const float val = p < KC - 1 ? state[p * H + e]
                                         : float(a_in[(size_t)(p - (KC - 1)) * H + e]);
            acc += float(conv_w[e * KC + j]) * val;
        }}
        const float x = float(a_in[row + e]);
        const float ac = float(bfloat((float(bfloat(acc)) + x) * cscale[0]));
        const float rv = {"ac" if boundary else "float(h[row + e]) + ac"};
        resid[row + e] = rv;
        pnt[e] = rv;
        ss += rv * rv;
        win[(size_t)(KC - 1 + t) * H + e] = x;
        if (t == 0) {{
            for (int j = 0; j < KC - 1; ++j) {{
                win[(size_t)j * H + e] = state[j * H + e];
                const int p = int(T) + j;
                next_state[j * H + e] = p < KC - 1 ? state[p * H + e]
                                                   : float(a_in[(size_t)(p - (KC - 1)) * H + e]);
            }}
        }}
    }}
    ss = tg_sum(ss, scratch, lane, sg, NSG);
    const float rinv = metal::precise::rsqrt(ss / H + 1e-6f);
    for (uint e = tid; e < H; e += TG) pnt[e] *= rinv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
""")

    def src_expr(n: int, names: list, partial: str) -> str:
        return names[n] + "[row + e]" if n < len(names) else partial

    def mix_block(names, partial, nq, qname, outw, tag):
        nsrc = len(names) + 1
        a(f"""
    {{
        float acc[{nsrc * nq}];
        for (uint k = 0; k < {nsrc * nq}; ++k) acc[k] = 0.0f;
        for (uint e = tid; e < H; e += TG) {{
""")
        for n in range(nsrc):
            a(f"            const float s{n} = {src_expr(n, names, partial)};\n")
        for qi in range(nq):
            a(f"            {{ const float qv = {qname}[{qi} * H + e];\n")
            for n in range(nsrc):
                a(f"              acc[{n * nq + qi}] += s{n} * qv;\n")
            a("            }\n")
        a(f"""        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < {nsrc * nq}; ++k) {{
            const float v = simd_sum(acc[k]);
            if (lane == 0) red[k][sg] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float w[{nsrc * nq}];
        for (uint k = 0; k < {nsrc * nq}; ++k) {{
            float s = 0.0f;
            for (uint j = 0; j < NSG; ++j) s += red[k][j];
            w[k] = s * rscale[0];
        }}
        for (uint qi = 0; qi < {nq}; ++qi) {{
            float m = -INFINITY;
            for (uint n = 0; n < {nsrc}; ++n) m = max(m, w[n * {nq} + qi]);
            float z = 0.0f;
            for (uint n = 0; n < {nsrc}; ++n) {{
                w[n * {nq} + qi] = metal::exp(w[n * {nq} + qi] - m);
                z += w[n * {nq} + qi];
            }}
            for (uint n = 0; n < {nsrc}; ++n) w[n * {nq} + qi] /= z;
        }}
""")
        for qi in range(nq):
            mix = " + ".join(f"w[{n * nq + qi}] * {src_expr(n, names, partial)}" for n in range(nsrc))
            if outw == "norm":
                a(f"""
        {{
            float ms = 0.0f;
            for (uint e = tid; e < H; e += TG) {{ const float v = {mix}; ms += v * v; }}
            ms = tg_sum(ms, scratch, lane, sg, NSG);
            const float inv = metal::precise::rsqrt(ms / H + RMS_EPS);
            for (uint e = tid; e < H; e += TG) {{
                const float v = {mix};
                out{qi}[row + e] = bfloat(v * inv * float(nw[{qi if nq > 1 else 0} * H + e]));
            }}
        }}
""")
            else:
                a(f"""
        for (uint e = tid; e < H; e += TG) {tag}[({qi} * T + t) * H + e] = {mix};
""")
        a("    }\n")

    mix_block(srcs, "pnt[e]", q, "query", "norm" if norm else "raw", "outraw")
    if nb >= 0:
        a("    threadgroup_barrier(mem_flags::mem_threadgroup);\n")
        mix_block(bank, "mp[row + e]", 4, "mquery", "raw", "msg")
    return "".join(L)


def resmix(
    *,
    mode: int,
    hidden: int,
    tokens: int,
    blocks: list,
    query: mx.array,
    rscale: mx.array,
    h: mx.array | None = None,
    norm_w: mx.array | None = None,
    first: bool = False,
    bank: tuple | None = None,
    mquery: mx.array | None = None,
    a_in: mx.array | None = None,
    state: mx.array | None = None,
    conv_w: mx.array | None = None,
    cscale: mx.array | None = None,
    boundary: bool = False,
    rms_eps: float = 1e-6,
):
    """See the mode table above. Returns a dict of the named outputs."""
    ns = len(blocks)
    nq = int(query.shape[0]) if query is not None else 0
    # -1: no loop bank; 0: a bank of just the previous pass's partial.
    nb = len(bank[0]) if bank is not None else -1
    do_norm = norm_w is not None
    key = ("resmix", mode, ns, nq, nb, do_norm, first, boundary)
    names, arrays = [], []
    if mode == 0:
        names.append("h"); arrays.append(h)
    else:
        names += ["a_in", "h", "state", "conv_w", "cscale"]
        arrays += [a_in, h, state, conv_w, cscale]
    for i, b in enumerate(blocks):
        names.append(f"b{i}"); arrays.append(b)
    if not first:
        names += ["query", "rscale"]
        arrays += [query, rscale]
    if do_norm:
        names.append("nw"); arrays.append(norm_w)
    if nb >= 0:
        for i, b in enumerate(bank[0]):
            names.append(f"m{i}"); arrays.append(b)
        names += ["mp", "mquery"]
        arrays += [bank[1], mquery]
    outputs, shapes, dtypes = [], [], []
    if mode == 0:
        outputs.append("hn"); shapes.append((tokens, hidden)); dtypes.append(mx.float32)
    else:
        kc = conv_w.shape[-1]
        outputs += ["resid", "win", "next_state"]
        shapes += [(tokens, hidden), (kc - 1 + tokens, hidden), (kc - 1, hidden)]
        dtypes += [mx.float32, mx.float32, mx.float32]
    if first:
        outputs.append("out0"); shapes.append((tokens, hidden)); dtypes.append(mx.bfloat16)
    elif do_norm:
        for qi in range(nq):
            outputs.append(f"out{qi}"); shapes.append((tokens, hidden)); dtypes.append(mx.bfloat16)
    else:
        outputs.append("outraw"); shapes.append((nq, tokens, hidden)); dtypes.append(mx.float32)
    if nb >= 0:
        outputs.append("msg"); shapes.append((4, tokens, hidden)); dtypes.append(mx.float32)
    name = "mtplx_inkling2_resmix_" + "_".join(str(int(x)) for x in key[1:]).replace("-", "n")
    k = _kernels.get(name)
    if k is None:
        source = _resmix_source(mode, ns, nq, nb, do_norm, first, boundary).replace(
            "RMS_EPS", f"{float(rms_eps)!r}f"
        ).replace("RESMIX_TG", str(_RESMIX_TG))
        k = _kernel(name, names, outputs, source, _HEADER)
    res = k(
        inputs=arrays,
        template=[("H", hidden)] + ([("KC", int(conv_w.shape[-1]))] if mode == 1 else []),
        grid=(_RESMIX_TG * tokens, 1, 1),
        threadgroup=(_RESMIX_TG, 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    return dict(zip(outputs, res))


# --------------------------------------------------------------------------
# RMS norm of FP32 rows with per-input weights -> BF16 (loop visits)
# --------------------------------------------------------------------------

_NORM_SOURCE = """
    constexpr uint TG = 1024;
    constexpr uint EPT = (H + TG - 1) / TG;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.x;     // input * T + t
    const uint T = threadgroups_per_grid.x / N;
    const uint input = r / T;
    threadgroup float scratch[32];
    float v[EPT];
    float ss = 0.0f;
    for (uint i = 0; i < EPT; ++i) {
        const uint e = tid + i * TG;
        v[i] = e < H ? y[(size_t)r * H + e] : 0.0f;
        ss += v[i] * v[i];
    }
    ss = tg_sum(ss, scratch, lane, sg, TG / 32);
    const float inv = metal::precise::rsqrt(ss / H + 1e-6f);
    const uint wrow = SHARED ? 0 : input;
    device bfloat* dst = input == 0 ? out0 : (input == 1 ? out1 : out2);
    for (uint i = 0; i < EPT; ++i) {
        const uint e = tid + i * TG;
        if (e < H) dst[(size_t)(r - input * T) * H + e] = bfloat(v[i] * inv * float(nw[wrow * H + e]));
    }
"""


def norm_rows(y: mx.array, weights: mx.array, shared: bool, stacked: bool = False):
    """``y`` [N, T, H] FP32 -> N arrays [T, H] BF16 (``rms_norm`` with row
    ``n`` of ``weights``, or row 0 for all when ``shared``); with
    ``stacked`` one [N, T, H] array (one projection call serves all rows)."""
    n, tokens, hidden = y.shape
    if stacked:
        outs = ["out0"]
        source = _NORM_SOURCE.replace(
            "device bfloat* dst = input == 0 ? out0 : (input == 1 ? out1 : out2);",
            "device bfloat* dst = out0 + (size_t)input * T * H;",
        )
        shapes = [(n, tokens, hidden)]
    else:
        outs = [f"out{i}" for i in range(n)]
        source = _NORM_SOURCE if n == 3 else _NORM_SOURCE.replace(
            "device bfloat* dst = input == 0 ? out0 : (input == 1 ? out1 : out2);",
            "device bfloat* dst = out0;",
        )
        shapes = [(tokens, hidden)] * n
    k = _kernel(f"mtplx_inkling2_norm_rows_{n}_{int(stacked)}", ["y", "nw"], outs, source, _HEADER)
    return k(
        inputs=[y, weights],
        template=[("H", hidden), ("N", n), ("SHARED", int(shared))],
        grid=(1024 * n * tokens, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=shapes,
        output_dtypes=[mx.bfloat16] * len(outs),
    )


# --------------------------------------------------------------------------
# Projection epilogue: splits, short convs, L2 clips, value norm, log tau
# --------------------------------------------------------------------------

_QKV_SOURCE = """
    // Threadgroups [0, HQ): query head (q and r); [HQ, HQ + HKV): key head;
    // [HQ + HKV, HQ + 2 HKV): value head. 128 threads = one per head dim.
    const uint d = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint g = threadgroup_position_in_grid.x;
    const int T = TOK;
    threadgroup float red[4];
    const int pos0 = info[0];


    if (g < HQ) {
        const uint h = g;
        for (int t = 0; t < T; ++t) {
            float tau = 1.0f;
            if (GLOBAL) {
                tau = 1.0f + LOG_ALPHA * metal::precise::log(
                    max(float(pos0 + t + 1) / LOG_FLOOR, 1.0f));
            }
            float x = float(pq[(size_t)t * NPQ + Q0 + h * D + d]) * float(q_scale[h * D + d]);
            const float nrm = metal::precise::sqrt(tg_sum4(x * x, red, lane, sg));
            x = float(bfloat(x * min(qk_clip[0] / (nrm + 1e-6f), 1.0f)));
            if (GLOBAL) x = float(bfloat(x * tau));
            qo[((size_t)t * HQ + h) * D + d] = bfloat(x);
            float rv = 0.0f;
            if (d < DR) rv = float(pq[(size_t)t * NPQ + R0 + h * DR + d]) * float(r_scale[h * DR + d]);
            const float rn = metal::precise::sqrt(tg_sum4(rv * rv, red, lane, sg));
            if (d < DR) {
                float y = float(bfloat(rv * min(r_clip[0] / (rn + 1e-6f), 1.0f)));
                if (GLOBAL) y = float(bfloat(y * tau));
                ro[((size_t)t * HQ + h) * DR + d] = bfloat(y);
            }
        }
        return;
    }
    const bool is_v = g >= HQ + HKV;
    const uint hk = is_v ? g - HQ - HKV : g - HQ;
    const uint c = hk * D + d;
    // STACKED: pk / pv are the second / third [T, NP] blocks of one array.
    const device bfloat* src = is_v ? pv + (STACKED ? 2 * T * NPV : 0)
                                    : pk + (STACKED ? T * NPK : 0);
    const int np = is_v ? NPV : NPK;
    const int col = is_v ? V0 : K0;
    const device float* st = is_v ? vstate : kstate;
    device float* win = is_v ? vwin : kwin;
    device float* nxt = is_v ? vnext : knext;
    float xs[TOK];
    for (int t = 0; t < T; ++t) xs[t] = float(src[(size_t)t * np + col + c]);
    for (int t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (int lag = 0; lag < KC; ++lag) {
            // lag 0 is the current token: xp index t + K - 1 - lag.
            const int p = t + KC - 1 - lag;
            const float val = p < KC - 1 ? st[p * DKV + c] : xs[p - (KC - 1)];
            float tap;
            if (GLOBAL) {
                tap = float((is_v ? vw : kw)[lag * DKV + c])
                    + float(src[(size_t)t * np + DYN0 + hk * KC + lag]);
            } else {
                // Static conv weights are [C, K] with column K - 1 = current.
                tap = float((is_v ? vw : kw)[c * KC + (KC - 1 - lag)]);
            }
            acc += val * tap;
        }
        float o;
        if (GLOBAL) {
            o = float(bfloat((acc + xs[t]) * cscale[0]));
        } else {
            o = float(bfloat((float(bfloat(acc)) + xs[t]) * cscale[0]));
        }
        float y;
        if (is_v) {
            const float ms = tg_sum4(o * o, red, lane, sg) / D;
            y = o * metal::precise::rsqrt(ms + 1e-6f);
        } else {
            const float nrm = metal::precise::sqrt(tg_sum4(o * o, red, lane, sg));
            y = o * min(qk_clip[0] / (nrm + 1e-6f), 1.0f);
        }
        kvo[(((size_t)(is_v ? 1 : 0) * HKV + hk) * T + t) * D + d] = bfloat(y);
        win[(size_t)(KC - 1 + t) * DKV + c] = xs[t];
    }
    for (int j = 0; j < KC - 1; ++j) {
        win[(size_t)j * DKV + c] = st[j * DKV + c];
        const int p = T + j;
        nxt[j * DKV + c] = p < KC - 1 ? st[p * DKV + c] : xs[p - (KC - 1)];
    }
"""


def qkv_epilogue(attn, pq, pk, pv, kstate, vstate, info, *, tokens: int, layout: dict, stacked: bool = False):
    """From the projection outputs to cache-ready q / k / v.

    Returns ``q`` [T, Hq, D], ``kv`` [2, Hkv, T, D], ``r`` [T, Hq, DR] (all
    BF16) and the conv windows / next states for k and v.
    """
    from . import kernels as K

    glob = not attn.sliding
    hq, hkv, d = attn.n_heads, attn.n_kv, attn.head_dim
    dkv, dr, kc = attn.dkv, attn.d_rel, attn.conv_k
    names = [
        "pq", "pk", "pv", "kstate", "vstate", "info", "q_scale", "r_scale",
        "qk_clip", "r_clip", "kw", "vw", "cscale",
    ]
    arrays = [
        pq, pk, pv, kstate, vstate, info, attn.q_scale, attn.r_scale,
        K._scalar(float(attn._qk_clip.item())), K._scalar(float(attn._r_clip.item())),
    ]
    if glob:
        arrays += [attn.k_static, attn.v_static, K._scalar(attn._dconv_scale)]
    else:
        arrays += [attn.k_conv, attn.v_conv, K._scalar(attn._sconv_scale)]
    tmpl = [
        ("TOK", tokens), ("HQ", hq), ("HKV", hkv), ("D", d), ("DR", dr), ("DKV", dkv),
        ("KC", kc), ("GLOBAL", int(glob)),
        ("NPQ", int(pq.shape[-1])), ("NPK", int(pk.shape[-1])), ("NPV", int(pv.shape[-1])),
        ("Q0", layout["q"]), ("R0", layout["r"]), ("K0", layout["k"]), ("V0", layout["v"]),
        ("DYN0", layout.get("dyn", 0)),
        ("STACKED", int(stacked)),
    ]
    source = _QKV_SOURCE.replace("LOG_ALPHA", f"{float(attn.log_alpha)!r}f").replace(
        "LOG_FLOOR", f"{float(attn.log_floor or 1)!r}f"
    )
    k = _kernel(
        f"mtplx_inkling2_qkv_{int(glob)}_{attn.log_floor or 0}",
        names,
        ["qo", "kvo", "ro", "kwin", "vwin", "knext", "vnext"],
        source,
        _HEADER,
    )
    return k(
        inputs=arrays,
        template=tmpl,
        grid=((hq + 2 * hkv) * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[
            (tokens, hq, d),
            (2, hkv, tokens, d),
            (tokens, hq, dr),
            (kc - 1 + tokens, dkv),
            (kc - 1 + tokens, dkv),
            (kc - 1, dkv),
            (kc - 1, dkv),
        ],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16] + [mx.float32] * 4,
    )


# --------------------------------------------------------------------------
# Attention over the step cache: bias band, sink, gate; T query rows
# --------------------------------------------------------------------------

_ATTN_SOURCE = """
    // One threadgroup per (KV head, token row, key split) serving all G
    // query heads of the group, so each key and value is read once.
    // Simdgroups walk keys; lanes hold D / 32 dims. RING: slot = pos % CAP.
    constexpr int TG = 512;
    constexpr int NSG = TG / 32;
    constexpr int PER = D / 32;
    constexpr int G = HQ / HKV;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int hk = threadgroup_position_in_grid.x;
    const int t = threadgroup_position_in_grid.y;
    const int split = threadgroup_position_in_grid.z;
    const int nsplit = threadgroups_per_grid.z;
    const int P = info[0] + t;
    const int lo = WINDOW > 0 ? max(0, P - WINDOW + 1) : 0;
    const int span = P + 1 - lo;
    const int per = (span + nsplit - 1) / nsplit;
    const int k_begin = lo + split * per;
    const int k_end = min(P + 1, k_begin + per);

    float qv[G][PER], rj[G], m[G], l[G], o[G][PER];
    for (int g = 0; g < G; ++g) {
        const int h = hk * G + g;
        for (int i = 0; i < PER; ++i) {
            qv[g][i] = float(q[((size_t)t * HQ + h) * D + lane * PER + i]) * scale[0];
            o[g][i] = 0.0f;
        }
        rj[g] = lane < DR ? float(r[((size_t)t * HQ + h) * DR + lane]) : 0.0f;
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    const device bfloat* kb = kv + (size_t)hk * CAP * D;
    const device bfloat* vb = kv + ((size_t)HKV + hk) * CAP * D;
    for (int key = k_begin + int(sg); key < k_end; key += NSG) {
        const int slot = RING ? key % CAP : key;
        float kx[PER], vx[PER];
        for (int i = 0; i < PER; ++i) {
            kx[i] = float(kb[(size_t)slot * D + lane * PER + i]);
            vx[i] = float(vb[(size_t)slot * D + lane * PER + i]);
        }
        const int dist = P - key;
        const float rp = (dist < R && lane < DR) ? float(rel_proj[lane * R + dist]) : 0.0f;
        for (int g = 0; g < G; ++g) {
            float s = rj[g] * rp;
            for (int i = 0; i < PER; ++i) s += qv[g][i] * kx[i];
            s = simd_sum(s);
            const float mn = max(m[g], s);
            const float f = metal::fast::exp(m[g] - mn);
            const float p = metal::fast::exp(s - mn);
            l[g] = l[g] * f + p;
            for (int i = 0; i < PER; ++i) o[g][i] = o[g][i] * f + p * vx[i];
            m[g] = mn;
        }
    }
    threadgroup float tm[NSG], tl[NSG], to[NSG * D];
    for (int g = 0; g < G; ++g) {
        const int h = hk * G + g;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) { tm[sg] = m[g]; tl[sg] = l[g]; }
        for (int i = 0; i < PER; ++i) to[sg * D + lane * PER + i] = o[g][i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg != 0) continue;
        float mm = -INFINITY;
        for (int j = 0; j < NSG; ++j) mm = max(mm, tm[j]);
        float ll = 0.0f, acc[PER];
        for (int i = 0; i < PER; ++i) acc[i] = 0.0f;
        for (int j = 0; j < NSG; ++j) {
            const float f = tm[j] == -INFINITY ? 0.0f : metal::fast::exp(tm[j] - mm);
            ll += tl[j] * f;
            for (int i = 0; i < PER; ++i) acc[i] += to[j * D + lane * PER + i] * f;
        }
        //@@SINGLE_BEGIN
        {
            const float sk = float(pg[(size_t)t * NPG + S0 + h]);
            const float top = max(mm, sk);
            const float a = mm == -INFINITY ? 0.0f : metal::fast::exp(mm - top);
            const float den = ll * a + metal::fast::exp(sk - top);
            for (int i = 0; i < PER; ++i) {
                const int c = h * D + lane * PER + i;
                const float val = float(bfloat(acc[i] * a / den));
                const float gt = float(pg[(size_t)t * NPG + G0 + c]);
                out[(size_t)t * HQ * D + c] = bfloat(val * (1.0f / (1.0f + metal::fast::exp(-gt))));
            }
        }
        //@@SINGLE_END
        //@@SPLIT_BEGIN
        {
            const int base = (t * HQ + h) * nsplit + split;
            if (lane == 0) { part_m[base] = mm; part_l[base] = ll; }
            for (int i = 0; i < PER; ++i) part_o[(size_t)base * D + lane * PER + i] = acc[i];
        }
        //@@SPLIT_END
    }
"""

_ATTN_COMBINE_SOURCE = """
    constexpr int PER = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const int h = threadgroup_position_in_grid.x;
    const int t = threadgroup_position_in_grid.y;
    const int base = (t * HQ + h) * NSPLIT;
    const float sk = float(pg[(size_t)t * NPG + S0 + h]);
    float mm = sk;
    for (int j = 0; j < NSPLIT; ++j) mm = max(mm, part_m[base + j]);
    float ll = metal::fast::exp(sk - mm), acc[PER];
    for (int i = 0; i < PER; ++i) acc[i] = 0.0f;
    for (int j = 0; j < NSPLIT; ++j) {
        const float pm = part_m[base + j];
        const float f = pm == -INFINITY ? 0.0f : metal::fast::exp(pm - mm);
        ll += part_l[base + j] * f;
        for (int i = 0; i < PER; ++i) acc[i] += part_o[(size_t)(base + j) * D + lane * PER + i] * f;
    }
    for (int i = 0; i < PER; ++i) {
        const int c = h * D + lane * PER + i;
        const float val = float(bfloat(acc[i] / ll));
        const float gt = float(pg[(size_t)t * NPG + G0 + c]);
        out[(size_t)t * HQ * D + c] = bfloat(val * (1.0f / (1.0f + metal::fast::exp(-gt))));
    }
"""

# Keys per threadgroup split; more splits give the GPU more threadgroups to
# run (24 heads alone leave most of its cores idle).
_KEYS_PER_SPLIT = 512
_MAX_SPLITS = 64


def _drop(source: str, region: str) -> str:
    start = source.index(f"//@@{region}_BEGIN")
    end = source.index(f"//@@{region}_END") + len(f"//@@{region}_END")
    return source[:start] + source[end:]


def attention(attn, q, kv, r, pg, info, *, tokens: int, keys: int, cap: int, ring: bool, gate_col: int, sink_col: int):
    """Gated attention of ``tokens`` query rows over the step cache ``kv``
    [2, Hkv, CAP, D]. ``keys``: valid keys after this step's write (host
    count, picks the split). ``pg``: the projection holding gate and sink.
    Returns [T, Hq * D] BF16."""
    from . import kernels as K

    hq, hkv, d = attn.n_heads, attn.n_kv, attn.head_dim
    window = attn.window
    span = min(keys, window) if window else keys
    nsplit = max(1, min(_MAX_SPLITS, (span + _KEYS_PER_SPLIT - 1) // _KEYS_PER_SPLIT))
    rel_proj = const(("relp", id(attn)), lambda: attn.rel_proj.astype(mx.bfloat16))
    tmpl = [
        ("D", d), ("HQ", hq), ("HKV", hkv), ("DR", attn.d_rel), ("R", int(attn.rel_proj.shape[-1])),
        ("WINDOW", int(window)), ("CAP", cap), ("RING", int(ring)),
        ("NPG", int(pg.shape[-1])), ("G0", gate_col), ("S0", sink_col),
    ]
    names = ["q", "kv", "r", "rel_proj", "pg", "info", "scale"]
    arrays = [q, kv, r, rel_proj, pg, info, K._scalar(float(attn.scale))]
    if nsplit == 1:
        k = _kernel("mtplx_inkling2_step_attn_1", names, ["out"], _drop(_ATTN_SOURCE, "SPLIT"), contiguous=True)
        return k(
            inputs=arrays, template=tmpl,
            grid=(hkv * 512, tokens, 1), threadgroup=(512, 1, 1),
            output_shapes=[(tokens, hq * d)], output_dtypes=[mx.bfloat16],
        )[0]
    k = _kernel("mtplx_inkling2_step_attn_n", names, ["part_m", "part_l", "part_o"], _drop(_ATTN_SOURCE, "SINGLE"))
    pm, pl, po = k(
        inputs=arrays, template=tmpl,
        grid=(hkv * 512, tokens, nsplit), threadgroup=(512, 1, 1),
        output_shapes=[(tokens * hq * nsplit,), (tokens * hq * nsplit,), (tokens * hq * nsplit * d,)],
        output_dtypes=[mx.float32] * 3,
    )
    comb = _kernel("mtplx_inkling2_step_attn_combine", ["part_m", "part_l", "part_o", "pg"], ["out"], _ATTN_COMBINE_SOURCE)
    return comb(
        inputs=[pm, pl, po, pg],
        template=[("D", d), ("HQ", hq), ("NSPLIT", nsplit), ("NPG", int(pg.shape[-1])), ("G0", gate_col), ("S0", sink_col)],
        grid=(hq * 32, tokens, 1), threadgroup=(32, 1, 1),
        output_shapes=[(tokens, hq * d)], output_dtypes=[mx.bfloat16],
    )[0]


# --------------------------------------------------------------------------
# Router: logits + top-k + renormalized weights, one threadgroup per row
# --------------------------------------------------------------------------

_ROUTER_SOURCE = """
    constexpr uint TG = 1024;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint t = threadgroup_position_in_grid.x;
    threadgroup float logits[E];
    // Each simdgroup computes E / 32 expert logits.
    for (uint e = sg; e < E; e += 32) {
        float acc = 0.0f;
        for (uint i = lane; i < H; i += 32) acc += float(x[(size_t)t * H + i]) * float(w[(size_t)e * H + i]);
        acc = simd_sum(acc);
        if (lane == 0) logits[e] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    constexpr uint PER = E / 32;
    float sc[PER], key[PER];
    for (uint j = 0; j < PER; ++j) {
        const uint e = lane + 32 * j;
        const float s = 1.0f / (1.0f + metal::exp(-logits[e]));
        sc[j] = s;
        key[j] = s + bias[e];
    }
    float picked[K];
    uint chosen[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) { best = key[j]; bj = j; }
        }
        const float top = simd_max(best);
        const uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
        const uint win = simd_min(mine);
        float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
        s = simd_sum(s);
        if (lane == win % 32) key[win / 32] = -INFINITY;
        picked[k] = s;
        chosen[k] = win;
        total += s;
    }
    if (lane == 0) {
        const float nrm = route_scale[0] / (total + 1e-20f);
        for (uint k = 0; k < K; ++k) {
            inds[t * K + k] = chosen[k];
            weights[t * K + k] = picked[k] * nrm;
        }
    }
"""


_ROUTER_LOGITS_SOURCE = """
    // One simdgroup per (row, expert): FP32 dot of the BF16 row with the
    // (BF16-valued) router row.
    const uint lane = thread_index_in_simdgroup;
    const uint e = thread_position_in_grid.x / 32;
    const uint t = thread_position_in_grid.y;
    if (e >= E) return;
    float acc = 0.0f;
    for (uint i = lane; i < H; i += 32) acc += float(x[(size_t)t * H + i]) * float(w[(size_t)e * H + i]);
    acc = simd_sum(acc);
    if (lane == 0) logits[t * E + e] = acc;
"""


def route(moe, x: mx.array):
    """``x`` [T, H] BF16 -> (inds [T, K] uint32, weights [T, K] FP32)."""
    from . import kernels as K

    tokens, hidden = x.shape
    experts = int(moe.router_weight.shape[0])
    w = const(("router", id(moe)), lambda: moe.router_weight.astype(mx.bfloat16))
    k = _kernel("mtplx_inkling2_step_router_logits", ["x", "w"], ["logits"], _ROUTER_LOGITS_SOURCE)
    logits = k(
        inputs=[x, w],
        template=[("E", experts), ("H", hidden)],
        grid=(experts * 32, tokens, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, experts)],
        output_dtypes=[mx.float32],
    )[0]
    return K.route_topk(logits, moe.router_bias, moe.top_k, moe.route_scale)


# --------------------------------------------------------------------------
# Small-batch affine quantized matvec (verify widths)
# --------------------------------------------------------------------------
#
# MLX's quantized matmul re-reads the weights per row at these widths (an
# 8-bit [8600, 2304] projection: 62 us for one row, 309 us for eight on an
# M1 Max). Here a simdgroup owns RPS output rows, decodes each weight chunk
# once and applies it to all M input rows.

_QMM_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint n0 = (threadgroup_position_in_grid.x * SGS + sg) * RPS;
    if (n0 >= N) return;
    constexpr uint VPW = 32 / BITS;          // values per uint32
    constexpr uint CHUNK = 8;                // values per lane step
    constexpr uint CHUNKS = K / CHUNK;
    constexpr uint WPC = CHUNK / VPW;        // words per chunk
    constexpr uint GROUPS = K / GS;
    float acc[RPS][M];
    for (int r = 0; r < RPS; ++r)
        for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    for (uint c = lane; c < CHUNKS; c += 32) {
        // Decode this chunk of every owned row once ...
        float wv[RPS][CHUNK];
        const uint gidx = (c * CHUNK) / GS;
        for (int r = 0; r < RPS; ++r) {
            const size_t n = n0 + r;
            const float s = float(scales[n * GROUPS + gidx]);
            const float b = float(biases[n * GROUPS + gidx]);
            const device uint32_t* wp = w + n * (K / VPW) + c * WPC;
            for (uint i = 0; i < WPC; ++i) {
                const uint32_t word = wp[i];
                for (uint j = 0; j < VPW; ++j) {
                    wv[r][i * VPW + j] = float((word >> (j * BITS)) & ((1u << BITS) - 1u)) * s + b;
                }
            }
        }
        // ... then stream the input rows through it.
        for (int m = 0; m < M; ++m) {
            const uint4 xw = *(const device uint4*)(x + (size_t)m * K + c * CHUNK);
            float xv[CHUNK];
            for (int q = 0; q < 4; ++q) {
                xv[2 * q] = as_type<float>(xw[q] << 16);
                xv[2 * q + 1] = as_type<float>(xw[q] & 0xFFFF0000u);
            }
            for (int r = 0; r < RPS; ++r) {
                float t = 0.0f;
                for (uint i = 0; i < CHUNK; ++i) t += wv[r][i] * xv[i];
                acc[r][m] += t;
            }
        }
    }
    for (int r = 0; r < RPS; ++r) {
        for (int m = 0; m < M; ++m) {
            const float v = simd_sum(acc[r][m]);
            if (lane == 0) y[(size_t)m * N + n0 + r] = bfloat(v);
        }
    }
"""

_QMM_RPS = 4
_QMM_SGS = 2


def qmm_small(layer, x: mx.array) -> mx.array:
    """``layer(x)`` for an affine ``nn.QuantizedLinear`` and 2..16 rows of
    BF16 ``x`` [..., K]; other cases call the layer."""
    shape = x.shape
    k = int(shape[-1])
    rows = int(x.size // k)
    mode = getattr(layer, "mode", "affine")
    if (
        rows < 2
        or rows > 8
        or mode != "affine"
        or getattr(layer, "bits", None) not in (4, 8)
        or "biases" not in layer
        or layer.get("bias") is not None
        or x.dtype != mx.bfloat16
        or k % (8 * 32) != 0
    ):
        return layer(x)
    n = int(layer.weight.shape[0])
    rps = _QMM_RPS if rows <= 4 else 2
    if n % (rps * _QMM_SGS):
        return layer(x)
    kern = _kernel("mtplx_inkling2_qmm_small", ["x", "w", "scales", "biases"], ["y"], _QMM_SOURCE)
    (y,) = kern(
        inputs=[x.reshape(rows, k), layer.weight, layer.scales, layer.biases],
        template=[
            ("K", k), ("N", n), ("M", rows), ("BITS", int(layer.bits)), ("GS", int(layer.group_size)),
            ("RPS", rps), ("SGS", _QMM_SGS),
        ],
        grid=((n // (rps * _QMM_SGS)) * _QMM_SGS * 32, 1, 1),
        threadgroup=(_QMM_SGS * 32, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
    )
    return y.reshape(*shape[:-1], n)
