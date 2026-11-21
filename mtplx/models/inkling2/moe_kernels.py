# SPDX-License-Identifier: Apache-2.0
"""Grouped routed-expert matvecs for Inkling 2 decode and verify rows.

``mx.gather_qmm`` runs one quantized matvec per (token, expert) pair and, at
verify widths, a sort-based path that is slow on M1. Here one threadgroup
owns one distinct expert and a block of output rows: it decodes each weight
word once and applies it to every token routed to that expert, so a verify
block reads each selected expert once.

Carriers: NVFP4 (E2M1 codes, E4M3 scales per 16, per-expert tensor scale)
and MXFP8 (E4M3 codes, E8M0 scales per 32), in MLX's packing.
"""

from __future__ import annotations

import mlx.core as mx

# Rows of one expert's output handled by one threadgroup (one simdgroup each).
_ROWS = 8
# Most tokens one expert can serve in one call (verify blocks are <= 16).
_MAX_TOKENS = 16
# v2 (NVFP4): rows per simdgroup, simdgroups per threadgroup.
_V2_RPS = 4
_V2_SGS = 2

_HEADER = """
inline float e4m3_to_float(uint v) {
    const uint s = v >> 7;
    const uint e = (v >> 3) & 0xF;
    const uint m = v & 0x7;
    float r;
    if (e == 0) {
        r = float(m) * 0.001953125f;  // 2^-9
    } else if (e == 0xF && m == 0x7) {
        r = NAN;
    } else {
        r = ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    }
    return s ? -r : r;
}
inline float e8m0_to_float(uint v) {
    return v == 0xFF ? NAN : ldexp(1.0f, int(v) - 127);
}
constant float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
"""

_SOURCE = """
    // grid: x = ROWS simdgroups per threadgroup over NOUT rows, y = segment.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint seg = threadgroup_position_in_grid.y;
    const uint n = threadgroup_position_in_grid.x * ROWS + sg;
    // DECODE: one token, its K pairs are the segments (seg == pair).
    const int len = DECODE ? 1 : seg_len[seg];
    if (len <= 0 || n >= NOUT) return;
    const int start = DECODE ? int(seg) : seg_start[seg];
    const uint e = DECODE ? uint(seg_expert[seg]) : uint(seg_expert[seg]);

    // A segment longer than MAXT (an expert picked twice by one token) runs
    // in chunks; routing picks distinct experts, so this is a guard.
    for (int base = 0; base < len; base += MAXT) {
    const int clen = min(MAXT, len - base);
    int prow[MAXT];
    size_t xoff[MAXT];
    float acc[MAXT];
    for (int j = 0; j < MAXT; ++j) {
        acc[j] = 0.0f;
        if (j < clen) {
            prow[j] = DECODE ? int(seg) : order[start + base + j];
            // PAIRIN: each pair has its own input row (the down projection
            // reads its activation); otherwise the pair's token row.
            xoff[j] = PAIRIN ? (size_t)prow[j] * KIN
                             : (size_t)(DECODE ? 0 : (prow[j] / TOPK)) * KIN;
        }
    }
    // One uint32 word = 32 / BITS values.
    constexpr uint VPW = 32 / BITS;
    constexpr uint WORDS = KIN / VPW;
    const device uint32_t* wrow = w + ((size_t)e * NOUT + n) * WORDS;
    const device uint8_t* srow = scales + ((size_t)e * NOUT + n) * (KIN / GROUP);
    for (uint wi = lane; wi < WORDS; wi += 32) {
        const uint32_t word = wrow[wi];
        const uint k0 = wi * VPW;
        const float s = FMT == 0 ? e4m3_to_float(srow[k0 / GROUP]) : e8m0_to_float(srow[k0 / GROUP]);
        float v[VPW];
        for (uint i = 0; i < VPW; ++i) {
            const uint code = (word >> (i * BITS)) & ((1u << BITS) - 1u);
            v[i] = (FMT == 0 ? FP4_LUT[code] : e4m3_to_float(code)) * s;
        }
        for (int j = 0; j < MAXT; ++j) {
            if (j >= clen) break;
            const device bfloat* xp = x + xoff[j] + k0;
            float t = 0.0f;
            for (uint i = 0; i < VPW; ++i) t += v[i] * float(xp[i]);
            acc[j] += t;
        }
    }
    // NVFP4 tensor scale (MLX convention: amax / (6 * 448)), BF16 like the
    // reference's ``y * scale.astype(y.dtype)``; POST == 0 leaves it to the
    // caller (the down projection folds it into the routing weights).
    const float post = (FMT == 0 && POST) ? float(bfloat(gscale[e] * (1.0f / 2688.0f))) : 1.0f;
    for (int j = 0; j < MAXT; ++j) {
        if (j >= clen) break;
        float r = simd_sum(acc[j]);
        if (lane == 0) {
            // Rounded like the reference: BF16 matmul output, then the scale.
            y[(size_t)prow[j] * NOUT + n] = float(bfloat(float(bfloat(r)) * post));
        }
    }
    }
"""

_SOURCE_V2 = """
    // NVFP4 grouped matvec, v2: each simdgroup owns RPS output rows; a lane
    // reads one 16-value scale group (8 bytes + 1 scale) per row per step,
    // so one input chunk serves RPS rows. A 256-entry table decodes both
    // nibbles of a byte in one lookup.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    threadgroup half2 lut2[256];
    for (uint i = tid; i < 256; i += SGS * 32) lut2[i] = half2(half(FP4_LUT[i & 15]), half(FP4_LUT[i >> 4]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint seg = threadgroup_position_in_grid.y;
    const uint n0 = (threadgroup_position_in_grid.x * SGS + sg) * RPS;
    const int len = DECODE ? 1 : seg_len[seg];
    if (len <= 0 || n0 >= NOUT) return;
    const int start = DECODE ? int(seg) : seg_start[seg];
    const uint e = uint(seg_expert[seg]);
    constexpr uint GROUPS = KIN / 16;
    constexpr uint WPR = KIN / 8;  // uint32 words per row
    const device uint32_t* wbase = w + ((size_t)e * NOUT + n0) * WPR;
    const device uint8_t* sbase = scales + ((size_t)e * NOUT + n0) * GROUPS;
    for (int base = 0; base < len; base += MAXT) {
        const int clen = min(MAXT, len - base);
        int prow[MAXT];
        size_t xoff[MAXT];
        float acc[RPS][MAXT];
        for (int j = 0; j < MAXT; ++j) {
            for (int r = 0; r < RPS; ++r) acc[r][j] = 0.0f;
            if (j < clen) {
                prow[j] = DECODE ? int(seg) : order[start + base + j];
                xoff[j] = PAIRIN ? (size_t)prow[j] * KIN
                                 : (size_t)(DECODE ? 0 : (prow[j] / TOPK)) * KIN;
            }
        }
        for (uint g = lane; g < GROUPS; g += 32) {
            // Decode this scale group of every owned row once; tokens that
            // share the expert reuse it.
            half2 wd[RPS][8];
            float sc[RPS];
            for (int r = 0; r < RPS; ++r) {
                const uint2 wv = *(const device uint2*)(wbase + (size_t)r * WPR + g * 2);
                sc[r] = e4m3_to_float(sbase[(size_t)r * GROUPS + g]);
                for (int i = 0; i < 4; ++i) {
                    wd[r][i] = lut2[(wv.x >> (8 * i)) & 0xFF];
                    wd[r][4 + i] = lut2[(wv.y >> (8 * i)) & 0xFF];
                }
            }
            for (int j = 0; j < MAXT; ++j) {
                if (j >= clen) break;
                const device uint4* xp = (const device uint4*)(x + xoff[j] + g * 16);
                const uint4 xa = xp[0];
                const uint4 xb = xp[1];
                half2 xh[8];
                for (int q = 0; q < 4; ++q) {
                    xh[q] = half2(as_type<float>(xa[q] << 16), as_type<float>(xa[q] & 0xFFFF0000u));
                    xh[4 + q] = half2(as_type<float>(xb[q] << 16), as_type<float>(xb[q] & 0xFFFF0000u));
                }
                for (int r = 0; r < RPS; ++r) {
                    half2 t2 = half2(0.0h);
                    for (int i = 0; i < 8; ++i) t2 = fma(wd[r][i], xh[i], t2);
                    acc[r][j] += (float(t2.x) + float(t2.y)) * sc[r];
                }
            }
        }
        const float post = POST ? float(bfloat(gscale[e] * (1.0f / 2688.0f))) : 1.0f;
        if (ACTOUT) {
            // Rows alternate gate / up: write the softcapped SwiGLU of each
            // pair (BF16, like the reference activation) instead of y.
            for (int r = 0; r < RPS; r += 2) {
                for (int j = 0; j < MAXT; ++j) {
                    if (j >= clen) break;
                    const float vg = simd_sum(acc[r][j]);
                    const float vu = simd_sum(acc[r + 1][j]);
                    if (lane == 0) {
                        const float g = float(bfloat(float(bfloat(vg)) * post));
                        const float u = float(bfloat(float(bfloat(vu)) * post));
                        const float a = metal::precise::tanh(g / caps[0]) * caps[0]
                            * (1.0f / (1.0f + metal::precise::exp(-g)))
                            * (metal::precise::tanh(u / caps[1]) * caps[1]);
                        y[(size_t)prow[j] * (NOUT / 2) + (n0 + r) / 2] = bfloat(a);
                    }
                }
            }
        } else {
            for (int r = 0; r < RPS; ++r) {
                const uint n = n0 + r;
                for (int j = 0; j < MAXT; ++j) {
                    if (j >= clen) break;
                    const float v = simd_sum(acc[r][j]);
                    // BF16-valued either way (y is FP32 or BF16 by instantiation).
                    if (lane == 0) y[(size_t)prow[j] * NOUT + n] = bfloat(float(bfloat(v)) * post);
                }
            }
        }
    }
"""

_SEG_SOURCE = """
    // One threadgroup of P threads: stable sort of pair ids by expert, then
    // one segment per distinct expert.
    const uint i = thread_position_in_threadgroup.x;
    threadgroup uint sorted_e[256];
    threadgroup int is_start[256];
    const uint ei = experts[i];
    uint rank = 0;
    for (uint j = 0; j < P; ++j) {
        const uint ej = experts[j];
        rank += (ej < ei || (ej == ei && j < i)) ? 1u : 0u;
    }
    order[rank] = int(i);
    sorted_e[rank] = ei;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    is_start[i] = (i == 0 || sorted_e[i] != sorted_e[i - 1]) ? 1 : 0;
    seg_len[i] = 0;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (is_start[i]) {
        int s = 0;
        for (uint j = 0; j < i; ++j) s += is_start[j];
        uint end = i + 1;
        while (end < P && !is_start[end]) ++end;
        seg_start[s] = int(i);
        seg_expert[s] = sorted_e[i];
        seg_len[s] = int(end - i);
    }
"""

_kernels: dict = {}


def _kernel(name: str, inputs, outputs, source, header=""):
    k = _kernels.get(name)
    if k is None:
        k = mx.fast.metal_kernel(
            name=name,
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=header,
        )
        _kernels[name] = k
    return k


def segments(inds: mx.array):
    """``inds`` [T, K] -> (order [P], seg_start [P], seg_len [P], seg_expert [P])."""
    p = inds.size
    if inds.shape[0] == 1:
        # One token: its K experts are distinct, one pair per segment.
        ar = mx.arange(p, dtype=mx.int32)
        return ar, ar, mx.ones((p,), mx.int32), inds.reshape(-1).astype(mx.uint32)
    if p > 256:
        raise ValueError("grouped expert kernel supports at most 256 pairs")
    k = _kernel(
        "mtplx_inkling2_moe_segments",
        ["experts"],
        ["order", "seg_start", "seg_len", "seg_expert"],
        _SEG_SOURCE,
    )
    order, start, length, expert = k(
        inputs=[inds.reshape(-1).astype(mx.uint32)],
        template=[("P", p)],
        grid=(p, 1, 1),
        threadgroup=(p, 1, 1),
        output_shapes=[(p,), (p,), (p,), (p,)],
        output_dtypes=[mx.int32, mx.int32, mx.int32, mx.uint32],
    )
    return order, start, length, expert


_const_cache: dict = {}


def _const(key, make):
    out = _const_cache.get(key)
    if out is None:
        out = make()
        mx.eval(out)
        _const_cache[key] = out
    return out


def grouped_matvec(x, segs, proj, tokens: int, top_k: int, *, pair_input=False, post=True, act_caps=None):
    """``y[p] = scale_e * (W_e @ in_p)`` for every (token, expert) pair ``p``.

    ``in_p`` is the token's row of ``x`` [T, KIN] (BF16), or with
    ``pair_input`` the pair's own row of ``x`` [P, KIN]. ``segs`` from
    ``segments`` (or the
    routing indices themselves for one token). Returns [P, NOUT] FP32 holding
    BF16-rounded values.
    """
    experts, nout, _ = proj.weight.shape
    kin = proj.weight.shape[-1] * 32 // proj.bits
    fmt = {"nvfp4": 0, "mxfp8": 1}[proj.mode]
    decode = tokens == 1
    p = tokens * top_k
    if decode:
        inds = segs
        zero = _const(("zi", 1), lambda: mx.zeros((1,), mx.int32))
        seg_inputs = [zero, zero, zero, inds.reshape(-1)]
    else:
        seg_inputs = list(segs)
    # One expert serves at most ``tokens`` pairs of a step (routing picks
    # distinct experts per token); size the per-thread accumulators to it.
    maxt = 1 if decode else min(tokens, _MAX_TOKENS)
    v2 = fmt == 0 and kin % 32 == 0 and nout % (_V2_RPS * _V2_SGS) == 0
    # act_caps: write the softcapped SwiGLU of gate/up row pairs (BF16) instead
    # of y; v2 only (its simdgroups own whole row pairs).
    act_out = act_caps is not None and v2
    k = _kernel(
        f"mtplx_inkling2_moe_gemv_{fmt}_{int(v2)}",
        ["x", "order", "seg_start", "seg_len", "seg_expert", "w", "scales", "gscale", "caps"],
        ["y"],
        _SOURCE_V2 if v2 else _SOURCE,
        _HEADER,
    )
    gscale = proj.global_scale if fmt == 0 else _const(("zf", 1), lambda: mx.zeros((1,), mx.float32))
    caps = _const(("caps",) + tuple(act_caps or (1.0, 1.0)), lambda: mx.array(list(act_caps or (1.0, 1.0)), mx.float32))
    return k(
        inputs=[x, *seg_inputs, proj.weight, proj.scales, gscale, caps],
        template=[
            ("KIN", kin),
            ("NOUT", nout),
            ("BITS", proj.bits),
            ("GROUP", proj.group_size),
            ("FMT", fmt),
            ("ROWS", _ROWS),
            ("MAXT", maxt),
            ("POST", int(post)),
            ("DECODE", int(decode)),
            ("PAIRIN", int(pair_input)),
            ("TOPK", top_k),
            ("RPS", _V2_RPS),
            ("SGS", _V2_SGS),
            ("ACTOUT", int(act_out)),
        ],
        grid=(
            (nout // (_V2_RPS * _V2_SGS)) * _V2_SGS * 32 if v2 else ((nout + _ROWS - 1) // _ROWS) * _ROWS * 32,
            p,
            1,
        ),
        threadgroup=((_V2_SGS if v2 else _ROWS) * 32, 1, 1),
        output_shapes=[(p, nout // 2) if act_out else (p, nout)],
        output_dtypes=[mx.bfloat16 if act_out else mx.float32],
    )[0]


def supports(proj) -> bool:
    return getattr(proj, "mode", None) in ("nvfp4", "mxfp8")


_ACT_SOURCE = """
    // Softcapped SwiGLU of the interleaved [gate, up] w13 rows -> BF16.
    const uint i = thread_position_in_grid.x;
    const uint p = thread_position_in_grid.y;
    if (i >= I) return;
    const float g = y13[(size_t)p * 2 * I + 2 * i];
    const float u = y13[(size_t)p * 2 * I + 2 * i + 1];
    const float a = metal::precise::tanh(g / caps[0]) * caps[0]
        * (1.0f / (1.0f + metal::precise::exp(-g)))
        * (metal::precise::tanh(u / caps[1]) * caps[1]);
    act[(size_t)p * I + i] = bfloat(a);
"""


def softcapped_swiglu(y13: mx.array, gate_cap: float, up_cap: float) -> mx.array:
    """[P, 2I] FP32 interleaved w13 output -> [P, I] BF16 activation."""
    p, width = y13.shape
    inter = width // 2
    k = _kernel("mtplx_inkling2_moe_act", ["y13", "caps"], ["act"], _ACT_SOURCE)
    caps = _const(("caps", gate_cap, up_cap), lambda: mx.array([gate_cap, up_cap], mx.float32))
    return k(
        inputs=[y13, caps],
        template=[("I", inter)],
        grid=(inter, p, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(p, inter)],
        output_dtypes=[mx.bfloat16],
    )[0]


_COMBINE_SOURCE = """
    // One thread per channel: weighted expert sum -> BF16, then the MLP short
    // conv over the block's tokens (with carried state) and the residual.
    const uint c = thread_position_in_grid.x;
    if (c >= H) return;
    constexpr int KC = KCONV;
    float mt[MAXT];
    for (int t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < TOPK; ++k) {
            const int p = t * TOPK + k;
            float wk = weights[p];
            if (NVFP4) wk *= gscale[inds[p]] * (1.0f / 2688.0f);
            acc += z[(size_t)p * H + c] * wk;
        }
        mt[t] = float(bfloat(acc));
    }
    for (int t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < KC; ++j) {
            const int q = t + j;  // xp index
            const float val = q < KC - 1 ? state[q * H + c] : mt[q - (KC - 1)];
            acc += float(conv_w[c * KC + j]) * val;
        }
        const float m = float(bfloat((float(bfloat(acc)) + mt[t]) * scale[0]));
        h[(size_t)t * H + c] = bfloat(residual[(size_t)t * H + c] + m);
        xp[(size_t)(KC - 1 + t) * H + c] = mt[t];
    }
    for (int j = 0; j < KC - 1; ++j) {
        xp[(size_t)j * H + c] = state[j * H + c];
        const int q = T + j;
        new_state[j * H + c] = q < KC - 1 ? state[q * H + c] : mt[q - (KC - 1)];
    }
"""


_W2_COMBINE_SOURCE = """
    // Decode (one token): down projection of all K pairs for RPS output
    // channels per simdgroup, the weighted expert sum, the MLP short conv
    // with its state and the residual, in one launch.
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    threadgroup half2 lut2[256];
    for (uint i = tid; i < 256; i += SGS * 32) lut2[i] = half2(half(FP4_LUT[i & 15]), half(FP4_LUT[i >> 4]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n0 = (threadgroup_position_in_grid.x * SGS + sg) * RPS;
    if (n0 >= H) return;
    constexpr uint GROUPS = KIN / 16;
    constexpr uint WPR = KIN / 8;
    float m[RPS];
    for (int r = 0; r < RPS; ++r) m[r] = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
        const uint e = inds[k];
        const device uint32_t* wbase = w + ((size_t)e * H + n0) * WPR;
        const device uint8_t* sbase = scales + ((size_t)e * H + n0) * GROUPS;
        const device bfloat* xr = act + (size_t)k * KIN;
        float acc[RPS];
        for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
        for (uint g = lane; g < GROUPS; g += 32) {
            const device uint4* xp = (const device uint4*)(xr + g * 16);
            const uint4 xa = xp[0];
            const uint4 xb = xp[1];
            half2 xh[8];
            for (int q = 0; q < 4; ++q) {
                xh[q] = half2(as_type<float>(xa[q] << 16), as_type<float>(xa[q] & 0xFFFF0000u));
                xh[4 + q] = half2(as_type<float>(xb[q] << 16), as_type<float>(xb[q] & 0xFFFF0000u));
            }
            for (int r = 0; r < RPS; ++r) {
                const uint2 wv = *(const device uint2*)(wbase + (size_t)r * WPR + g * 2);
                half2 t2 = half2(0.0h);
                for (int i = 0; i < 4; ++i) {
                    t2 = fma(lut2[(wv.x >> (8 * i)) & 0xFF], xh[i], t2);
                    t2 = fma(lut2[(wv.y >> (8 * i)) & 0xFF], xh[4 + i], t2);
                }
                acc[r] += (float(t2.x) + float(t2.y)) * e4m3_to_float(sbase[(size_t)r * GROUPS + g]);
            }
        }
        // Routing weight times the NVFP4 tensor scale (as in the combine).
        const float wk = weights[k] * (NVFP4 ? gscale[e] * (1.0f / 2688.0f) : 1.0f);
        for (int r = 0; r < RPS; ++r) m[r] += float(bfloat(simd_sum(acc[r]))) * wk;
    }
    if (lane != 0) return;
    for (int r = 0; r < RPS; ++r) {
        const uint c = n0 + r;
        const float mt = float(bfloat(m[r]));
        float acc = 0.0f;
        for (int j = 0; j < KCONV - 1; ++j) acc += float(conv_w[c * KCONV + j]) * state[j * H + c];
        acc += float(conv_w[c * KCONV + KCONV - 1]) * mt;
        const float mc = float(bfloat((float(bfloat(acc)) + mt) * scale[0]));
        h[c] = bfloat(residual[c] + mc);
        for (int j = 0; j < KCONV - 1; ++j) xp[(size_t)j * H + c] = state[j * H + c];
        xp[(size_t)(KCONV - 1) * H + c] = mt;
        for (int j = 0; j < KCONV - 1; ++j) {
            const int q = 1 + j;
            new_state[j * H + c] = q < KCONV - 1 ? state[q * H + c] : mt;
        }
    }
"""

_W2C_RPS = 4
_W2C_SGS = 2


def moe_block(experts, x, inds, weights, residual, conv_w, state, conv_scale: float):
    """Routed experts + MLP short conv + residual for ``T`` <= 16 tokens of
    one row (``B == 1``).

    ``x`` [T, H] (normalized MLP input), ``inds``/``weights`` [T, K] routing,
    ``residual`` [T, H] FP32, ``conv_w`` [H, Kc], ``state`` [1, Kc-1, H].
    Returns ``(h [T, H] BF16, window [1, Kc-1+T, H], next_state)``.
    """
    from . import kernels

    t, kk = inds.shape
    hidden = x.shape[-1]
    segs = inds if t == 1 else segments(inds)
    act = grouped_matvec(x.astype(mx.bfloat16), segs, experts.w13, t, kk, act_caps=experts.caps)
    if act.dtype != mx.bfloat16:
        act = softcapped_swiglu(act, *experts.caps)
    nvfp4 = experts.w2.mode == "nvfp4"
    kc = conv_w.shape[-1]
    if t == 1 and nvfp4 and hidden % (_W2C_RPS * _W2C_SGS) == 0:
        k2 = _kernel(
            "mtplx_inkling2_moe_w2_combine",
            ["act", "inds", "weights", "w", "scales", "gscale", "residual", "conv_w", "state", "scale"],
            ["h", "xp", "new_state"],
            _W2_COMBINE_SOURCE,
            _HEADER,
        )
        kin = experts.w2.weight.shape[-1] * 8
        return tuple(
            k2(
                inputs=[
                    act, inds.reshape(-1).astype(mx.uint32), weights.reshape(-1).astype(mx.float32),
                    experts.w2.weight, experts.w2.scales, experts.w2.global_scale,
                    residual.astype(mx.float32), conv_w, state.astype(mx.float32),
                    kernels._scalar(float(conv_scale)),
                ],
                template=[
                    ("H", hidden), ("KIN", kin), ("TOPK", kk), ("KCONV", kc), ("NVFP4", 1),
                    ("RPS", _W2C_RPS), ("SGS", _W2C_SGS),
                ],
                grid=((hidden // (_W2C_RPS * _W2C_SGS)) * _W2C_SGS * 32, 1, 1),
                threadgroup=(_W2C_SGS * 32, 1, 1),
                output_shapes=[(1, hidden), (1, kc, hidden), (1, kc - 1, hidden)],
                output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
            )
        )
    z = grouped_matvec(act, segs, experts.w2, t, kk, pair_input=True, post=False)
    z = z.reshape(t * kk, -1)
    k = _kernel(
        "mtplx_inkling2_moe_combine",
        ["z", "weights", "inds", "gscale", "residual", "conv_w", "state", "scale"],
        ["h", "xp", "new_state"],
        _COMBINE_SOURCE,
    )
    gscale = experts.w2.global_scale if nvfp4 else _const(("zf", 1), lambda: mx.zeros((1,), mx.float32))
    return tuple(
        k(
            inputs=[
                z,
                weights.astype(mx.float32),
                inds.astype(mx.uint32),
                gscale,
                residual.astype(mx.float32),
                conv_w,
                state.astype(mx.float32),
                kernels._scalar(float(conv_scale)),
            ],
            template=[
                ("H", hidden),
                ("T", t),
                ("TOPK", kk),
                ("KCONV", kc),
                ("MAXT", max(1, t)),
                ("NVFP4", int(nvfp4)),
            ],
            grid=(hidden, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(t, hidden), (1, kc - 1 + t, hidden), (1, kc - 1, hidden)],
            output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
        )
    )


def experts_forward(experts, x: mx.array, inds: mx.array, weights: mx.array):
    """Drop-in for ``Inkling2Experts.__call__`` on few-token inputs (tests).

    ``x`` [T, H], ``inds`` [T, K], ``weights`` [T, K] FP32 -> [T, H].
    """
    t, kk = inds.shape
    segs = inds if t == 1 else segments(inds)
    y13 = grouped_matvec(x.astype(mx.bfloat16), segs, experts.w13, t, kk)
    act = softcapped_swiglu(y13, *experts.caps)
    z = grouped_matvec(act, segs, experts.w2, t, kk, pair_input=True, post=False)
    z = z.reshape(t, kk, -1)
    s2 = experts.w2.post_scale(inds)
    if s2 is not None:
        weights = weights * s2
    out = mx.sum(z * weights[..., None], axis=-2)
    return out.astype(x.dtype)
