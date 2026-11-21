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
    const int len = seg_len[seg];
    if (len <= 0 || n >= NOUT) return;
    const int start = seg_start[seg];
    const uint e = seg_expert[seg];

    int prow[MAXT];
    const device bfloat* xr[MAXT];
    float acc[MAXT];
    for (int j = 0; j < MAXT; ++j) {
        acc[j] = 0.0f;
        if (j < len) {
            prow[j] = order[start + j];
            xr[j] = x + (size_t)x_row[prow[j]] * KIN;
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
            if (j >= len) break;
            const device bfloat* xp = xr[j] + k0;
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
        if (j >= len) break;
        float r = simd_sum(acc[j]);
        if (lane == 0) {
            // Rounded like the reference: BF16 matmul output, then the scale.
            y[(size_t)prow[j] * NOUT + n] = float(bfloat(float(bfloat(r)) * post));
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


def grouped_matvec(x, x_row, segs, proj, max_tokens: int, post: bool = True):
    """``y[p] = scale_e * (W_e @ x[x_row[p]])`` for every pair ``p``.

    ``x`` [rows, KIN] BF16; ``proj`` an ``ExpertProj`` (NVFP4 or MXFP8).
    Returns [P, NOUT] FP32 holding BF16-rounded values.
    """
    order, start, length, expert = segs
    experts, nout, _ = proj.weight.shape
    kin = proj.weight.shape[-1] * 32 // proj.bits
    fmt = {"nvfp4": 0, "mxfp8": 1}[proj.mode]
    p = order.shape[0]
    maxt = 1 if max_tokens <= 1 else (8 if max_tokens <= 8 else _MAX_TOKENS)
    k = _kernel(
        f"mtplx_inkling2_moe_gemv_{fmt}",
        ["x", "x_row", "order", "seg_start", "seg_len", "seg_expert", "w", "scales", "gscale"],
        ["y"],
        _SOURCE,
        _HEADER,
    )
    gscale = proj.global_scale if fmt == 0 else mx.zeros((1,), mx.float32)
    return k(
        inputs=[x, x_row, order, start, length, expert, proj.weight, proj.scales, gscale],
        template=[
            ("KIN", kin),
            ("NOUT", nout),
            ("BITS", proj.bits),
            ("GROUP", proj.group_size),
            ("FMT", fmt),
            ("ROWS", _ROWS),
            ("MAXT", maxt),
            ("POST", int(post)),
        ],
        grid=(((nout + _ROWS - 1) // _ROWS) * _ROWS * 32, p, 1),
        threadgroup=(_ROWS * 32, 1, 1),
        output_shapes=[(p, nout)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )[0]


def supports(proj) -> bool:
    return getattr(proj, "mode", None) in ("nvfp4", "mxfp8")


def experts_forward(experts, x: mx.array, inds: mx.array, weights: mx.array):
    """Drop-in for ``Inkling2Experts.__call__`` on few-token inputs.

    ``x`` [T, H], ``inds`` [T, K], ``weights`` [T, K] FP32 -> [T, H].
    """
    from .model import _softcapped_swiglu

    t, kk = inds.shape
    segs = segments(inds)
    tok_rows = (mx.arange(t * kk, dtype=mx.int32) // kk)
    y13 = grouped_matvec(x.astype(mx.bfloat16), tok_rows, segs, experts.w13, t)
    act = _softcapped_swiglu(
        y13[:, 0::2].astype(mx.bfloat16),
        y13[:, 1::2].astype(mx.bfloat16),
        experts._gcap,
        experts._ucap,
    )
    pair_rows = mx.arange(t * kk, dtype=mx.int32)
    z = grouped_matvec(act, pair_rows, segs, experts.w2, t, post=False)
    z = z.reshape(t, kk, -1)
    s2 = experts.w2.post_scale(inds)
    if s2 is not None:
        weights = weights * s2
    out = mx.sum(z * weights[..., None], axis=-2)
    return out.astype(x.dtype)
