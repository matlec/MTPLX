# SPDX-License-Identifier: Apache-2.0
"""Fused Metal kernels for Inkling 2 decode and verify rows.

The reference graph (``model.py``) spends most of a decode step launching
small kernels: a looped layer visit mixes its loop messages through ~15 ops
per controlled input. These kernels do the same math in one or two launches
and keep the reference rounding points (BF16 inputs to the low-rank down
projection, FP32 accumulation, a BF16-rounded low-rank state).
"""

from __future__ import annotations

import mlx.core as mx

# --------------------------------------------------------------------------
# Loop controllers: (1 - tanh(forget)) * x + tanh(gain) * message
# --------------------------------------------------------------------------

_MIX_DOWN_SOURCE = """
    // One threadgroup per (input row, rank output); 8 simdgroups split the
    // 2H-long dot. The row RMS is recomputed per threadgroup (L2-resident).
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint input = row / ROWS;
    const device float* xr = x + (size_t)row * H;
    const device float* mr = m + (size_t)row * H;
    threadgroup float red[2][8];

    float sx = 0.0f, sm = 0.0f;
    for (uint h = tid; h < H; h += 256) {
        float a = xr[h], b = mr[h];
        sx += a * a;
        sm += b * b;
    }
    sx = simd_sum(sx);
    sm = simd_sum(sm);
    if (lane == 0) { red[0][sg] = sx; red[1][sg] = sm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tx = 0.0f, tm = 0.0f;
    for (uint i = 0; i < 8; ++i) { tx += red[0][i]; tm += red[1][i]; }
    const float ix = metal::precise::rsqrt(tx / H + 1e-6f);
    const float im = metal::precise::rsqrt(tm / H + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device bfloat* w = down + ((size_t)input * R + k) * (2 * H);
    float acc = 0.0f;
    for (uint h = tid; h < H; h += 256) {
        acc += float(bfloat(xr[h] * ix)) * float(w[h]);
        acc += float(bfloat(mr[h] * im)) * float(w[H + h]);
    }
    acc = simd_sum(acc);
    if (lane == 0) red[0][sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (uint i = 0; i < 8; ++i) t += red[0][i];
        low[(size_t)row * R + k] = float(bfloat(t));
    }
"""

_MIX_UP_SOURCE = """
    // One simdgroup per (row, h): lanes split the rank dimension, so each
    // weight row is read as one contiguous 2R-byte segment.
    const uint lane = thread_index_in_simdgroup;
    const uint h = thread_position_in_grid.x / 32;
    const uint row = thread_position_in_grid.y;
    const uint input = row / ROWS;
    const device float* lr = low + (size_t)row * R;
    const device bfloat* gw = gf + ((size_t)input * 2 * H + h) * R;
    const device bfloat* fw = gw + (size_t)H * R;
    float g = 0.0f, f = 0.0f;
    for (uint k = lane; k < R; k += 32) {
        float l = lr[k];
        g += l * float(gw[k]);
        f += l * float(fw[k]);
    }
    g = simd_sum(g);
    f = simd_sum(f);
    if (lane == 0) {
        const size_t at = (size_t)row * H + h;
        out[at] = (1.0f - metal::precise::tanh(f)) * x[at] + metal::precise::tanh(g) * m[at];
    }
"""

_mix_down_kernel = None
_mix_up_kernel = None


def pack_loop_controllers(down: mx.array, gain_forget: mx.array):
    """``down`` [N, 2, H, r] FP32 and ``gain_forget`` [N, r, 2H] FP32 (BF16
    values) -> kernel layouts ``[N, r, 2H]`` and ``[N, 2H, r]`` in BF16."""
    n, _, hidden, rank = down.shape
    down_t = down.reshape(n, 2 * hidden, rank).transpose(0, 2, 1).astype(mx.bfloat16)
    gf_t = gain_forget.transpose(0, 2, 1).astype(mx.bfloat16)
    return mx.contiguous(down_t), mx.contiguous(gf_t)


def mix_messages(x: mx.array, message: mx.array, down_t: mx.array, gf_t: mx.array):
    """Fused ``model._mix_messages``: ``x``/``message`` [N, B, L, H] FP32."""
    global _mix_down_kernel, _mix_up_kernel
    n, b, length, hidden = x.shape
    rows = b * length
    rank = down_t.shape[1]
    if _mix_down_kernel is None:
        _mix_down_kernel = mx.fast.metal_kernel(
            name="mtplx_inkling2_mix_down",
            input_names=["x", "m", "down"],
            output_names=["low"],
            source=_MIX_DOWN_SOURCE,
        )
        _mix_up_kernel = mx.fast.metal_kernel(
            name="mtplx_inkling2_mix_up",
            input_names=["low", "gf", "x", "m"],
            output_names=["out"],
            source=_MIX_UP_SOURCE,
        )
    xf = x.astype(mx.float32).reshape(n * rows, hidden)
    mf = message.astype(mx.float32).reshape(n * rows, hidden)
    tmpl = [("H", hidden), ("R", rank), ("ROWS", rows)]
    low = _mix_down_kernel(
        inputs=[xf, mf, down_t],
        template=tmpl,
        grid=(rank * 256, n * rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n * rows, rank)],
        output_dtypes=[mx.float32],
    )[0]
    out = _mix_up_kernel(
        inputs=[low, gf_t, xf, mf],
        template=tmpl,
        grid=(hidden * 32, n * rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n * rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape(n, b, length, hidden)
