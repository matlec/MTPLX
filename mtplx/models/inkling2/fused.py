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


# --------------------------------------------------------------------------
# Decode attention: relative bias band + sink + output gate in one launch
# --------------------------------------------------------------------------

_DECODE_ATTN_SOURCE = """
    // One threadgroup per (query head, key split); 32 simdgroups, each
    // walking keys with lanes over D (4 values per lane at D = 128).
    constexpr int TG = 1024;
    constexpr int NSG = TG / 32;
    constexpr int PER = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int h = threadgroup_position_in_grid.x;
    const int split = threadgroup_position_in_grid.y;
    const int hk = h / (HQ / HKV);
    const int S = dims[0];
    const int lo = WINDOW > 0 ? max(0, S - WINDOW) : 0;
    const int span = S - lo;
    const int per_split = (span + NSPLIT - 1) / NSPLIT;
    const int k_begin = lo + split * per_split;
    const int k_end = min(S, k_begin + per_split);

    float qv[PER];
    for (int i = 0; i < PER; ++i) {
        qv[i] = float(q[h * D + lane * PER + i]) * scale[0];
    }
    // Lanes < DREL carry one relative-state term each.
    const float rj = lane < DREL ? float(r[h * DREL + lane]) : 0.0f;

    float m = -INFINITY;
    float l = 0.0f;
    float o[PER];
    for (int i = 0; i < PER; ++i) o[i] = 0.0f;
    const device T* kb = k + hk * k_strides[1];
    const device T* vb = v + hk * v_strides[1];
    for (int key = k_begin + int(sg); key < k_end; key += NSG) {
        const device T* kp = kb + key * k_strides[2] + lane * PER;
        float s = 0.0f;
        for (int i = 0; i < PER; ++i) s += qv[i] * float(kp[i]);
        const int d = S - 1 - key;
        if (d < R && lane < DREL) {
            s += rj * float(rel_proj[lane * R + d]);
        }
        s = simd_sum(s);
        const float m_new = max(m, s);
        const float f = metal::fast::exp(m - m_new);
        const float p = metal::fast::exp(s - m_new);
        l = l * f + p;
        const device T* vp = vb + key * v_strides[2] + lane * PER;
        for (int i = 0; i < PER; ++i) o[i] = o[i] * f + p * float(vp[i]);
        m = m_new;
    }
    threadgroup float tm[NSG];
    threadgroup float tl[NSG];
    threadgroup float to[NSG * D];
    if (lane == 0) { tm[sg] = m; tl[sg] = l; }
    for (int i = 0; i < PER; ++i) to[sg * D + lane * PER + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float mm = -INFINITY;
        for (int j = 0; j < NSG; ++j) mm = max(mm, tm[j]);
        float ll = 0.0f;
        float acc[PER];
        for (int i = 0; i < PER; ++i) acc[i] = 0.0f;
        for (int j = 0; j < NSG; ++j) {
            const float f = tm[j] == -INFINITY ? 0.0f : metal::fast::exp(tm[j] - mm);
            ll += tl[j] * f;
            for (int i = 0; i < PER; ++i) acc[i] += to[j * D + lane * PER + i] * f;
        }
        //@@SINGLE_BEGIN
        {
            // Zero-value sink: adds exp(sink) to the denominator only.
            const float sk = float(sink[h]);
            const float top = max(mm, sk);
            const float a = mm == -INFINITY ? 0.0f : metal::fast::exp(mm - top);
            const float denom = ll * a + metal::fast::exp(sk - top);
            for (int i = 0; i < PER; ++i) {
                const int c = h * D + lane * PER + i;
                const float val = float(T(acc[i] * a / denom));
                const float g = float(gate[c]);
                out[c] = T(val * (1.0f / (1.0f + metal::fast::exp(-g))));
            }
        }
        //@@SINGLE_END
        //@@SPLIT_BEGIN
        {
            const int base = (h * NSPLIT + split);
            if (lane == 0) { part_m[base] = mm; part_l[base] = ll; }
            for (int i = 0; i < PER; ++i) part_o[base * D + lane * PER + i] = acc[i];
        }
        //@@SPLIT_END
    }
"""


def _variant(source: str, keep: str, drop: str) -> str:
    """Drop the ``//@@{drop}_BEGIN .. END`` region (Metal sees no undeclared
    outputs)."""
    start = source.index(f"//@@{drop}_BEGIN")
    end = source.index(f"//@@{drop}_END") + len(f"//@@{drop}_END")
    return source[:start] + source[end:]

_DECODE_COMBINE_SOURCE = """
    // One simdgroup per head: merge the key splits, add the sink, gate.
    constexpr int PER = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const int h = threadgroup_position_in_grid.x;
    float mm = float(sink[h]);
    for (int j = 0; j < NSPLIT; ++j) mm = max(mm, part_m[h * NSPLIT + j]);
    float ll = metal::fast::exp(float(sink[h]) - mm);
    float acc[PER];
    for (int i = 0; i < PER; ++i) acc[i] = 0.0f;
    for (int j = 0; j < NSPLIT; ++j) {
        const float pm = part_m[h * NSPLIT + j];
        const float f = pm == -INFINITY ? 0.0f : metal::fast::exp(pm - mm);
        ll += part_l[h * NSPLIT + j] * f;
        for (int i = 0; i < PER; ++i) acc[i] += part_o[(h * NSPLIT + j) * D + lane * PER + i] * f;
    }
    for (int i = 0; i < PER; ++i) {
        const int c = h * D + lane * PER + i;
        const float val = float(T(acc[i] / ll));
        const float g = float(gate[c]);
        out[c] = T(val * (1.0f / (1.0f + metal::fast::exp(-g))));
    }
"""

_decode_attn_kernels: dict = {}
# Keys per split before a global (full) layer splits its keys over more
# threadgroups.
_KEYS_PER_SPLIT = 2048


def decode_attention(q, k, v, r, rel_proj, sink, gate, *, scale: float, window: int):
    """Gated single-token attention for ``B == 1, L == 1``.

    ``q`` [1, Hq, 1, D] (clipped, log-scaled), ``k``/``v`` [1, Hkv, S, D]
    cache views, ``r`` [1, 1, Hq, d_rel], ``rel_proj`` [d_rel, R], ``sink``
    [1, 1, Hq], ``gate`` [1, 1, Hq * D]. Returns ``attention * sigmoid(gate)``
    as [1, 1, Hq * D].
    """
    _, hq, _, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    span = min(s, window) if window else s
    nsplit = 1 if window else max(1, (span + _KEYS_PER_SPLIT - 1) // _KEYS_PER_SPLIT)
    key = ("attn", nsplit > 1)
    kernel = _decode_attn_kernels.get(key)
    outputs = ["out"] if nsplit == 1 else ["part_m", "part_l", "part_o"]
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"mtplx_inkling2_decode_attn_{int(nsplit > 1)}",
            input_names=["q", "k", "v", "r", "rel_proj", "sink", "gate", "dims", "scale"],
            output_names=outputs,
            source=_variant(
                _DECODE_ATTN_SOURCE,
                "SPLIT" if nsplit > 1 else "SINGLE",
                "SINGLE" if nsplit > 1 else "SPLIT",
            ),
            ensure_row_contiguous=False,
        )
        _decode_attn_kernels[key] = kernel
    dtype = q.dtype
    tmpl = [
        ("T", dtype),
        ("D", d),
        ("HQ", hq),
        ("HKV", hkv),
        ("R", rel_proj.shape[-1]),
        ("DREL", rel_proj.shape[0]),
        ("WINDOW", int(window)),
        ("NSPLIT", nsplit),
    ]
    inputs = [
        mx.contiguous(q.reshape(hq, d)),
        k,
        v,
        mx.contiguous(r.reshape(hq, -1)),
        rel_proj,
        mx.contiguous(sink.reshape(hq)),
        mx.contiguous(gate.reshape(hq * d)),
        mx.array([s], dtype=mx.int32),
        mx.array([scale], dtype=mx.float32),
    ]
    if nsplit == 1:
        out = kernel(
            inputs=inputs,
            template=tmpl,
            grid=(hq * 1024, 1, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(hq * d,)],
            output_dtypes=[dtype],
        )[0]
        return out.reshape(1, 1, hq * d)
    part_m, part_l, part_o = kernel(
        inputs=inputs,
        template=tmpl,
        grid=(hq * 1024, nsplit, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(hq * nsplit,), (hq * nsplit,), (hq * nsplit * d,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    combine = _decode_attn_kernels.get("combine")
    if combine is None:
        combine = mx.fast.metal_kernel(
            name="mtplx_inkling2_decode_attn_combine",
            input_names=["part_m", "part_l", "part_o", "sink", "gate"],
            output_names=["out"],
            source=_DECODE_COMBINE_SOURCE,
        )
        _decode_attn_kernels["combine"] = combine
    out = combine(
        inputs=[part_m, part_l, part_o, inputs[5], inputs[6]],
        template=[("T", dtype), ("D", d), ("NSPLIT", nsplit)],
        grid=(hq * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(hq * d,)],
        output_dtypes=[dtype],
    )[0]
    return out.reshape(1, 1, hq * d)
