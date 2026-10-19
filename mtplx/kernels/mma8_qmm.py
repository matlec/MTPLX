"""M1 verify qmm for 7-8 rows: 4-bit affine on simdgroup MMA, dequantized in registers.

Per lane one whole 32-k group of one row (uint4), scale/bias applied in the dequant (half2 fma),
one MMA chain per n-tile over the whole K range of the simdgroup's part.

Chunk = 128 k = 4 groups (g32). Lane (fm, j): row n0+fm, group 4c+j. Step u (0..15):
A[fm][2j+e] = w(n0+fm, k) with k = 128c + 32j + 8(u/4) + (u%4) + 4e  (nibbles u%4 and u%4+4 of word u/4)
B[r][m]     = x[m, k]   with the same k for r = 2j'+e'.
"""
import mlx.core as mx

_CACHE = {}


def _kernel(nt: int, kp: int, gs: int = 32, bits: int = 4):
    key = (nt, kp, gs, bits)
    if key in _CACHE:
        return _CACHE[key]
    src = f"""
    const uint NT = {nt}, KP = {kp}, GS = {gs}, BITS = {bits};
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tile0 = threadgroup_position_in_grid.x * NT;
    const uint K = xT_shape[0];
    const uint N = w_shape[0];
    const uint KW = K * BITS / 32;                 // packed words per row
    const uint NG = K / GS;
    const uint NC = K / 128;
    const uint qid = lane / 4;
    const uint fm = (qid & 4) + ((lane / 2) % 4);
    const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint j = fn / 2;

    simdgroup_matrix<float, 8, 8> C[NT];
    for (uint t = 0; t < NT; ++t) C[t] = simdgroup_matrix<float, 8, 8>(0.0f);

    const uint c_per = (NC + KP - 1) / KP;
    const uint c0 = sg * c_per;
    const uint c1 = min(NC, c0 + c_per);
    for (uint c = c0; c < c1; ++c) {{
      uint4 wv[NT][BITS / 4];
      half2 sc[NT], bi[NT];
      for (uint t = 0; t < NT; ++t) {{
        const uint n = (tile0 + t) * 8 + fm;
        const uint g = 4 * c + j;                 // 32-k block
        for (uint h = 0; h < BITS / 4; ++h)
          wv[t][h] = *((const device uint4*)(w + n * KW + g * (BITS) + 4 * h));
        const uint sg_i = (g * 32) / GS;           // scale group
        half s = scales[n * NG + sg_i];
        half b = biases[n * NG + sg_i];
        sc[t] = half2(s, s);
        bi[t] = half2(b, b);
      }}
      for (uint u = 0; u < 16; ++u) {{
        // 4-bit: word u/4, nibbles u%4 and u%4+4; 8-bit: word u/2, bytes u%2 and u%2+2.
        const uint k = (BITS == 4)
            ? 128 * c + 32 * (fm / 2) + 8 * (u / 4) + (u % 4) + 4 * (fm % 2)
            : 128 * c + 32 * (fm / 2) + 4 * (u / 2) + (u % 2) + 2 * (fm % 2);
        half2 xv = *((const device half2*)(xT + k * 8 + fn));
        simdgroup_matrix<half, 8, 8> B;
        B.thread_elements()[0] = xv.x;
        B.thread_elements()[1] = xv.y;
        for (uint t = 0; t < NT; ++t) {{
          uint bits;
          if (BITS == 4) {{
            uint word = wv[t][0][u / 4];
            bits = ((word >> (4 * (u % 4))) & 0x000F000Fu) | 0x64006400u;
          }} else {{
            uint word = wv[t][(u / 2) / 4][(u / 2) % 4];
            bits = ((word >> (8 * (u % 2))) & 0x00FF00FFu) | 0x64006400u;
          }}
          half2 a = (as_type<half2>(bits) - half2(1024.0h, 1024.0h)) * sc[t] + bi[t];
          simdgroup_matrix<half, 8, 8> A;
          A.thread_elements()[0] = a.x;
          A.thread_elements()[1] = a.y;
          simdgroup_multiply_accumulate(C[t], A, B, C[t]);
        }}
      }}
    }}
    threadgroup float red[KP][NT][32][2];
    for (uint t = 0; t < NT; ++t) {{ red[sg][t][lane][0] = C[t].thread_elements()[0]; red[sg][t][lane][1] = C[t].thread_elements()[1]; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
      for (uint t = 0; t < NT; ++t) {{
        float v0 = 0.0f, v1 = 0.0f;
        for (uint p = 0; p < KP; ++p) {{ v0 += red[p][t][lane][0]; v1 += red[p][t][lane][1]; }}
        const uint n = (tile0 + t) * 8 + fm;
        y[fn * N + n] = half(v0);
        y[(fn + 1) * N + n] = half(v1);
      }}
    }}
    """
    k = mx.fast.metal_kernel(name=f"mma8v2_nt{nt}_kp{kp}_gs{gs}_b{bits}", input_names=["xT", "w", "scales", "biases"],
                             output_names=["y"], source=src)
    _CACHE[key] = k
    return k


def mma8v2_qmm(x, wq, scales, biases, *, nt=2, kp=4, group_size=32, bits=4):
    m, K = x.shape
    N = wq.shape[0]
    x8 = mx.pad(x, [(0, 8 - m), (0, 0)]) if m < 8 else x
    xT = mx.contiguous(x8.T)
    out = _kernel(nt, kp, group_size, bits)(inputs=[xT, wq, scales, biases], output_shapes=[(8, N)], output_dtypes=[x.dtype],
                          grid=(N // (8 * nt) * 32, kp, 1), threadgroup=(32, kp, 1))[0]
    return out[:m]
