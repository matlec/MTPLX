#include "ane_ffn.h"

#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mtplx_ane {
namespace {
constexpr uint32_t kSeg = 2560, kBlock = 512;

std::string S(uint64_t v) { return std::to_string(v); }
std::string T(const char* t, uint64_t r, uint64_t w) {
  return std::string("tensor<") + t + ", [1, 1, " + S(r) + ", " + S(w) + "]>";
}

const char* kStageSource = R"(
#include <metal_stdlib>
using namespace metal;
// x [R, H] int8 -> surface rows = channels of segment `off`, columns = tokens: dst[j * ld + r] = src[r * H + off + j]
kernel void ane_stage_x(const device char* src [[buffer(0)]], device char* dst [[buffer(1)]],
                        constant uint4& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const uint r = g.x, j = g.y;  // p: R, H, off, ld
  if (r >= p.x || j >= 2560) return;
  dst[j * p.w + r] = src[r * p.y + p.z + j];
}
// src [rows, src_ld] half -> dst[row * dst_ld + c] = src[row * src_ld + off + c], c < cols
kernel void ane_stage_h(const device half* src [[buffer(0)]], device half* dst [[buffer(1)]],
                        constant uint4& p [[buffer(2)]], constant uint& rows [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]]) {
  const uint c = g.x, row = g.y;  // p: cols, src_ld, off, dst_ld
  if (c >= p.x || row >= rows) return;
  dst[row * p.w + c] = src[row * p.y + p.z + c];
}
// Quantized weight decoding. The kernels take p.z = bits | mode << 8 (mode as mtplx_ane::QuantMode):
// 0 affine: code * scale + bias, scales and biases half;
// 1 mxfp4: e2m1(code) * 2^(scale - 127);  2 nvfp4: e2m1(code) * e4m3(scale);  3 mxfp8: e4m3(code) * 2^(scale - 127).
// Scales arrive as raw bytes (half for affine, uint8 otherwise); the fp modes read no biases.
constant float kE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
inline float e4m3(uint v) {
  const uint e = (v >> 3) & 15u, m = v & 7u;
  const float a = e ? ldexp(1.0f + float(m) * 0.125f, int(e) - 7) : ldexp(float(m) * 0.125f, -6);
  return (v & 128u) ? -a : a;
}
inline float group_scale(const device uchar* sc, uint gi, uint mode) {
  if (mode == 0) return float(((const device half*)sc)[gi]);
  if (mode == 2) return e4m3(sc[gi]);
  return ldexp(1.0f, int(sc[gi]) - 127);
}
inline float group_bias(const device half* bi, uint gi, uint mode) { return mode == 0 ? float(bi[gi]) : 0.0f; }
inline float decode(uint code, float s, float b, uint mode) {
  if (mode == 0) return float(code) * s + b;
  if (mode == 3) return e4m3(code) * s;
  return kE2M1[code & 15u] * s;
}
// Packed quantized weights (bits 4 or 8, per-group scale [and bias], see decode) of `rows` rows of `in_dim`
// inputs, from input `col0`: each threadgroup dequantizes one 512-input block of one row, rotates it with the
// orthonormal 512-point Walsh-Hadamard transform and writes it to dst[row * ld + dcol0 + j].
kernel void ane_prep_w(const device uint* w [[buffer(0)]], const device uchar* sc [[buffer(1)]],
                       const device half* bi [[buffer(2)]], device half* dst [[buffer(3)]],
                       constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  // p: in_dim (packed row length in elements), col0 (first input of block 0), bits | mode << 8, group
  // q: dst ld (elements), dst column of block 0, rows, dst row offset
  threadgroup float v[512];
  const uint row = tg.y, blk = tg.x;
  if (row >= q.z) return;
  const uint bits = p.z & 255u, mode = p.z >> 8;
  const uint per = 32 / bits, words = p.x / per, groups = p.x / p.w;
  for (uint i = t; i < 512; i += 256) {
    const uint col = p.y + blk * 512 + i;
    const uint word = w[row * words + col / per];
    const uint qv = (word >> ((col % per) * bits)) & ((1u << bits) - 1u);
    const uint gi = row * groups + col / p.w;
    v[i] = decode(qv, group_scale(sc, gi, mode), group_bias(bi, gi, mode), mode);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 1; h < 512; h <<= 1) {
    const uint i = (t / h) * 2 * h + (t % h);
    const float a = v[i], b = v[i + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v[i] = a + b; v[i + h] = a - b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float n = 0.04419417382415922f;  // 1/sqrt(512)
  for (uint i = t; i < 512; i += 256) dst[(q.w + row) * q.x + q.y + blk * 512 + i] = half(v[i] * n);
}
kernel void ane_prep_w8(const device uint* w [[buffer(0)]], const device uchar* sc [[buffer(1)]],
                        const device half* bi [[buffer(2)]], device char* dst [[buffer(3)]],
                        constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                        const device half* rs [[buffer(6)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  // As ane_prep_w, then w / rs[row] rounded to int8 (rs = the rotated row's max / 127).
  threadgroup float v[512];
  const uint row = tg.y, blk = tg.x;
  if (row >= q.z) return;
  const uint bits = p.z & 255u, mode = p.z >> 8;
  const uint per = 32 / bits, words = p.x / per, groups = p.x / p.w;
  for (uint i = t; i < 512; i += 256) {
    const uint col = p.y + blk * 512 + i;
    const uint word = w[row * words + col / per];
    const uint qv = (word >> ((col % per) * bits)) & ((1u << bits) - 1u);
    const uint gi = row * groups + col / p.w;
    v[i] = decode(qv, group_scale(sc, gi, mode), group_bias(bi, gi, mode), mode);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 1; h < 512; h <<= 1) {
    const uint i = (t / h) * 2 * h + (t % h);
    const float a = v[i], b = v[i + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v[i] = a + b; v[i + h] = a - b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float n = 0.04419417382415922f / max(float(rs[row]), 1e-12f);
  for (uint i = t; i < 512; i += 256) dst[(q.w + row) * q.x + q.y + blk * 512 + i] = char(clamp(rint(v[i] * n), -127.0f, 127.0f));
}

// Faster weight prep: 64 threads per 512-block, 8 values each. Stages 1,2,4 in registers on 8 consecutive
// values, then a transpose to stride 8 (stages 8,16,32) and to stride 64 (stages 64,128,256).
template <typename OUT>
inline void prep_block(const device uint* w, const device uchar* sc, const device half* bi, device OUT* dst,
                       constant uint4& p, constant uint4& q, float post, uint row, uint blk, uint t,
                       threadgroup float* v, bool to_int8) {
  // The 8 values share one scale: every mode's group (16 to 128) is a multiple of 8.
  const uint bits = p.z & 255u, mode = p.z >> 8;
  const uint per = 32 / bits, words = p.x / per, groups = p.x / p.w;
  float r[8];
  const uint col0 = p.y + blk * 512 + t * 8;
  const uint gi = row * groups + col0 / p.w;
  const float s = group_scale(sc, gi, mode), b = group_bias(bi, gi, mode);
  if (bits == 4) {
    const uint word = w[row * words + col0 / 8];
    for (uint j = 0; j < 8; ++j) r[j] = decode((word >> (4 * j)) & 15u, s, b, mode);
  } else {
    const uint w0 = w[row * words + col0 / 4], w1 = w[row * words + col0 / 4 + 1];
    for (uint j = 0; j < 4; ++j) {
      r[j] = decode((w0 >> (8 * j)) & 255u, s, b, mode);
      r[4 + j] = decode((w1 >> (8 * j)) & 255u, s, b, mode);
    }
  }
  // stages 1, 2, 4 (within the 8 consecutive values)
  for (uint h = 1; h < 8; h <<= 1)
    for (uint j = 0; j < 8; ++j) if ((j & h) == 0) { const float a = r[j], c = r[j + h]; r[j] = a + c; r[j + h] = a - c; }
  for (uint j = 0; j < 8; ++j) v[t * 8 + j] = r[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // stride 8: thread holds base + 8k, base = (t % 8) + (t / 8) * 64
  const uint b8 = (t % 8) + (t / 8) * 64;
  for (uint k = 0; k < 8; ++k) r[k] = v[b8 + 8 * k];
  for (uint h = 1; h < 8; h <<= 1)
    for (uint j = 0; j < 8; ++j) if ((j & h) == 0) { const float a = r[j], c = r[j + h]; r[j] = a + c; r[j + h] = a - c; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0; k < 8; ++k) v[b8 + 8 * k] = r[k];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // stride 64: thread holds t + 64k
  for (uint k = 0; k < 8; ++k) r[k] = v[t + 64 * k];
  for (uint h = 1; h < 8; h <<= 1)
    for (uint j = 0; j < 8; ++j) if ((j & h) == 0) { const float a = r[j], c = r[j + h]; r[j] = a + c; r[j + h] = a - c; }
  device OUT* o = dst + (q.w + row) * q.x + q.y + blk * 512;
  for (uint k = 0; k < 8; ++k) {
    const float val = r[k] * post;
    if (to_int8) o[t + 64 * k] = OUT(clamp(rint(val), -127.0f, 127.0f));
    else o[t + 64 * k] = OUT(val);
  }
}
kernel void ane_prep_w2(const device uint* w [[buffer(0)]], const device uchar* sc [[buffer(1)]],
                        const device half* bi [[buffer(2)]], device half* dst [[buffer(3)]],
                        constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float v[512];
  if (tg.y >= q.z) return;
  prep_block<half>(w, sc, bi, dst, p, q, 0.04419417382415922f, tg.y, tg.x, t, v, false);
}
kernel void ane_prep_w8b(const device uint* w [[buffer(0)]], const device uchar* sc [[buffer(1)]],
                         const device half* bi [[buffer(2)]], device char* dst [[buffer(3)]],
                         constant uint4& p [[buffer(4)]], constant uint4& q [[buffer(5)]],
                         const device half* rs [[buffer(6)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float v[512];
  if (tg.y >= q.z) return;
  prep_block<char>(w, sc, bi, dst, p, q, 0.04419417382415922f / max(float(rs[tg.y]), 1e-12f), tg.y, tg.x, t, v, true);
}
// Input prep: one threadgroup (256 threads) per token row of H = 5120 fp16 values: rotate every 512-block
// (orthonormal Walsh-Hadamard), take the row's peak, quantize to int8 (x / (peak/127)) and write it
// transposed into the segment surfaces (dst_k[j * ld + r]), and the token scale (peak/127 * 128) to tx.
kernel void ane_prep_x(const device half* x [[buffer(0)]], device char* d0 [[buffer(1)]], device char* d1 [[buffer(2)]],
                       device half* txo [[buffer(3)]], constant uint4& p [[buffer(4)]],
                       uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                       uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  // p: rows R, ld (bytes per surface row), 0, 0. H fixed at 5120 = 10 blocks; thread t owns 20 values.
  threadgroup float v[5120];
  threadgroup float pk[8];
  if (r >= p.x) return;
  for (uint i = t; i < 5120; i += 256) v[i] = float(x[r * 5120 + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 1; h < 512; h <<= 1) {
    for (uint pidx = t; pidx < 2560; pidx += 256) {  // 10 blocks x 256 pairs
      const uint blk = pidx / 256, q = pidx % 256;
      const uint i = blk * 512 + (q / h) * 2 * h + (q % h);
      const float a = v[i], c = v[i + h];
      v[i] = a + c; v[i + h] = a - c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float m = 0.0f;
  for (uint i = t; i < 5120; i += 256) m = max(m, fabs(v[i]));
  m = simd_max(m);
  if (sl == 0) pk[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float peak = 0.0f;
  for (uint k = 0; k < 8; ++k) peak = max(peak, pk[k]);
  peak = max(peak * 0.04419417382415922f, 1e-6f);
  const float inv = 127.0f / peak, n = 0.04419417382415922f;
  for (uint i = t; i < 5120; i += 256) {
    const char qv = char(clamp(rint(v[i] * n * inv), -127.0f, 127.0f));
    if (i < 2560) d0[i * p.y + r] = qv; else d1[(i - 2560) * p.y + r] = qv;
  }
  if (t == 0) txo[r] = half(peak / 127.0f * 128.0f);
}
)";
}  // namespace

AneFfn::Target AneFfn::make(uint32_t rows, uint32_t width, bool int8) {
  surfaces_.push_back(make_surface(rows, width, int8));
  Surface* s = surfaces_.back().get();
  auto& d = mx::metal::device(mx::Device::gpu);
  MTL::Buffer* b = d.mtl_device()->newBuffer(s->base, s->size, MTL::ResourceStorageModeShared, nullptr);
  if (!b) throw std::runtime_error("ANE FFN: no Metal buffer over an IOSurface");
  return {s, b};
}

AneFfn::AneFfn(int hidden, int channels, std::vector<int> rows, bool w8)
    : hidden_(hidden), channels_(channels), rows_(std::move(rows)), w8_(w8) {
  if (const std::string why = unavailable(); !why.empty()) throw std::runtime_error("ANE unavailable: " + why);
  if (hidden % kSeg || channels % kBlock || rows_.empty()) throw std::runtime_error("ANE FFN: unsupported shape");
  std::sort(rows_.begin(), rows_.end());
  const uint32_t H = hidden, C = channels, Rm = rows_.back();
  if (Rm % 32) throw std::runtime_error("ANE FFN: rows must be multiples of 32");
  for (uint32_t k = 0; k < H / kSeg; ++k) x_.push_back(make(kSeg, Rm, true));
  tx_ = make(1, Rm, false);
  for (uint32_t k = 0; k < H / kSeg; ++k) { wg_.push_back(make(C, kSeg, w8_)); wu_.push_back(make(C, kSeg, w8_)); }
  for (uint32_t b = 0; b < C; b += kSeg) dseg_.push_back(std::min(kSeg, C - b));
  for (int w : dseg_) wd_.push_back(make(H, w, w8_));
  if (w8_) { sg_ = make(C, 1, false); su_ = make(C, 1, false); sd_ = make(H, 1, false); }
  y_ = make(H + 1, Rm, false);

  std::ostringstream mil;
  mil << "program(1.3)\n{\n";
  for (int R : rows_) {
    std::string P, B;
    auto L = [&](const std::string& t) { B += "        " + t + ";\n"; };
    auto param = [&](const std::string& name, const Target& t, uint32_t width) {
      P += (P.empty() ? "" : ", ") + t.surface->mil_type(width) + " " + name;
      const bool i8 = t.surface->int8;
      L(T(i8 ? "int8" : "fp16", t.surface->rows, width) + " " + name + (i8 ? "_t" : "_d") +
        " = tensor_buffer_to_tensor<ios17>(input = " + name + ")");
      if (i8) L(T("fp16", t.surface->rows, width) + " " + name + "_d = dequantize(input = " + name + "_t, scale = fp16(0x1p-7))");
    };
    for (size_t k = 0; k < x_.size(); ++k) param("x" + S(k), x_[k], R);
    param("tx", tx_, R);
    for (size_t k = 0; k < wg_.size(); ++k) { param("wg" + S(k), wg_[k], kSeg); param("wu" + S(k), wu_[k], kSeg); }
    for (size_t i = 0; i < wd_.size(); ++i) param("wd" + S(i), wd_[i], dseg_[i]);
    if (w8_) { param("sg", sg_, 1); param("su", su_, 1); param("sd", sd_, 1); }
    auto mm = [&](const std::string& o, const std::string& w, const std::string& x, uint64_t h) {
      L(T("fp16", h, R) + " " + o + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = " + w + "_d, y = " + x + ")");
    };
    for (const char* p : {"g", "u"}) {
      std::string tot;
      for (size_t k = 0; k < x_.size(); ++k) {
        const std::string o = std::string(p) + "m" + S(k);
        mm(o, std::string("w") + p + S(k), "x" + S(k) + "_d", C);
        if (k == 0) tot = o;
        else { const std::string n = std::string(p) + "a" + S(k); L(T("fp16", C, R) + " " + n + " = add(x = " + tot + ", y = " + o + ")"); tot = n; }
      }
      if (w8_) { L(T("fp16", C, R) + " " + p + "w = mul(x = " + tot + ", y = s" + p + "_d)"); tot = std::string(p) + "w"; }
      L(T("fp16", C, R) + " " + p + "s = mul(x = " + tot + ", y = tx_d)");
    }
    // silu(g) = g/2 (1 + tanh(g/2)) (Splash: the ANE's tanh is closer than its sigmoid)
    L(T("fp16", C, R) + " gh = mul(x = gs, y = fp16(0x1p-1))");
    L(T("fp16", C, R) + " th = tanh(x = gh)");
    L(T("fp16", C, R) + " tp = add(x = th, y = fp16(0x1p+0))");
    L(T("fp16", C, R) + " si = mul(x = gh, y = tp)");
    L(T("fp16", C, R) + " h = mul(x = si, y = us)");
    L("tensor<fp16, [1, " + S(C) + ", 1, " + S(R) + "]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, " + S(C) + ", 1, " + S(R) + "]))");
    L("tensor<fp16, [" + S(C) + ", " + S(kBlock) + ", 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [" + S(C) +
      ", " + S(kBlock) + ", 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]");
    L("tensor<fp16, [1, " + S(C) + ", 1, " + S(R) + "]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32(" + S(C / kBlock) +
      "), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)");
    L(T("fp16", C, R) + " hr = reshape(x = hr4, shape = tensor<int32, [4]>([1, 1, " + S(C) + ", " + S(R) + "]))");
    L(T("fp16", C, R) + " habs = abs(x = hr)");
    L(T("fp16", 1, R) + " peak = reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))");
    L(T("fp16", 1, R) + " fl = maximum(x = peak, y = fp16(0x1p-9))");
    L(T("fp16", 1, R) + " inv = real_div(x = fp16(0x1.fcp+6), y = fl)");
    L(T("fp16", C, R) + " hs = mul(x = hr, y = inv)");
    L(T("int8", C, R) + " hq = quantize(input = hs, scale = fp16(0x1p+0), output_dtype = string(\"int8\"))");
    L(T("fp16", C, R) + " hd = dequantize(input = hq, scale = fp16(0x1p-7))");
    L(T("fp16", 1, R) + " hscale = mul(x = fl, y = fp16(0x1.0204p+0))");  // 128 / 127
    std::string tot;
    uint32_t beg = 0;
    for (size_t i = 0; i < dseg_.size(); ++i) {
      L(T("fp16", dseg_[i], R) + " hd" + S(i) + " = slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, " + S(beg) +
        ", 0]), size = tensor<int32, [4]>([1, 1, " + S(dseg_[i]) + ", " + S(R) + "]))");
      mm("dm" + S(i), "wd" + S(i), "hd" + S(i), H);
      beg += dseg_[i];
      if (i == 0) tot = "dm0";
      else { const std::string n = "da" + S(i); L(T("fp16", H, R) + " " + n + " = add(x = " + tot + ", y = dm" + S(i) + ")"); tot = n; }
    }
    if (w8_) { L(T("fp16", H, R) + " dw = mul(x = " + tot + ", y = sd_d)"); tot = "dw"; }
    L(T("fp16", H + 1, R) + " yt = concat(axis = int32(2), interleave = bool(false), values = (" + tot + ", hscale))");
    L(y_.surface->mil_type(R) + " y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" +
      y_.surface->mil_strides() + "))");
    mil << "    func ffn" << R << "<ios18>(" << P << ") {\n" << B << "    } -> (y);\n";
  }
  mil << "}\n";
  // weights.bin: one fp16 blob, the 512x512 orthonormal Sylvester Hadamard per channel group.
  std::vector<_Float16> rot(size_t(C) * kBlock);
  const double inv = 1.0 / std::sqrt(double(kBlock));
  for (uint32_t c = 0; c < C; ++c)
    for (uint32_t j = 0; j < kBlock; ++j)
      rot[size_t(c) * kBlock + j] = (_Float16)((__builtin_popcount((c % kBlock) & j) & 1) ? -inv : inv);
  std::vector<uint8_t> blob(128 + rot.size() * 2, 0);
  auto put32 = [&](size_t o, uint32_t v) { std::memcpy(blob.data() + o, &v, 4); };
  auto put64 = [&](size_t o, uint64_t v) { std::memcpy(blob.data() + o, &v, 8); };
  put32(0, 1); put32(4, 2); put32(64, 0xdeadbeef); put32(68, 1); put64(72, rot.size() * 2); put64(80, 128);
  std::memcpy(blob.data() + 128, rot.data(), rot.size() * 2);
  program_ = std::make_unique<Program>(mil.str(), blob);
}

AneFfn::~AneFfn() {
  for (auto* v : {&x_, &wg_, &wu_, &wd_}) for (auto& t : *v) t.buffer->release();
  tx_.buffer->release();
  y_.buffer->release();
  if (w8_) { sg_.buffer->release(); su_.buffer->release(); sd_.buffer->release(); }
}

mx::array AneFfn::stage(const mx::array& xq, const mx::array& tx, const mx::array& wg, const mx::array& wu, const mx::array& wd) {
  auto s = mx::default_stream(mx::Device::gpu);
  return mx::array({1}, mx::int32, std::make_shared<AneStage>(s, this, true),
                   {mx::contiguous(xq, false, s), mx::contiguous(tx, false, s), mx::contiguous(wg, false, s),
                    mx::contiguous(wu, false, s), mx::contiguous(wd, false, s)});
}
// A quantized weight's (w, scales, biases) as the staging kernels read them: affine scales and biases as
// half (a bf16 checkpoint's were read as fp16 bits, so the ANE's weights were garbage and its prefill NaN);
// the fp modes' uint8 scales as they are.
static mx::array quant_part(const mx::array& a, mx::Stream s) {
  return mx::contiguous(mx::issubdtype(a.dtype(), mx::floating) ? mx::astype(a, mx::float16, s) : a, false, s);
}

QuantMode quant_mode(const std::string& name, int bits, int group) {
  // The staging kernels decode 4- and 8-bit codes, one scale per 8 consecutive values.
  struct Spec { const char* name; QuantMode mode; int bits, group; };
  static const Spec fp[] = {{"mxfp4", QuantMode::mxfp4, 4, 32}, {"nvfp4", QuantMode::nvfp4, 4, 16},
                            {"mxfp8", QuantMode::mxfp8, 8, 32}};
  if (name == "affine") {
    if ((bits == 4 || bits == 8) && group >= 8 && group % 8 == 0) return QuantMode::affine;
  } else {
    for (const auto& f : fp)
      if (name == f.name && bits == f.bits && group == f.group) return f.mode;
  }
  throw std::invalid_argument("ANE weight staging: unsupported quantization " + name + " bits=" + S(bits) +
                              " group=" + S(group));
}

mx::array AneFfn::stage_weights_q(const std::vector<mx::array>& g, const std::vector<mx::array>& u,
                                  const std::vector<mx::array>& d, int bits, int group,
                                  const std::vector<mx::array>& row_scales, const std::string& mode) {
  const QuantMode qm = quant_mode(mode, bits, group);
  auto s = mx::default_stream(mx::Device::gpu);
  std::vector<mx::array> in;
  for (auto* v : {&g, &u, &d}) for (auto& a : *v) in.push_back(quant_part(a, s));
  if (w8_) {
    if (row_scales.size() != 3) throw std::runtime_error("ANE FFN int8 weights need gate/up/down row scales");
    // the kernel divides by the scale (in[9..11]); the program multiplies by scale*128 (in[12..14])
    for (auto& a : row_scales) in.push_back(mx::contiguous(mx::astype(a, mx::float16, s), false, s));
    for (auto& a : row_scales) in.push_back(mx::contiguous(mx::astype(mx::multiply(a, mx::array(128.0f), s), mx::float16, s), false, s));
  }
  auto prim = std::make_shared<AneStage>(s, this, true);
  prim->quantized_ = true; prim->bits_ = bits; prim->group_ = group; prim->mode_ = qm;
  return mx::array({1}, mx::int32, prim, std::move(in));
}

mx::array AneFfn::stage_x(const mx::array& x) {
  auto s = mx::default_stream(mx::Device::gpu);
  auto prim = std::make_shared<AneStage>(s, this, false);
  prim->prep_x_ = true;
  return mx::array({1}, mx::int32, prim, {mx::contiguous(mx::astype(x, mx::float16, s), false, s)});
}

mx::array AneFfn::stage_input(const mx::array& xq, const mx::array& tx) {
  auto s = mx::default_stream(mx::Device::gpu);
  return mx::array({1}, mx::int32, std::make_shared<AneStage>(s, this, false),
                   {mx::contiguous(xq, false, s), mx::contiguous(tx, false, s)});
}

int AneFfn::run(int rows) {
  int R = -1;
  for (int r : rows_) if (r >= rows) { R = r; break; }
  if (R < 0) throw std::runtime_error("ANE FFN: chunk wider than the program");
  std::map<std::string, Surface*> in;
  for (size_t k = 0; k < x_.size(); ++k) in["x" + S(k)] = x_[k].surface;
  in["tx"] = tx_.surface;
  for (size_t k = 0; k < wg_.size(); ++k) { in["wg" + S(k)] = wg_[k].surface; in["wu" + S(k)] = wu_[k].surface; }
  for (size_t i = 0; i < wd_.size(); ++i) in["wd" + S(i)] = wd_[i].surface;
  if (w8_) { in["sg"] = sg_.surface; in["su"] = su_.surface; in["sd"] = sd_.surface; }
  program_->run(program_->procedure("ffn" + S(R)), in, y_.surface);
  return R;
}

mx::array AneFfn::output() {
  y_.buffer->retain();
  MTL::Buffer* b = y_.buffer;
  return mx::array(mx::allocator::Buffer(b), {hidden_ + 1, rows_.back()}, mx::float16,
                   [](mx::allocator::Buffer buf) { static_cast<MTL::Buffer*>(buf.ptr())->release(); });
}

void AneStage::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("AneStage has no CPU implementation");
}

void AneStage::eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) {
  out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& enc = mx::metal::get_command_encoder(s);
  auto* lib = d.get_library("mtplx_ane_stage4", [] { return std::string(kStageSource); });
  if (prep_x_) {
    AneFfn& f = *ffn_;
    const uint32_t R = in[0].shape(0);
    if (f.hidden_ != 5120 || f.x_.size() != 2) throw std::runtime_error("ane_prep_x: hidden 5120 only");
    enc.set_compute_pipeline_state(d.get_kernel("ane_prep_x", lib));
    const uint32_t p[4] = {R, f.x_[0].surface->stride, 0, 0};
    enc.set_input_array(in[0], 0);
    enc.set_buffer(f.x_[0].buffer, 1);
    enc.set_buffer(f.x_[1].buffer, 2);
    enc.set_buffer(f.tx_.buffer, 3);
    enc.set_bytes(p, 4, 4);
    enc.dispatch_threadgroups(MTL::Size(R, 1, 1), MTL::Size(256, 1, 1));
    return;
  }
  if (quantized_) {
    // in: gate (w, scales, biases) [C, H], up likewise, down (w, scales, biases) [H, C] (already the ANE's slices);
    // with int8 weights also the rotated rows' scales of gate [C], up [C], down [H].
    AneFfn& f = *ffn_;
    auto* kp = d.get_kernel(f.w8_ ? "ane_prep_w8b" : "ane_prep_w2", lib);
    enc.set_compute_pipeline_state(kp);
    const uint32_t H = f.hidden_, C = f.channels_;
    auto prep = [&](int base, uint32_t rows, uint32_t in_dim, uint32_t col0, const AneFfn::Target& dst, uint32_t blocks) {
      const uint32_t p[4] = {in_dim, col0, uint32_t(bits_) | uint32_t(mode_) << 8, uint32_t(group_)};
      const uint32_t q[4] = {dst.surface->stride / 2, 0, rows, 0};
      enc.set_input_array(in[base], 0);
      enc.set_input_array(in[base + 1], 1);
      enc.set_input_array(in[base + 2], 2);
      enc.set_buffer(dst.buffer, 3);
      enc.set_bytes(p, 4, 4);
      enc.set_bytes(q, 4, 5);
      if (f.w8_) enc.set_input_array(in[9 + base / 3], 6);
      enc.dispatch_threadgroups(MTL::Size(blocks, rows, 1), MTL::Size(64, 1, 1));
    };
    for (size_t k = 0; k < f.wg_.size(); ++k) {
      prep(0, C, H, uint32_t(k * kSeg), f.wg_[k], kSeg / kBlock);
      prep(3, C, H, uint32_t(k * kSeg), f.wu_[k], kSeg / kBlock);
    }
    uint32_t beg = 0;
    for (size_t i = 0; i < f.wd_.size(); ++i) {
      prep(6, H, C, beg, f.wd_[i], f.dseg_[i] / kBlock);
      beg += f.dseg_[i];
    }
    if (f.w8_) {
      // scales x 128 (the program dequantizes int8 weights by 1/128): ane_stage_h copies, so pass pre-scaled arrays.
      enc.set_compute_pipeline_state(d.get_kernel("ane_stage_h", lib));
      const AneFfn::Target* dst[3] = {&f.sg_, &f.su_, &f.sd_};
      for (int j = 0; j < 3; ++j) {
        const uint32_t rows = in[12 + j].shape(0);
        const uint32_t pp[4] = {1, 1, 0, dst[j]->surface->stride / 2};
        enc.set_input_array(in[12 + j], 0);
        enc.set_buffer(dst[j]->buffer, 1);
        enc.set_bytes(pp, 4, 2);
        enc.set_bytes(rows, 3);
        enc.dispatch_threads(MTL::Size(1, rows, 1), MTL::Size(1, 64, 1));
      }
    }
    return;
  }
  auto* kx = d.get_kernel("ane_stage_x", lib);  // (input / fp16-weight staging)
  auto* kh = d.get_kernel("ane_stage_h", lib);
  AneFfn& f = *ffn_;
  const uint32_t R = in[0].shape(0), H = f.hidden_, C = f.channels_;
  if (int(R) > f.max_rows()) throw std::runtime_error("AneStage: chunk wider than the program");
  // x
  enc.set_compute_pipeline_state(kx);
  for (size_t k = 0; k < f.x_.size(); ++k) {
    const uint32_t p[4] = {R, H, uint32_t(k * kSeg), f.x_[k].surface->stride};
    enc.set_input_array(in[0], 0);
    enc.set_buffer(f.x_[k].buffer, 1);
    enc.set_bytes(p, 4, 2);
    enc.dispatch_threads(MTL::Size(R, kSeg, 1), MTL::Size(32, 8, 1));
  }
  enc.set_compute_pipeline_state(kh);
  auto copy_h = [&](const mx::array& src, const AneFfn::Target& dst, uint32_t rows, uint32_t cols, uint32_t src_ld, uint32_t off) {
    const uint32_t p[4] = {cols, src_ld, off, dst.surface->stride / 2};
    enc.set_input_array(src, 0);
    enc.set_buffer(dst.buffer, 1);
    enc.set_bytes(p, 4, 2);
    enc.set_bytes(rows, 3);
    enc.dispatch_threads(MTL::Size(cols, rows, 1), MTL::Size(64, 4, 1));
  };
  copy_h(in[1], f.tx_, 1, R, R, 0);
  if (weights_) {
    for (size_t k = 0; k < f.wg_.size(); ++k) {
      copy_h(in[2], f.wg_[k], C, kSeg, H, uint32_t(k * kSeg));
      copy_h(in[3], f.wu_[k], C, kSeg, H, uint32_t(k * kSeg));
    }
    uint32_t beg = 0;
    for (size_t i = 0; i < f.wd_.size(); ++i) {
      copy_h(in[4], f.wd_[i], H, f.dseg_[i], C, beg);
      beg += f.dseg_[i];
    }
  }
}


// ---------------------------------------------------------------------------------------------------
AneLinear::AneLinear(int hidden, int out, std::vector<int> rows, int slots)
    : hidden_(hidden), out_(out), slots_(slots), rows_(std::move(rows)) {
  if (const std::string why = unavailable(); !why.empty()) throw std::runtime_error("ANE unavailable: " + why);
  seg_ = hidden % int(kSeg) == 0 ? int(kSeg) : (hidden % int(kBlock) == 0 && hidden <= 4096 ? hidden : 0);
  if (!seg_ || out % 64 || rows_.empty() || slots < 1) throw std::runtime_error("ANE linear: unsupported shape");
  const uint32_t kSeg = uint32_t(seg_);  // this projection's segment width (2560 for the dense 27B as before)
  std::sort(rows_.begin(), rows_.end());
  const uint32_t H = hidden, N = out, Rm = rows_.back();
  auto make = [&](uint32_t r, uint32_t w, bool i8) {
    surfaces_.push_back(make_surface(r, w, i8));
    Surface* sf = surfaces_.back().get();
    auto& d = mx::metal::device(mx::Device::gpu);
    MTL::Buffer* b = d.mtl_device()->newBuffer(sf->base, sf->size, MTL::ResourceStorageModeShared, nullptr);
    return AneFfn::Target{sf, b};
  };
  for (uint32_t k = 0; k < H / kSeg; ++k) x_.push_back(make(kSeg, Rm, true));
  tx_ = make(1, Rm, false);
  for (int slot = 0; slot < slots; ++slot)
    for (uint32_t k = 0; k < H / kSeg; ++k) w_.push_back(make(N, kSeg, false));
  y_ = make(N, Rm, false);
  std::ostringstream mil;
  mil << "program(1.3)\n{\n";
  for (int R : rows_) {
    std::string P, B;
    auto L = [&](const std::string& t) { B += "        " + t + ";\n"; };
    auto param = [&](const std::string& name, const AneFfn::Target& t, uint32_t width) {
      P += (P.empty() ? "" : ", ") + t.surface->mil_type(width) + " " + name;
      const bool i8 = t.surface->int8;
      L(T(i8 ? "int8" : "fp16", t.surface->rows, width) + " " + name + (i8 ? "_t" : "_d") + " = tensor_buffer_to_tensor<ios17>(input = " + name + ")");
      if (i8) L(T("fp16", t.surface->rows, width) + " " + name + "_d = dequantize(input = " + name + "_t, scale = fp16(0x1p-7))");
    };
    for (size_t k = 0; k < x_.size(); ++k) param("x" + S(k), x_[k], R);
    param("tx", tx_, R);
    for (size_t k = 0; k < x_.size(); ++k) param("w" + S(k), w_[k], kSeg);
    std::string tot;
    for (size_t k = 0; k < x_.size(); ++k) {
      const std::string o = "m" + S(k);
      L(T("fp16", N, R) + " " + o + " = matmul(transpose_x = bool(false), transpose_y = bool(false), x = w" + S(k) + "_d, y = x" + S(k) + "_d)");
      if (k == 0) tot = o;
      else { const std::string n = "a" + S(k); L(T("fp16", N, R) + " " + n + " = add(x = " + tot + ", y = " + o + ")"); tot = n; }
    }
    L(T("fp16", N, R) + " ys = mul(x = " + tot + ", y = tx_d)");
    L(y_.surface->mil_type(R) + " y = tensor_to_tensor_buffer<ios17>(input = ys, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>(" +
      y_.surface->mil_strides() + "))");
    mil << "    func lin" << R << "<ios18>(" << P << ") {\n" << B << "    } -> (y);\n";
  }
  mil << "}\n";
  std::vector<uint8_t> blob(128, 0);
  uint32_t one = 1, two = 2;
  std::memcpy(blob.data(), &one, 4); std::memcpy(blob.data() + 4, &two, 4);
  program_ = std::make_unique<Program>(mil.str(), blob);
}

AneLinear::~AneLinear() {
  for (auto& t : x_) t.buffer->release();
  for (auto& t : w_) t.buffer->release();
  tx_.buffer->release(); y_.buffer->release();
}

mx::array AneLinear::stage(const mx::array& xq, const mx::array& tx, const std::vector<mx::array>& weights,
                           const std::vector<int>& row_offsets, int bits, int group, int slot,
                           const std::string& mode) {
  if (slot < 0 || slot >= slots_) throw std::runtime_error("ANE linear: weight slot out of range");
  // Input-only stages pass placeholder bits/group; the mode only matters with weights.
  const QuantMode qm = weights.empty() ? QuantMode::affine : quant_mode(mode, bits, group);
  auto s = mx::default_stream(mx::Device::gpu);
  std::vector<mx::array> in = {mx::contiguous(xq, false, s), mx::contiguous(tx, false, s)};
  for (auto& a : weights) in.push_back(quant_part(a, s));
  auto prim = std::make_shared<AneLinStage>(s, this, weights.empty() ? std::vector<int>{} : row_offsets, bits, group,
                                            slot, qm);
  return mx::array({1}, mx::int32, prim, std::move(in));
}

int AneLinear::run(int rows, int slot) {
  int R = -1;
  for (int r : rows_) if (r >= rows) { R = r; break; }
  if (R < 0) throw std::runtime_error("ANE linear: chunk wider than the program");
  if (slot < 0 || slot >= slots_) throw std::runtime_error("ANE linear: weight slot out of range");
  std::map<std::string, Surface*> in;
  const size_t base = size_t(slot) * x_.size();
  for (size_t k = 0; k < x_.size(); ++k) { in["x" + S(k)] = x_[k].surface; in["w" + S(k)] = w_[base + k].surface; }
  in["tx"] = tx_.surface;
  program_->run(program_->procedure("lin" + S(R)), in, y_.surface);
  return R;
}

mx::array AneLinear::output() {
  y_.buffer->retain();
  return mx::array(mx::allocator::Buffer(y_.buffer), {out_, rows_.back()}, mx::float16,
                   [](mx::allocator::Buffer buf) { static_cast<MTL::Buffer*>(buf.ptr())->release(); });
}

void AneLinStage::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("AneLinStage has no CPU implementation");
}

void AneLinStage::eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) {
  out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& enc = mx::metal::get_command_encoder(s);
  auto* lib = d.get_library("mtplx_ane_stage4", [] { return std::string(kStageSource); });
  AneLinear& f = *lin_;
  const uint32_t R = in[0].shape(0), H = f.hidden_;
  const uint32_t kSeg = uint32_t(f.seg_);
  const size_t base = size_t(slot_) * f.x_.size();
  enc.set_compute_pipeline_state(d.get_kernel("ane_stage_x", lib));
  for (size_t k = 0; k < f.x_.size(); ++k) {
    const uint32_t p[4] = {R, H, uint32_t(k * kSeg), f.x_[k].surface->stride};
    enc.set_input_array(in[0], 0);
    enc.set_buffer(f.x_[k].buffer, 1);
    enc.set_bytes(p, 4, 2);
    enc.dispatch_threads(MTL::Size(R, kSeg, 1), MTL::Size(32, 8, 1));
  }
  enc.set_compute_pipeline_state(d.get_kernel("ane_stage_h", lib));
  {
    const uint32_t p[4] = {R, R, 0, f.tx_.surface->stride / 2};
    const uint32_t one = 1;
    enc.set_input_array(in[1], 0);
    enc.set_buffer(f.tx_.buffer, 1);
    enc.set_bytes(p, 4, 2);
    enc.set_bytes(one, 3);
    enc.dispatch_threads(MTL::Size(R, 1, 1), MTL::Size(64, 1, 1));
  }
  enc.set_compute_pipeline_state(d.get_kernel("ane_prep_w", lib));
  // weights: (w, scales, biases) per segment of output rows, landing at row_offsets
  for (size_t j = 0; j < offsets_.size(); ++j) {
    const auto& w = in[2 + 3 * j];
    const uint32_t rows = w.shape(0);
    for (size_t k = 0; k < f.x_.size(); ++k) {
      const AneFfn::Target& wt = f.w_[base + k];
      const uint32_t p[4] = {H, uint32_t(k * kSeg), uint32_t(bits_) | uint32_t(mode_) << 8, uint32_t(group_)};
      const uint32_t q[4] = {wt.surface->stride / 2, 0, rows, uint32_t(offsets_[j])};
      enc.set_input_array(w, 0);
      enc.set_input_array(in[3 + 3 * j], 1);
      enc.set_input_array(in[4 + 3 * j], 2);
      enc.set_buffer(wt.buffer, 3);
      enc.set_bytes(p, 4, 4);
      enc.set_bytes(q, 4, 5);
      enc.dispatch_threadgroups(MTL::Size(kSeg / kBlock, rows, 1), MTL::Size(256, 1, 1));
    }
  }
}

}  // namespace mtplx_ane
