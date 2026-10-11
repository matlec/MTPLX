#include "mfa_attn.h"

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace mtplx_mfa {

namespace {

std::string current_binary_dir() {
  static std::string dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("mtplx_mfa: cannot locate the extension binary");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return dir;
}

constexpr int kHeadDim = 256;
constexpr int kBlockRows = 32;     // parallelization block (gid * 32)
constexpr int kThreads = 128;      // threadgroupSize
constexpr int kThreadgroupMem = 2560;

// Private-member access through explicit instantiation (the standard allows
// naming private members in an explicit instantiation's template arguments).
template <typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag) {
    return M;
  }
};
struct OpsTag {
  using type = int mx::metal::Device::*;
  friend type get(OpsTag);
};
struct MbTag {
  using type = int mx::metal::Device::*;
  friend type get(MbTag);
};
template struct Rob<OpsTag, &mx::metal::Device::max_ops_per_buffer_>;
template struct Rob<MbTag, &mx::metal::Device::max_mb_per_buffer_>;

} // namespace

std::vector<mx::array> mfa_attention_fwd(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    int n_keys,
    mx::StreamOrDevice s) {
  if (q.ndim() != 3 || k.ndim() != 3 || v.ndim() != 3) {
    throw std::invalid_argument("mfa_attention_fwd: q, k, v must be 3-D");
  }
  if (q.shape(2) != kHeadDim || k.shape(2) != kHeadDim || v.shape(2) != kHeadDim) {
    throw std::invalid_argument("mfa_attention_fwd: head_dim must be 256");
  }
  if (q.dtype() != mx::float16 || k.dtype() != mx::float16 || v.dtype() != mx::float16) {
    throw std::invalid_argument("mfa_attention_fwd: fp16 only");
  }
  int hq = q.shape(0), hk = k.shape(0);
  if (hk <= 0 || hq % hk || v.shape(0) != hk || v.shape(1) != k.shape(1)) {
    throw std::invalid_argument("mfa_attention_fwd: bad head layout");
  }
  if (n_keys <= 0 || n_keys > k.shape(1)) {
    throw std::invalid_argument("mfa_attention_fwd: n_keys out of range");
  }
  auto stream = mx::to_stream(s);
  auto qc = mx::contiguous(q, false, stream);
  auto kc = mx::contiguous(k, false, stream);
  auto vc = mx::contiguous(v, false, stream);
  return mx::array::make_arrays(
      {q.shape(), {q.shape(0), q.shape(1)}},
      {mx::float16, mx::float32},
      std::make_shared<MfaAttentionFwd>(stream, n_keys),
      {qc, kc, vc});
}

void MfaAttentionFwd::eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("MfaAttentionFwd has no CPU implementation");
}

void MfaAttentionFwd::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& o = outputs[0];
  auto& l = outputs[1];
  o.set_data(mx::allocator::malloc(o.nbytes()));
  l.set_data(mx::allocator::malloc(l.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto& enc = mx::metal::get_command_encoder(s);
  const int hq_ = q.shape(0), lq_ = q.shape(1), hk_ = k.shape(0);
  const uint32_t rows_c = static_cast<uint32_t>((hq_ / hk_) * lq_);
  const uint32_t keys_c = static_cast<uint32_t>(n_keys_);
  // One pipeline per (rows, keys): MFA's edge handling is specialized on
  // both through function constants. MFA also budgets registers for 1,024
  // threads per threadgroup ("force the occupancy to 1024 on M1").
  static std::unordered_map<uint64_t, MTL::ComputePipelineState*> cache;
  const uint64_t key = (uint64_t(rows_c) << 32) | keys_c;
  MTL::ComputePipelineState* pso = nullptr;
  if (auto it = cache.find(key); it != cache.end()) {
    pso = it->second;
  } else {
    auto lib = d.get_library("mtplx_mfa_attn", current_binary_dir());
    auto consts = MTL::FunctionConstantValues::alloc()->init();
    consts->setConstantValue(&rows_c, MTL::DataTypeUInt, NS::UInteger(0));
    consts->setConstantValue(&keys_c, MTL::DataTypeUInt, NS::UInteger(1));
    NS::Error* err = nullptr;
    auto fn = lib->newFunction(
        NS::String::string("mfa_attention_fwd_d256", NS::UTF8StringEncoding), consts, &err);
    consts->release();
    if (!fn) {
      throw std::runtime_error("mtplx_mfa: kernel specialization failed");
    }
    auto desc = MTL::ComputePipelineDescriptor::alloc()->init();
    desc->setComputeFunction(fn);
    desc->setMaxTotalThreadsPerThreadgroup(1024);
    pso = d.mtl_device()->newComputePipelineState(desc, MTL::PipelineOptionNone, nullptr, &err);
    desc->release();
    fn->release();
    if (!pso) {
      throw std::runtime_error("mtplx_mfa: pipeline creation failed");
    }
    if (cache.size() >= 256) {
      // A long conversation specializes one pipeline per prefill chunk.
      for (auto& [k_, v_] : cache) {
        v_->release();
      }
      cache.clear();
    }
    cache.emplace(key, pso);
  }
  enc.set_compute_pipeline_state(pso);
  enc.set_threadgroup_memory_length(kThreadgroupMem, 0);

  const int hq = q.shape(0), lq = q.shape(1), hk = k.shape(0), cap = k.shape(1);
  const int g = hq / hk;
  const uint32_t rows = static_cast<uint32_t>(g * lq);
  const uint32_t keys = static_cast<uint32_t>(n_keys_);
  const MTL::Size grid((rows + kBlockRows - 1) / kBlockRows, hk, 1);
  const MTL::Size group(kThreads, 1, 1);
  const uint32_t strides[4] = {
      rows * uint32_t(kHeadDim), uint32_t(cap) * uint32_t(kHeadDim), rows, 0};
  enc.set_input_array(q, 0);
  enc.set_input_array(k, 1);
  enc.set_input_array(v, 2);
  enc.set_output_array(o, 3);
  enc.set_output_array(l, 4);
  enc.set_bytes(strides, 4, 5);
  enc.dispatch_threadgroups(grid, group);
}

std::tuple<int, int> get_command_buffer_limits() {
  auto& d = mx::metal::device(mx::Device::gpu);
  return d.get_max_ops_mb_per_buffer();
}

std::tuple<int, int> set_command_buffer_limits(int max_ops, int max_mb) {
  auto& d = mx::metal::device(mx::Device::gpu);
  auto prev = d.get_max_ops_mb_per_buffer();
  if (max_ops > 0) {
    d.*get(OpsTag{}) = max_ops;
  }
  if (max_mb > 0) {
    d.*get(MbTag{}) = max_mb;
  }
  return prev;
}

} // namespace mtplx_mfa
