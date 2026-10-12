#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mtplx_mfa {

// Non-causal attention of every query row against the first n_keys keys, on the
// metal-flash-attention forward kernel (D = 256, fp16). q [Hq, Lq, 256],
// k / v [Hk, cap, 256] (row-contiguous); the Hq / Hk query heads of one KV head
// run as one block of rows. Returns {O [Hq, Lq, 256] fp16 (normalized),
// L [Hq, Lq] fp32 (the kernel's log-sum-exp)}. Scale is 1 / sqrt(256).
std::vector<mx::array> mfa_attention_fwd(
    const mx::array& q,
    const mx::array& k,
    const mx::array& v,
    int n_keys,
    mx::StreamOrDevice s = {});

class MfaAttentionFwd : public mx::Primitive {
 public:
  MfaAttentionFwd(mx::Stream stream, int n_keys)
      : mx::Primitive(stream), n_keys_(n_keys) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "MfaAttentionFwd";
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    return n_keys_ == static_cast<const MfaAttentionFwd&>(other).n_keys_;
  }

 private:
  int n_keys_;
};

// Write src [.., n, D] into rows [row, row + n) of dst [.., cap, D] IN PLACE and
// return an array that shares dst's buffer (MLX's donated slice_update without
// the donation check). Safe only when every reader of dst reads rows < row
// (a KV cache append): the chained MTP draft keeps its cache buffer referenced
// by in-flight command buffers, so MLX's own slice_update copies the whole
// cache on every depth.
mx::array inplace_row_write(
    const mx::array& dst,
    const mx::array& src,
    int row,
    mx::StreamOrDevice s = {});

class InplaceRowWrite : public mx::Primitive {
 public:
  InplaceRowWrite(mx::Stream stream, int row) : mx::Primitive(stream), row_(row) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;

  const char* name() const override {
    return "InplaceRowWrite";
  }
  bool is_equivalent(const mx::Primitive& other) const override {
    return row_ == static_cast<const InplaceRowWrite&>(other).row_;
  }

 private:
  int row_;
};

// Set MLX's per-command-buffer limits on the default Metal device at runtime
// (MLX reads MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER once, at device
// creation). Returns the previous {ops, mb}.
std::tuple<int, int> set_command_buffer_limits(int max_ops, int max_mb);
std::tuple<int, int> get_command_buffer_limits();

} // namespace mtplx_mfa
