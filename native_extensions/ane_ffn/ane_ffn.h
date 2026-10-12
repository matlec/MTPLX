// The ANE's share of a SwiGLU FFN in prefill: the trailing C intermediate channels of a layer, computed on
// the Neural Engine as one W16A8 MIL program while the GPU computes the leading channels. See
// mtplx/ane_prefill.py. Program layout adapted from Splash (runtime/ops/AneFfn.cpp, Apache-2.0).
#pragma once
#include <memory>
#include <vector>
#include "ane_client.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace MTL { class Buffer; }

namespace mtplx_ane {

class AneFfn {
 public:
  AneFfn(int hidden, int channels, std::vector<int> rows, bool w8 = false);
  ~AneFfn();
  // Copies the chunk's int8 rotated input [R, H] and per-token scales [R] (x = xq * tx / 128), and the
  // ANE channels' rotated fp16 weights gate/up [C, H] and down [H, C], into the program's surfaces.
  // Returns a 1-element array to evaluate before run().
  mx::array stage(const mx::array& xq, const mx::array& tx, const mx::array& wg, const mx::array& wu,
                  const mx::array& wd);
  // Stages the ANE's channels from the packed affine weights directly: g/u/d are (w, scales, biases) of
  // gate/up [C, H] and down [H, C]; dequantized, rotated in 512-blocks and written in one kernel each.
  mx::array stage_weights_q(const std::vector<mx::array>& g, const std::vector<mx::array>& u,
                            const std::vector<mx::array>& d, int bits, int group,
                            const std::vector<mx::array>& row_scales = {});
  // Rotates, quantizes and stages an fp16 input [R, 5120] in one kernel (replaces the MLX-side prep + stage_input).
  mx::array stage_x(const mx::array& x);
  // Stages only the input (weights kept from an earlier stage of the same layer).
  mx::array stage_input(const mx::array& xq, const mx::array& tx);
  // Runs the function of the smallest row count >= rows; blocks until the ANE reports.
  int run(int rows);
  // The output surface as an array [H + 1, max_rows] fp16: rows 0..H-1 are down's output before the
  // per-token intermediate scale, row H is that scale (y = out[:H] * out[H]).
  mx::array output();
  int max_rows() const { return rows_.back(); }
  double compile_seconds() const { return program_ ? program_->compile_seconds() : 0; }

  // For the staging primitive.
  struct Target { Surface* surface; MTL::Buffer* buffer; };
  std::vector<Target> x_, wg_, wu_, wd_;
  Target tx_{}, y_{}, sg_{}, su_{}, sd_{};
  int hidden_, channels_;
  bool w8_ = false;
  std::vector<int> rows_, dseg_;

 private:
  std::unique_ptr<Program> program_;
  std::vector<std::unique_ptr<Surface>> surfaces_;
  Target make(uint32_t rows, uint32_t width, bool int8);
};

class AneStage : public mx::Primitive {
 public:
  AneStage(mx::Stream s, AneFfn* ffn, bool weights) : mx::Primitive(s), ffn_(ffn), weights_(weights) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  const char* name() const override { return "AneStage"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }
  bool quantized_ = false, prep_x_ = false;
  int bits_ = 4, group_ = 32;

 private:
  AneFfn* ffn_;
  bool weights_;
};

// A plain projection on the ANE: y[N, R] = W x for the rows of a prefill chunk, x the rotated int8 input
// (as for AneFfn) and W fp16 rotated; the output is already scaled by the token scales.
class AneLinear {
 public:
  AneLinear(int hidden, int out, std::vector<int> rows);
  ~AneLinear();
  // weights: (w, scales, biases) triples of packed affine rows; triple j lands at output row row_offsets[j].
  mx::array stage(const mx::array& xq, const mx::array& tx, const std::vector<mx::array>& weights,
                  const std::vector<int>& row_offsets, int bits, int group);
  int run(int rows);
  mx::array output();  // [N, max_rows] fp16
  int max_rows() const { return rows_.back(); }
  double compile_seconds() const { return program_ ? program_->compile_seconds() : 0; }
  std::vector<AneFfn::Target> x_, w_;
  AneFfn::Target tx_{}, y_{};
  int hidden_, out_;
  std::vector<int> rows_;

 private:
  std::unique_ptr<Program> program_;
  std::vector<std::unique_ptr<Surface>> surfaces_;
};

class AneLinStage : public mx::Primitive {
 public:
  AneLinStage(mx::Stream s, AneLinear* lin, std::vector<int> offsets, int bits, int group)
      : mx::Primitive(s), lin_(lin), offsets_(std::move(offsets)), bits_(bits), group_(group) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override;
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override;
  const char* name() const override { return "AneLinStage"; }
  bool is_equivalent(const mx::Primitive&) const override { return false; }

 private:
  AneLinear* lin_;
  std::vector<int> offsets_;
  int bits_, group_;
};

}  // namespace mtplx_ane
