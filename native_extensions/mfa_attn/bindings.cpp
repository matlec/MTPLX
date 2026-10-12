#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mfa_attn.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "metal-flash-attention forward (D=256) and MLX command-buffer limits for MTPLX on M1";
  m.def("mfa_attention_fwd", &mtplx_mfa::mfa_attention_fwd, "q"_a, "k"_a, "v"_a, "n_keys"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        "Non-causal GQA attention over the first n_keys keys; returns (O fp16, L fp32).");
  m.def("inplace_row_write", &mtplx_mfa::inplace_row_write, "dst"_a, "src"_a, "row"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        "Write src into rows [row, row+n) of dst in place; returns an array sharing dst's buffer.");
  m.def("set_command_buffer_limits", &mtplx_mfa::set_command_buffer_limits, "max_ops"_a, "max_mb"_a);
  m.def("get_command_buffer_limits", &mtplx_mfa::get_command_buffer_limits);
}
