#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ane_ffn.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Neural Engine share of the prefill FFN for MTPLX on M1 (private AppleNeuralEngine client)";
  m.def("unavailable", &mtplx_ane::unavailable);
  nb::class_<mtplx_ane::AneFfn>(m, "AneFfn")
      .def(nb::init<int, int, std::vector<int>, bool>(), "hidden"_a, "channels"_a, "rows"_a, "w8"_a = false)
      .def("stage", &mtplx_ane::AneFfn::stage, "xq"_a, "tx"_a, "wg"_a, "wu"_a, "wd"_a)
      .def("stage_weights_q", &mtplx_ane::AneFfn::stage_weights_q, "g"_a, "u"_a, "d"_a, "bits"_a, "group"_a,
           "row_scales"_a = std::vector<mx::array>{})
      .def("stage_input", &mtplx_ane::AneFfn::stage_input, "xq"_a, "tx"_a)
      .def("run", &mtplx_ane::AneFfn::run, "rows"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("output", &mtplx_ane::AneFfn::output)
      .def_prop_ro("max_rows", &mtplx_ane::AneFfn::max_rows)
      .def_prop_ro("compile_seconds", &mtplx_ane::AneFfn::compile_seconds);
  nb::class_<mtplx_ane::AneLinear>(m, "AneLinear")
      .def(nb::init<int, int, std::vector<int>>(), "hidden"_a, "out"_a, "rows"_a)
      .def("stage", &mtplx_ane::AneLinear::stage, "xq"_a, "tx"_a, "weights"_a, "row_offsets"_a, "bits"_a, "group"_a)
      .def("run", &mtplx_ane::AneLinear::run, "rows"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("output", &mtplx_ane::AneLinear::output)
      .def_prop_ro("max_rows", &mtplx_ane::AneLinear::max_rows)
      .def_prop_ro("compile_seconds", &mtplx_ane::AneLinear::compile_seconds);
}
