// Minimal client of the private AppleNeuralEngine service (the one behind Core ML): IOSurfaces the GPU
// and the ANE share without a copy, and MIL programs compiled, loaded and evaluated on the ANE.
// Adapted from Splash (incoai/splash, M1/M2 port paperniuk/splash, runtime/ane/Program.mm), Apache-2.0.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mtplx_ane {

struct Surface {
  void* surface = nullptr;  // IOSurfaceRef (retained)
  void* base = nullptr;     // CPU / GPU address of its memory
  uint32_t rows = 0, width = 0, esize = 0, stride = 0;  // stride in bytes, rows padded to 64 bytes
  uint64_t size = 0;
  bool int8 = false;
  ~Surface();
  std::string mil_strides() const;
  std::string mil_type(uint32_t width = 0) const;  // tensor_buffer<...> of its leading `width` columns (0: all)
};
std::unique_ptr<Surface> make_surface(uint32_t rows, uint32_t width, bool int8);

// Why the private interface cannot be used here, or "" when it can.
std::string unavailable();

class Program {
 public:
  // Writes model.mil and weights.bin under the cache directory (keyed by their hash), compiles if the
  // service has no compiled copy, loads, and reads the procedures' input symbols.
  Program(const std::string& mil, const std::vector<uint8_t>& weights);
  ~Program();
  int procedure(const std::string& function) const;
  // Runs `procedure` on the named inputs (every input of the function) into `output`, blocking until
  // the ANE reports. Thread-safe against other Programs; one evaluation at a time per Program.
  void run(int procedure, const std::map<std::string, Surface*>& inputs, Surface* output);
  double compile_seconds() const { return compile_s_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  double compile_s_ = 0;
};

}  // namespace mtplx_ane
