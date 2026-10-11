cmake_minimum_required(VERSION 3.27)

project(mtplx_mfa_attn_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(mtplx_mfa_attn_ext)
target_sources(mtplx_mfa_attn_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/mfa_attn.cpp)
target_include_directories(mtplx_mfa_attn_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mtplx_mfa_attn_ext PUBLIC mlx)

# The kernel uses the air.simdgroup_async_copy intrinsics, which the Metal
# front end no longer accepts as __asm declarations: build_metallib.py emits
# LLVM IR and points the calls at the intrinsics (see its docstring).
set(MFA_METALLIB ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/mtplx_mfa_attn.metallib)
add_custom_command(
  OUTPUT ${MFA_METALLIB}
  COMMAND "${Python_EXECUTABLE}" ${CMAKE_CURRENT_LIST_DIR}/build_metallib.py
          ${CMAKE_CURRENT_LIST_DIR}/mfa_fwd_d256.metal ${MFA_METALLIB}
  DEPENDS ${CMAKE_CURRENT_LIST_DIR}/mfa_fwd_d256.metal ${CMAKE_CURRENT_LIST_DIR}/build_metallib.py)
add_custom_target(mtplx_mfa_attn_metallib DEPENDS ${MFA_METALLIB})
add_dependencies(mtplx_mfa_attn_ext mtplx_mfa_attn_metallib)

nanobind_add_module(_ext NB_STATIC LTO NOMINSIZE NB_DOMAIN mlx
                    ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mtplx_mfa_attn_ext)
if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
