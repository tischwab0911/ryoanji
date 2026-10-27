# -----------------------------------------------------------------------------
# extern/ryoanji/CMakeLists.txt
#
# Vendored copy of the Ryoanji Barnes-Hut / FMM N-body solver from SPH-EXA.
# Only the nbody/ and interface/ subtrees are included — the upstream tests
# and demos are not vendored.
#
# Upstream: https://github.com/sphexa-org/SPH-EXA (ryoanji/src/ryoanji/)
# License:  MIT
#
# This CMake file replaces the upstream per-subdirectory CMakeLists.txt files.
# -----------------------------------------------------------------------------

# Header-only target: lets CPU-only builds compile-test the header surface
# (e.g. cartesian_qpole.hpp, traversal_cpu.hpp, kernel.hpp).
add_library(ryoanji_headers INTERFACE)
target_include_directories(ryoanji_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_compile_features(ryoanji_headers INTERFACE cxx_std_20)
target_link_libraries(ryoanji_headers INTERFACE ippl::cstone_headers)
add_library(ippl::ryoanji_headers ALIAS ryoanji_headers)

# GPU library: only when CUDA (or HIP) is the active language. Mirrors the
# upstream split into ryoanji_kernels (warp-level kernels) and ryoanji
# (tree builder + multipole holder + ewald glue).
if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
  set(_RYOANJI_KERNELS_SOURCES
    src/ryoanji/nbody/traversal_gpu.cu
    src/ryoanji/nbody/upsweep_gpu.cu
    src/ryoanji/nbody/heaviside_gpu.cu)

  set(_RYOANJI_GLUE_SOURCES
    src/ryoanji/interface/ewald.cu
    src/ryoanji/interface/multipole_holder.cu
    src/ryoanji/interface/heaviside_holder.cu)

  if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    set_source_files_properties(${_RYOANJI_KERNELS_SOURCES} ${_RYOANJI_GLUE_SOURCES}
      PROPERTIES LANGUAGE HIP)
  endif()

  add_library(ryoanji_kernels OBJECT ${_RYOANJI_KERNELS_SOURCES})
  target_include_directories(ryoanji_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
  target_compile_features(ryoanji_kernels PUBLIC cxx_std_20)
  target_link_libraries(ryoanji_kernels PUBLIC ippl::cstone_gpu)
  # CUDA_SEPARABLE_COMPILATION (-rdc=true) is deliberately OFF: matches
  # sphexa's upstream ryoanji_kernels target, lets nvcc whole-program-inline
  # the inner-loop device helpers in traversal_gpu.cu (applyMAC, traverseWarp,
  # approxAcc, P2P, shflSync, warpMin/Max), and recovers the per-P2P throughput
  # that -rdc otherwise costs us.
  set_target_properties(ryoanji_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

  add_library(ryoanji ${_RYOANJI_GLUE_SOURCES} $<TARGET_OBJECTS:ryoanji_kernels>)
  target_include_directories(ryoanji PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
  target_compile_features(ryoanji PUBLIC cxx_std_20)
  target_link_libraries(ryoanji PUBLIC ippl::cstone_gpu)
  set_target_properties(ryoanji PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

  # ewald.cu / multipole_holder.cu include <mpi.h>
  find_package(MPI COMPONENTS CXX QUIET)
  if(MPI_FOUND)
    target_link_libraries(ryoanji PUBLIC MPI::MPI_CXX)
  endif()

  add_library(ippl::ryoanji_kernels ALIAS ryoanji_kernels)
  add_library(ippl::ryoanji ALIAS ryoanji)
endif()
