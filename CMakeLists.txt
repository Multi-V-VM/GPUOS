cmake_minimum_required(VERSION 3.24)
# Respect -DCMAKE_CUDA_ARCHITECTURES (e.g. 120 for RTX 5090).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()
project(gpuos_persistent_jit LANGUAGES CXX CUDA)

# Toolchain
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# Keep runtime headers/libraries with the selected compiler on multi-toolkit hosts.
# An explicit CUDAToolkit_ROOT still takes precedence. Use a fresh build directory
# after switching toolkits; cached CUDA_*_LIBRARY paths may otherwise survive.
if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED CMAKE_CUDA_COMPILER_TOOLKIT_ROOT)
  set(CUDAToolkit_ROOT "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}")
endif()
find_package(CUDAToolkit REQUIRED)
option(GPUOS_ENABLE_TMA "Enable experimental float32 TMA add at operator slot 11 (SM90+)" OFF)
if(GPUOS_ENABLE_TMA)
  if(CUDAToolkit_VERSION VERSION_LESS 13.1)
    message(FATAL_ERROR "The TMA implementation is validated with CUDA 13.1 or newer")
  endif()
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch MATCHES "^[0-9]+" AND CMAKE_MATCH_0 LESS 90)
      message(FATAL_ERROR "TMA requires CUDA architecture 90 or newer")
    endif()
  endforeach()
  add_compile_definitions(GPUOS_ENABLE_TMA=1)
endif()

# Executables: compile CUDA TU directly into targets
add_executable(persistent_jit
  src/host.cpp
  src/persistent_kernel.cu
)
target_include_directories(persistent_jit PRIVATE ${CUDAToolkit_INCLUDE_DIRS} src)
target_link_libraries(persistent_jit PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)
set_target_properties(persistent_jit PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_online_switch
  test/online_switch.cpp
  src/persistent_kernel.cu
)
target_include_directories(test_online_switch PRIVATE ${CUDAToolkit_INCLUDE_DIRS} src)
target_link_libraries(test_online_switch PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)
set_target_properties(test_online_switch PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(test_dual_slot_switch
  test/dual_slot_switch.cpp
  src/persistent_kernel.cu
)
target_include_directories(test_dual_slot_switch PRIVATE ${CUDAToolkit_INCLUDE_DIRS} src)
target_link_libraries(test_dual_slot_switch PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)
set_target_properties(test_dual_slot_switch PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(dispatch_bench experiments/dispatch_bench.cu)
target_link_libraries(dispatch_bench PRIVATE CUDA::cudart)
add_executable(online_bench experiments/online_bench.cu)
target_link_libraries(online_bench PRIVATE CUDA::cudart)
add_executable(online_extended_bench experiments/online_extended_bench.cu)
target_link_libraries(online_extended_bench PRIVATE CUDA::cudart)

add_executable(test_worker_protocol test/worker_protocol.cu src/persistent_kernel.cu)
target_include_directories(test_worker_protocol PRIVATE src)
target_link_libraries(test_worker_protocol PRIVATE CUDA::cudart)
include(CTest)
if(BUILD_TESTING)
  add_test(NAME worker_protocol COMMAND test_worker_protocol)
  set_tests_properties(worker_protocol PROPERTIES TIMEOUT 30)
endif()
if(GPUOS_ENABLE_TMA)
  add_executable(tma_bench experiments/tma_bench.cu)
  target_include_directories(tma_bench PRIVATE src)
  target_link_libraries(tma_bench PRIVATE CUDA::cudart CUDA::cuda_driver)
  add_executable(tma_grid_bench experiments/tma_grid_bench.cu)
  target_include_directories(tma_grid_bench PRIVATE src)
  target_link_libraries(tma_grid_bench PRIVATE CUDA::cudart CUDA::cuda_driver)
  add_executable(test_tma_worker test/tma_worker.cu src/persistent_kernel.cu)
  target_include_directories(test_tma_worker PRIVATE src)
  target_link_libraries(test_tma_worker PRIVATE CUDA::cudart)
  if(BUILD_TESTING)
    add_test(NAME tma_worker COMMAND test_tma_worker)
    add_test(NAME tma_handler_bounded COMMAND test_tma_worker --bounded)
    set_tests_properties(tma_worker tma_handler_bounded PROPERTIES TIMEOUT 30)
  endif()
endif()

# If you want mold for host link, this is OK (nvlink is still handled correctly):
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
