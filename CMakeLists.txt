cmake_minimum_required(VERSION 3.18)
project(gpuos_persistent_jit LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Require relocatable device code for device function pointers
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# NVRTC and CUDA Driver APIs
find_package(CUDAToolkit REQUIRED)

add_executable(persistent_jit
  src/host.cpp
  src/persistent_kernel.cu
  src/common.h
)

target_include_directories(persistent_jit PRIVATE ${CUDAToolkit_INCLUDE_DIRS} src)

set_target_properties(persistent_jit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(persistent_jit PRIVATE
  CUDA::cuda_driver
  CUDA::nvrtc
  CUDA::cudart
)

# Test: online switch mid-run (add -> mul in slot 0)
add_executable(test_online_switch
  test/online_switch.cpp
  src/persistent_kernel.cu
  src/common.h
)

set_target_properties(test_online_switch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_online_switch PRIVATE ${CUDAToolkit_INCLUDE_DIRS} src)
target_link_libraries(test_online_switch PRIVATE
  CUDA::cuda_driver
  CUDA::nvrtc
  CUDA::cudart
)

# Test: dual-slot master/backup alias switching with rollback
add_executable(test_dual_slot_switch
  test/dual_slot_switch.cpp
  src/persistent_kernel.cu
  src/common.h
)

set_target_properties(test_dual_slot_switch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(test_dual_slot_switch PRIVATE ${CUDAToolkit_INCLUDE_DIRS} src)
target_link_libraries(test_dual_slot_switch PRIVATE
  CUDA::cuda_driver
  CUDA::nvrtc
  CUDA::cudart
)
