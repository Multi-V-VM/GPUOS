cmake_minimum_required(VERSION 3.24)
project(gpuos_persistent_jit LANGUAGES CXX CUDA)

# Toolchain
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

# If you want mold for host link, this is OK (nvlink is still handled correctly):
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold")
