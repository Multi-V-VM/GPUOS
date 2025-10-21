// Ensures CUDA device-link is activated for executables that otherwise
// only link against CUDA static libraries.
__global__ void __cuda_linker_stub_kernel() {}

