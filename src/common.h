// Minimal shared definitions between host and device.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Max number of operators in the jump table
#ifndef GPUOS_MAX_OPS
#define GPUOS_MAX_OPS 128
#endif

// DType codes (must match JIT)
enum DType : int {
  kF32 = 0,
  kF16 = 1,
  kBF16 = 2,
  kI32 = 3,
  kF64 = 4,
};

static constexpr int MAX_NDIM = 8;

// Tensor reference with shape/stride metadata (strides in elements)
struct TensorRef {
  void*      data;
  int        dtype;               // DType code
  int        ndim;                // number of dimensions
  int64_t    sizes[MAX_NDIM];     // per-dimension sizes
  int64_t    strides[MAX_NDIM];   // per-dimension strides (in elements)
};

// Task descriptor (ABI must match JIT code)
struct Task {
  int        op;        // operator id
  int        flags;     // reserved/attributes
  int        ndim;      // output ndim (redundant with out0.ndim; for convenience)
  int64_t    numel;     // total number of output elements
  TensorRef  in0;       // input 0
  TensorRef  in1;       // input 1 (optional)
  TensorRef  out0;      // output 0
};

// Work queue (host producer, device consumers)
struct WorkQueue {
  Task* tasks;     // ring buffer
  int   capacity;  // ring capacity
  int*  head;      // device-side pop index
  int*  tail;      // host-side push index
  int*  quit;      // stop flag (host sets to 1)
};

// Device function pointer signature for operators
using OpFn = void(*)(const Task&);

static_assert(sizeof(void*) == sizeof(unsigned long long), "Assumes 64-bit pointers");
using OpPtrInt = unsigned long long;

// Globals for control (defined in .cu)
extern __device__ __managed__ OpFn g_op_table[GPUOS_MAX_OPS];
extern __device__ __managed__ int  g_op_alias[GPUOS_MAX_OPS];
extern __device__ __managed__ unsigned long long g_processed_count;
