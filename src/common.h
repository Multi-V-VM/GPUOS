// Minimal shared definitions between host and device.

#pragma once

#include <cuda_runtime.h>

// Max number of operators in the jump table
#ifndef GPUOS_MAX_OPS
#define GPUOS_MAX_OPS 128
#endif

// Task descriptor (ABI must match JIT code)
struct Task {
  int   op;     // operator id
  int   n;      // number of elements
  void* in0;    // input 0
  void* in1;    // input 1 (optional)
  void* out0;   // output 0
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
