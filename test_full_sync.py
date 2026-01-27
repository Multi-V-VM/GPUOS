#!/usr/bin/env python3
"""
Test the zero-copy synchronization mechanism for GPUOS.

This test demonstrates the reliable completion detection using:
1. Host-mapped memory (zero-copy) for the processed counter
2. __threadfence_system() to ensure memory visibility
3. atomicAdd_system() for system-scope atomic updates
4. Host-side polling without any CUDA API calls

IMPORTANT:
- Create tensors BEFORE importing gpuos_ext to avoid GPU resource conflicts
- Call peek_queue() after init() to sync streams before creating new tensors
- Don't call torch.cuda.synchronize() while persistent kernel is running
- Verify results after shutdown() or use pre-computed expected values
"""
import sys
sys.path.insert(0, 'pytorch_ext')
import torch
import time

print('=== GPUOS Zero-Copy Synchronization Test ===', flush=True)
print('', flush=True)

# Pre-create tensors BEFORE importing gpuos
# This ensures PyTorch is fully initialized before the persistent kernel starts
a = torch.randn(10000, device='cuda')
b = torch.randn(10000, device='cuda')
c = torch.empty(10000, device='cuda')

# Pre-compute expected results
expected_add = (a + b).clone()
expected_mul = (a * b).clone()
print('1. Tensors and expected results created', flush=True)

import gpuos_ext
print('2. gpuos_ext loaded', flush=True)

gpuos_ext.init(1024, 128)
print('3. gpuos_ext init done', flush=True)

# Important: call peek_queue after init to sync streams
state = gpuos_ext.peek_queue()
print(f'4. Initial state: sync_enabled={state.get("sync_enabled")}, sync_ready={state.get("sync_ready")}', flush=True)

# Test 1: Single operation with async flush and zero-copy polling
print('', flush=True)
print('=== Test 1: Async flush + zero-copy polling ===', flush=True)
before = gpuos_ext.sync_poll_processed()
print(f'   Before: sync_processed={before}', flush=True)

gpuos_ext.submit_add(a, b, c)
gpuos_ext.flush(sync=False)  # Don't wait

# Poll for completion using zero-copy (no CUDA API calls!)
start = time.time()
while True:
    sp = gpuos_ext.sync_poll_processed()  # Direct volatile read
    if sp > before:
        elapsed = (time.time() - start) * 1000
        print(f'   Completed in {elapsed:.2f}ms: sync_processed={sp}', flush=True)
        break
    if time.time() - start > 5:
        print('   TIMEOUT!', flush=True)
        break
    time.sleep(0.0001)  # 100us

# Test 2: Sync flush (uses zero-copy polling internally)
print('', flush=True)
print('=== Test 2: Sync flush ===', flush=True)
before = gpuos_ext.sync_poll_processed()
gpuos_ext.submit_mul(a, b, c)
start = time.time()
gpuos_ext.flush(sync=True)  # Wait for completion using zero-copy
elapsed = (time.time() - start) * 1000
after = gpuos_ext.sync_poll_processed()
print(f'   Sync flush took {elapsed:.2f}ms', flush=True)
print(f'   processed: {before} -> {after}', flush=True)

# Test 3: Batch operations
print('', flush=True)
print('=== Test 3: Batch operations (5 ops) ===', flush=True)
before = gpuos_ext.sync_poll_processed()
for i in range(5):
    gpuos_ext.submit_add(a, b, c)
start = time.time()
gpuos_ext.flush(sync=True)
elapsed = (time.time() - start) * 1000
after = gpuos_ext.sync_poll_processed()
print(f'   Batch took {elapsed:.2f}ms', flush=True)
print(f'   Tasks processed: {after - before}', flush=True)

# Final state before shutdown
print('', flush=True)
print('=== Final State ===', flush=True)
state = gpuos_ext.peek_queue()
print(f'   sync_processed: {state.get("sync_processed")}', flush=True)
print(f'   sync_heartbeat: {state.get("sync_heartbeat")}', flush=True)
print(f'   sync_ready: {state.get("sync_ready")}', flush=True)

# Shutdown first, then verify (safe to sync after kernel stops)
gpuos_ext.shutdown()
print('', flush=True)
print('Shutdown complete', flush=True)

# Verify results
print('', flush=True)
print('=== Verification ===', flush=True)
c_final = c.clone()
correct_final = torch.allclose(c_final, expected_add)
print(f'   Final result (should be a+b): {correct_final}', flush=True)

if correct_final:
    print('', flush=True)
    print('All tests passed!', flush=True)
else:
    print('', flush=True)
    print('TESTS FAILED!', flush=True)
    print(f'   Max diff: {(c_final - expected_add).abs().max().item()}', flush=True)
