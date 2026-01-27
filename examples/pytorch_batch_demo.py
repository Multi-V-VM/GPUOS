import os
import time
import torch

# Build and import extension in-place
from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)

gpuos_ext = load(
    name='gpuos_ext',
    sources=[
        os.path.join(root, 'pytorch_ext', 'gpuos_ext.cpp'),
        os.path.join(root, 'src', 'persistent_kernel.cu'),
    ],
    extra_cflags=['-O3', '-std=c++17'],
    extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                       '-gencode=arch=compute_120,code=sm_120'],
    extra_ldflags=['-L/usr/local/cuda/lib64', '-lcuda', '-lnvrtc', '-lcudart'],
    with_cuda=True,
    verbose=True,
)

def main():
    device = torch.device('cuda:0')

    # Create tensors BEFORE gpuos init to avoid CUDA context issues
    M = 2000
    N = 4096
    print(f'Creating {M*4} tensors...', flush=True)
    as_list = [torch.randn(N, device=device, dtype=torch.float32) for _ in range(M)]
    bs_list = [torch.randn(N, device=device, dtype=torch.float32) for _ in range(M)]
    outs_add = [torch.empty(N, device=device, dtype=torch.float32) for _ in range(M)]
    outs_mul = [torch.empty(N, device=device, dtype=torch.float32) for _ in range(M)]
    print('Tensors created', flush=True)

    # Initialize GPUOS
    print('Initializing GPUOS...', flush=True)
    gpuos_ext.init(capacity=8192, threads_per_block=256)
    gpuos_ext.peek_queue()  # Sync streams after init
    print('GPUOS ready', flush=True)

    t0 = time.time()

    # Submit small ops interleaved: add then mul
    for i in range(M):
        gpuos_ext.submit_add(as_list[i], bs_list[i], outs_add[i])
        gpuos_ext.submit_mul(as_list[i], bs_list[i], outs_mul[i])

        # Flush in batches of K
        if (i + 1) % 64 == 0:
            gpuos_ext.flush(sync=False)

    # Final flush and wait (uses zero-copy polling, no CUDA sync needed)
    gpuos_ext.flush(sync=True)
    t1 = time.time()

    # Shutdown BEFORE verification to allow safe CUDA operations
    gpuos_ext.shutdown()

    # Spot check (safe now that persistent kernel is stopped)
    ok = True
    for i in range(3):
        a, b = as_list[i], bs_list[i]
        e_add = (a + b).cpu()
        e_mul = (a * b).cpu()
        if not torch.allclose(outs_add[i].cpu(), e_add, rtol=1e-4, atol=1e-4):
            ok = False
        if not torch.allclose(outs_mul[i].cpu(), e_mul, rtol=1e-4, atol=1e-4):
            ok = False
    print('OK' if ok else 'MISMATCH', 'elapsed = %.3f s' % (t1 - t0))

if __name__ == '__main__':
    main()

