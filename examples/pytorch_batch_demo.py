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
    extra_cuda_cflags=['-O3', '-std=c++17', '-rdc=true'],
    extra_ldflags=['-lcuda', '-lnvrtc', '-lcudart'],
    with_cuda=True,
    verbose=True,
)

def main():
    torch.cuda.init()
    device = torch.device('cuda:0')
    gpuos_ext.init(capacity=8192, threads_per_block=256)

    # micro requests
    M = 2000
    N = 4096
    as_list = [torch.randn(N, device=device, dtype=torch.float32) for _ in range(M)]
    bs_list = [torch.randn(N, device=device, dtype=torch.float32) for _ in range(M)]
    outs_add = [torch.empty(N, device=device, dtype=torch.float32) for _ in range(M)]
    outs_mul = [torch.empty(N, device=device, dtype=torch.float32) for _ in range(M)]

    t0 = time.time()

    # Submit small ops interleaved: add then mul
    for i in range(M):
        gpuos_ext.submit_add(as_list[i], bs_list[i], outs_add[i])
        gpuos_ext.submit_mul(as_list[i], bs_list[i], outs_mul[i])

        # Flush in batches of K
        if (i + 1) % 64 == 0:
            gpuos_ext.flush(sync=False)

    # Final flush and wait
    gpuos_ext.flush(sync=True)

    torch.cuda.synchronize()
    t1 = time.time()

    # Spot check
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

    gpuos_ext.shutdown()

if __name__ == '__main__':
    main()

