import torch
import os

# Ensure the extension is buildable — demo uses dynamic build in examples/pytorch_batch_demo.py
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
    verbose=False,
)

from pytorch_ext.scheduler import scheduler_context


def main():
    torch.cuda.init()
    device = torch.device('cuda:0')

    # Small tensors — candidate for aggregation
    M = 1000
    N = 2048
    as_list = [torch.randn(N, device=device, dtype=torch.float32) for _ in range(M)]
    bs_list = [torch.randn(N, device=device, dtype=torch.float32) for _ in range(M)]

    with torch.no_grad():
        with scheduler_context(capacity=8192, threads_per_block=256, size_threshold=1 << 15, auto_flush_ms=2.0):
            outs = []
            for i in range(M):
                # Mix of add and mul; results may be consumed later
                if i % 2 == 0:
                    outs.append(as_list[i] + bs_list[i])
                else:
                    outs.append(as_list[i] * bs_list[i])
            # Context exit will flush and join

    # Quick correctness check
    ok = True
    for i in range(3):
        a, b = as_list[i], bs_list[i]
        ref = (a + b) if i % 2 == 0 else (a * b)
        if not torch.allclose(outs[i], ref, rtol=1e-4, atol=1e-4):
            ok = False
    print('scheduler demo', 'OK' if ok else 'MISMATCH')


if __name__ == '__main__':
    main()

