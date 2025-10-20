import os
import torch

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
    dev = torch.device('cuda:0')
    ok = True

    with torch.no_grad():
        with scheduler_context(capacity=8192, threads_per_block=256, size_threshold=1 << 20, auto_flush_ms=2.0):
            # 1) clamp broadcast: a.clamp(0, 6)
            a = torch.randn(1024, device=dev, dtype=torch.float32) * 10
            y1 = a.clamp(0, 6)
            ref1 = torch.clamp(a, 0, 6)
            ok &= torch.allclose(y1, ref1, rtol=1e-4, atol=1e-4)

            # 2) non-contiguous slicing + add/mul
            x = torch.randn(64, 128, device=dev, dtype=torch.float32)
            xT = x.t()  # non-contiguous view (128,64)
            v = torch.randn(64, device=dev, dtype=torch.float32)
            y2 = xT + v  # broadcast add over last dim
            ref2 = xT + v
            ok &= torch.allclose(y2, ref2, rtol=1e-4, atol=1e-4)

            # 3) mixed dtype conversion: float16 + float32 -> float32
            m = torch.randn(4096, device=dev, dtype=torch.float16)
            n = torch.randn(4096, device=dev, dtype=torch.float32)
            y3 = m + n
            ref3 = (m + n)
            ok &= torch.allclose(y3, ref3.to(y3.dtype), rtol=1e-3, atol=1e-3)

            # 4) minimum/maximum with broadcast and non-contiguous
            a4 = torch.randn(32, 16, device=dev)
            b4 = torch.randn(16, device=dev)[::2].repeat(2)  # non-contiguous then contiguous via repeat
            y4 = torch.maximum(a4, b4)
            ref4 = torch.maximum(a4, b4)
            ok &= torch.allclose(y4, ref4, rtol=1e-4, atol=1e-4)

    print('advanced scheduler demo', 'OK' if ok else 'MISMATCH')


if __name__ == '__main__':
    main()

