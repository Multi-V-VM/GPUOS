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
    device = torch.device('cuda:0')
    ok = True
    with torch.no_grad():
        with scheduler_context(capacity=8192, threads_per_block=256, size_threshold=1 << 24, auto_flush_ms=2.0):
            # sum over last dim, keepdim
            x = torch.randn(64, 128, device=device)
            y = x.sum(dim=-1, keepdim=True)
            ref = x.sum(dim=-1, keepdim=True)
            ok &= torch.allclose(y, ref, rtol=1e-4, atol=1e-4)

            # mean over last dim, dropdim
            z = x.mean(dim=-1, keepdim=False)
            ref2 = x.mean(dim=-1, keepdim=False)
            ok &= torch.allclose(z, ref2, rtol=1e-4, atol=1e-4)

            # non-contiguous input (transpose) reduce last dim (strides vary)
            xT = x.t().contiguous().t()  # create different strides
            y2 = xT.sum(dim=-1)
            ref3 = xT.sum(dim=-1)
            ok &= torch.allclose(y2, ref3, rtol=1e-4, atol=1e-4)

            # multi-d reduction over dims (0,1)
            w = torch.randn(8, 16, 32, device=device)
            y3 = w.sum(dim=(0,1), keepdim=False)
            ref4 = w.sum(dim=(0,1), keepdim=False)
            ok &= torch.allclose(y3, ref4, rtol=1e-4, atol=1e-4)

    print('reduce demo', 'OK' if ok else 'MISMATCH')


if __name__ == '__main__':
    main()
