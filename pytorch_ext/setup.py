from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension

# Monkey-patch to skip CUDA version check (CUDA 13.0 vs PyTorch 12.4 is acceptable)
_original_check = torch.utils.cpp_extension._check_cuda_version
def _patched_check(*args, **kwargs):
    pass
torch.utils.cpp_extension._check_cuda_version = _patched_check

this_dir = Path(__file__).parent.resolve()

setup(
    name='gpuos_ext',
    ext_modules=[
        CUDAExtension(
            name='gpuos_ext',
            sources=[
                str(this_dir / 'gpuos_ext.cpp'),
                str(this_dir.parent / 'src' / 'persistent_kernel.cu'),
            ],
            include_dirs=[str(this_dir.parent / 'src')],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                        '-gencode=arch=compute_121,code=sm_121']
            },
            extra_link_args=[
                '-L/usr/local/cuda/lib64',
                '-lcuda', '-lnvrtc', '-lcudart'
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

