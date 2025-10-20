from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
                'nvcc': ['-O3', '-std=c++17', '-rdc=true']
            },
            extra_link_args=['-lcuda', '-lnvrtc', '-lcudart'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

