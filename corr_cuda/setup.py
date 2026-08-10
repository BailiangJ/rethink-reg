from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="corr_cuda",
    version="0.1.0",
    packages=["corr_cuda"],
    package_dir={"corr_cuda": "."},
    ext_modules=[
        CUDAExtension(
            name="corr_cuda._C",
            sources=[
                "csrc/corr_cuda.cpp",
                "csrc/corr_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
