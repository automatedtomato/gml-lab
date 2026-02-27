import warnings
from pathlib import Path

import setuptools
from torch.utils import cpp_extension

CUDA_AVAILABLE = cpp_extension.CUDA_HOME is not None
ext_modules = []
cmdclass = {}

if CUDA_AVAILABLE:
    root_dir = Path(__file__).parent.absolute()
    cutlass_dir = root_dir / "third_party" / "cutlass"
    cutlass_includes = [
        str(cutlass_dir / "include"),
        str(cutlass_dir / "tools" / "util" / "include"),
    ]

    source_files = list(Path("csrc").rglob("*.cu")) + list(Path("csrc").rglob("*.cpp"))
    sources = [str(p) for p in source_files]

    nvcc_args = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-gencode=arch=compute_80,code=sm_80",
    ]
    cxx_args = ["-O3", "-std=c++17"]

    ext_modules = [
        cpp_extension.CUDAExtension(
            "gml_lab_custom_ops",
            sources=sources,
            include_dirs=cutlass_includes,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        )
    ]

    cmdclass = {"build_ext": cpp_extension.BuildExtension}
else:
    warnings.warn(
        "CUDA not available. Skipping custom CUDA kernel build.", stacklevel=1
    )

setuptools.setup(
    name="gml_lab_custom_ops",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
