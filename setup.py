from pathlib import Path

import setuptools
from torch.utils import cpp_extension

sources = [str(p) for p in Path("csrc").glob("*.cu") if p.is_file()]

nvcc_args = [
    "-O3",
]
cxx_args = ["-O3"]

setuptools.setup(
    name="gml_lab_custom_ops",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "gml_lab_custom_ops",
            sources=sources,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
