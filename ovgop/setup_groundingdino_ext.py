"""Build the multi-scale deformable-attention CUDA extension in place."""

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def collect_sources(source_directory):
    vision_source = source_directory / "vision.cpp"
    if not vision_source.is_file():
        raise FileNotFoundError(f"vision.cpp not found in {source_directory}")
    sources = [vision_source]
    sources.extend(path for path in source_directory.rglob("*.cpp") if path != vision_source)
    sources.extend(source_directory.rglob("*.cu"))
    return [str(path) for path in sources]


def extension_modules(module_name, source_directory):
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit not found; set CUDA_HOME before building")
    if not torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        print("Warning: no GPU is visible; set TORCH_CUDA_ARCH_LIST to cross-compile.")

    return [
        CUDAExtension(
            module_name,
            collect_sources(source_directory),
            include_dirs=[str(source_directory)],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    source_directory = project_root / "models" / "dino" / "csrc"
    module_name = os.environ.get("MODULE_NAME", "models.dino._C")
    setup(
        name="ovgop-deformable-attention",
        version="0.1.0",
        description="OVGOP multi-scale deformable-attention CUDA extension",
        ext_modules=extension_modules(module_name, source_directory),
        cmdclass={"build_ext": BuildExtension},
        packages=[],
    )
