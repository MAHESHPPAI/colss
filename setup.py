from setuptools import setup, Extension
import pybind11
import numpy as np

ext = Extension(
    "colss._colss",
    sources=["binding.cpp"],
    include_dirs=[
        pybind11.get_include(),
        np.get_include(),
        "include",
    ],
    extra_compile_args=[
        "-std=c++20",
        "-O3",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
        "-fopenmp",
    ],
    extra_link_args=["-fopenmp"],
    language="c++",
)

setup(
    name="colss",
    version="0.1.0",
    ext_modules=[ext],
    packages=["colss"],
)
