import setuptools
import os
from torch.utils.cpp_extension import CppExtension, BuildExtension

include_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
setuptools.setup(
    name="rasterizer",
    version="0.0.3",
    author="Daiver",
    author_email="",
    description="",
    long_description="",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(exclude=("tests",)),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    ext_modules=[CppExtension('rasterizer_cpp', ['csrc/rasterizer.cpp'])],
    cmdclass={'build_ext': BuildExtension}
)
