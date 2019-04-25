from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='lltm_cpp',
      ext_modules=[CppExtension('lltm', ['lltm.cpp'])],
      cmdclass={'build_ext': BuildExtension})
