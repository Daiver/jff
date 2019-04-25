#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension(
    'FernBuiltins', 
    sources = ['FernBuiltins.c'], 
    extra_compile_args = ["-O3"]) ]

setup(
        name = 'FernBuiltins',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )

