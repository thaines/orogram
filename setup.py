#! /usr/bin/env python3
# Copyright 2022 Tom SF Haines

import numpy

from setuptools import setup
from Cython.Build import cythonize

setup(
  name = 'Orogram',
  version = '0.1',
  author = 'Tom SF Haines',
  author_email = 'thaines@gmail.com',
  description = 'A library for working with 1D PDFs represented with piecewise linear functions.',
  license = 'Unselected',
  packages = ['orogram'],
  ext_modules = cythonize('orogram/*.pyx', include_path=[numpy.get_include()]),
  install_requires = ['numpy'],
  python_requires = '>=3.6')
