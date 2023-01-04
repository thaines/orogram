# Copyright 2023 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy

from libc.math cimport log, fabs, INFINITY

from .xentropy cimport section_crossentropy



cpdef float simplify():
  return 0.0
