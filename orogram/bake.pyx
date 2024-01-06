# Copyright 2024 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy



cpdef void fitgapmass(float[:] center, float[:] gap, float[:] prob):
  """Given bin centers this outputs the probabilities (into prob) of a piecewise linear PDF (Orogram) using those centers such that the probability mass between each pair of centers is as given in the gap array, which is of length one less than center/prob."""
  cdef long i
  cdef float v

  # Allocate needed memory...
  cdef float[:] gsum = numpy.empty(gap.shape[0], dtype=numpy.float32)
  cdef float[:] low = numpy.empty(prob.shape[0], dtype=numpy.float32)
  cdef float[:] high = numpy.empty(prob.shape[0], dtype=numpy.float32)

  with nogil:
    # Convert gaps into sums, i.e. the constraint becomes that two adjacent output probabilites must sum to the given value...
    for i in range(gsum.shape[0]):
      gsum[i] = 2 * gap[i] / (center[i+1] - center[i])

    # Initialise minimum/maximum value for each output probability...
    for i in range(prob.shape[0]):
      low[i] = 0.0
      high[i] = gap[i-1] if i>0 and gap[i-1]<gap[i] else gap[i]

    # Do a forward then backwards pass, updating the minimum/maximum so they are consistent...
    ## Forward...
    for i in range(gsum.shape[0]):
      v = gap[i] - high[i]
      if v>low[i+1]:
        low[i+1] = v

      v = gap[i] - low[i]
      if v<high[i+1]:
        high[i+1] = v

    ## Backwards...
    for i in range(gsum.shape[0], 0, -1):
      v = gap[i-1] - high[i]
      if v>low[i-1]:
        low[i-1] = v

      v = gap[i-1] - low[i]
      if v<high[i-1]:
        high[i-1] = v

    # Transfer into the prob array with a single forward pass...
    prob[0] = low[0]
    for i in range(1, prob.shape[0]):
      prob[i] = gsum[i-1] - prob[i-1]
