# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy


cpdef void fitgapmass(float[:] center, float[:] gap, float[:] prob):
  """Given bin centers this outputs the probabilities (into prob) of a piecewise linear PDF (Orogram) using those centers such that the probability mass between each pair of centers is as given in the gap array, which is of length one less than center/prob."""
  cdef long i
  cdef float est1, est2
  cdef float low, high
  cdef bint change
  
  # Allocate needed memory...
  cdef float[:] average = numpy.empty(gap.shape[0], dtype=numpy.float32)
  
  with nogil:
    # Convert gaps into average, i.e. the constraint becomes that two adjacent output probabilites must average to the given value...
    for i in range(average.shape[0]):
      average[i] = gap[i] / (center[i+1] - center[i])
      
    # Initialise prob with the average of adjacent gaps...
    prob[0] = average[0]
    for i in range(1, average.shape[0]):
      prob[i] = 0.5*(average[i-1] + average[i])
    prob[average.shape[0]] = average[average.shape[0]-1]
    
    # Do forwards/backwards passes of updating each probability to be within the range the constraints imply - improves things a little...
    for _ in range(128):
      change = False
      # Forwards...
      for i in range(1, average.shape[0]):
        est1 = 2*average[i-1] - prob[i-1]
        est2 = 2*average[i] - prob[i+1]
        
        low = est1 if est1<est2 else est2
        high = est1 if est1>est2 else est2
        
        if low < 0.0:
          low = 0.0
        
        if prob[i]<low:
          change = True
          prob[i] = low
        
        if prob[i]>high:
          change = True
          prob[i] = high

      prob[average.shape[0]] = 2*average[average.shape[0]-1] - prob[average.shape[0]-1]
      if prob[average.shape[0]]<0.0:
        prob[average.shape[0]] = 0.0
      
      # Backwards...
      for i in range(average.shape[0]-1, 0, -1):
        est1 = 2*average[i-1] - prob[i-1]
        est2 = 2*average[i] - prob[i+1]
        
        low = est1 if est1<est2 else est2
        high = est1 if est1>est2 else est2
        
        if low < 0.0:
          low = 0.0
        
        if prob[i]<low:
          change = True
          prob[i] = low
        
        if prob[i]>high:
          change = True
          prob[i] = high
      
      prob[0] = 2*average[0] - prob[1]
      if prob[0]<0.0:
        prob[0] = 0.0
      
      # Exit early if done...
      if not change:
        break



cpdef void fitgapmass_meh(float[:] center, float[:] gap, float[:] prob):
  """Given bin centers this outputs the probabilities (into prob) of a piecewise linear PDF (Orogram) using those centers such that the probability mass between each pair of centers is as given in the gap array, which is of length one less than center/prob. This version solves the problem exactly... but then suffers from accidental hedgehog. Have dropped for now, as a basic solver, while not as precise, gives a more reasonable result and this isn't my focus at this time."""
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
      high[i] = gsum[i-1] if i>0 and gsum[i-1]<gsum[i] else gsum[i]

    # Do a forward then backwards pass, updating the minimum/maximum so they are consistent...
    ## Forward...
    for i in range(gsum.shape[0]):
      v = gsum[i] - high[i]
      if v>low[i+1]:
        low[i+1] = v

      v = gsum[i] - low[i]
      if v<high[i+1]:
        high[i+1] = v

    ## Backwards...
    for i in range(gsum.shape[0], 0, -1):
      v = gsum[i-1] - high[i]
      if v>low[i-1]:
        low[i-1] = v

      v = gsum[i-1] - low[i]
      if v<high[i-1]:
        high[i-1] = v

    # Transfer into the prob array with a single forward pass...
    prob[0] = low[0]
    for i in range(1, prob.shape[0]):
      prob[i] = gsum[i-1] - prob[i-1]

      #if prob[i]<low[i]:
        #prob[i] = low[i]
      #elif prob[i]>high[i]:
        #prob[i] = high[i]
