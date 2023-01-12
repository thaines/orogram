# Copyright 2023 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy

from libc.math cimport exp, log, fabs, INFINITY

from .xentropy cimport section_crossentropy



cpdef tuple dp(float[:] x, float[:] p, float samples, float perbin):
  """Simplifies an orogram by picking a subset of bin centers to keep; uses dynamic programing to find the maximum a posteriori with cross entropy as a substitute for not having the actual data. Main input is x[:] and p[:], two algined arrays describing an orogram. Because this is normalised you also provide how many samples were used to generate the input PDF; you also need to provide the prior, as the expected number of samples in each bin of the output — this is converted into the parameter of an exponential distribution (lambda = 1/perbin). The return is (new x, new p, cost of output, cost with prior term only, cost of prior term if all bins kept). Note that the costs are negative log probability and includes ratios for including the points it does relative to the, not being included, meaning it is not directly comparable to the cost if calculated manually. Cost of input is the cost of the input, which is just the sum of ratios for all entries; for comparison really."""
  cdef long i, j, k, bad
  
  # Calculate the quantity of probability mass that snaps to every center in x, counting the number of zeroes while we're at it...
  cdef float[:] mass = numpy.empty(x.shape[0], dtype=numpy.float32)
  bad = 0
  
  with nogil:
    mass[0] = 0.125 * (x[1] - x[0]) * (3*p[0] + p[1])
    if mass[0]<1e-12:
      bad += 1
  
    for i in range(1, mass.shape[0]-1):
      mass[i] = 0.125 * ((x[i] - x[i-1]) * (p[i-1] + 3*p[i]) + (x[i+1] - x[i]) * (3*p[i] + p[i+1]))
      if mass[i]<1e-12:
        bad += 1
  
    mass[mass.shape[0]-1] = 0.125 * (x[mass.shape[0]-1] - x[mass.shape[0]-2]) * (p[mass.shape[0]-2] + 3*p[mass.shape[0]-1])
    if mass[mass.shape[0]-1]<1e-12:
      bad += 1
  
  # If we have any zeros we need to remove them — can just delete entries as have no density and impossible to be selected by simplification anyway...
  cdef float[:] old_x
  cdef float[:] old_p
  cdef float[:] old_mass
  
  if bad>0:
    old_x = x
    old_p = p
    old_mass = mass
    
    x = numpy.empty(old_x.shape[0] - bad, dtype=numpy.float32)
    p = numpy.empty(x.shape[0], dtype=numpy.float32)
    mass = numpy.empty(x.shape[0], dtype=numpy.float32)
    
    with nogil:
      j = 0
      for i in range(old_x.shape[0]):
        if old_mass[i]>=1e-12:
          x[j] = old_x[i]
          p[j] = old_p[i]
          mass[j] = old_mass[i]
          j += 1

  # Use the mass to calculate the prior ratio, as in the change in negative log liklihood given that the indexed point is included...
  cdef float[:] p_rat = numpy.empty(x.shape[0], dtype=numpy.float32)
  cdef float priorall = 0.0
  
  with nogil:
    for i in range(p_rat.shape[0]):
      p_rat[i] = -log(exp(samples*mass[i]/perbin) - 1)
      priorall += p_rat[i]
  
  # We need multiple data structures with data for all pairs of x indices; to pack them in we use 1D arrays where for two indices, i and j with i<j, we can find the value for the given pair at <some array>[index[i] + j - i - 1]
  cdef long pairs = (x.shape[0] * (x.shape[0]-1)) // 2
  
  cdef long[:] index = numpy.empty(x.shape[0], dtype=int)
  with nogil:
    index[0] = 0
    for i in range(1, index.shape[0]):
      index[i] = index[i-1] + x.shape[0] - i

  # Pre-calculate the amount of probability mass allocated to the limits of each pairing, i.e. sum in the bins within the range interpolated correctly. As each range is needed twice (once for each end) this halves computation at the expense of allocating an array, so may not be faster for smaller bin counts...
  cdef float[:] mass_start = numpy.empty(pairs, dtype=numpy.float32)
  cdef float[:] mass_end = numpy.empty(pairs, dtype=numpy.float32)
  cdef float ms, me, t, width
  
  with nogil:
    for i in range(x.shape[0]):
      for j in range(i+1, x.shape[0]):
        ms = 0.0
        me = 0.0
      
        for k in range(i,j+1):
          t = (x[k] - x[i]) / (x[j] - x[i])
          width = 0.5 * (x[k+1 if k+1<x.shape[0] else k] - x[k-1 if k>0 else k])
          ms += (1-t) * p[k] * width
          me += t * p[k] * width
      
        mass_start[index[i] + j - i - 1] = ms
        mass_end[index[i] + j - i - 1] = me
  
  # Create data structure to record the partial objective evaluations, for dyanmic programming. An upper triangular matrix where we record both the cost of the best solution and the index of the previous bin that 'won' the argmax. The index for the evaluation of R(b,c) is in cost[index[b] + c - b - 1] with the a selected by the argmin at the same position in the cost_a[] array. In addition the probability mass assigned to position b is recorded in cost_bm...
  cdef float[:] cost = numpy.empty(pairs, dtype=numpy.float32)
  cdef long[:] cost_a = numpy.empty(pairs, dtype=int)
  cdef float[:] cost_bm = numpy.empty(pairs, dtype=numpy.float32)
  
  # Fill in the data structure by solving the optimisation problem for each entry...
  cdef long ai, bi, ci
  cdef float am, bm
  
  cdef float best, best_bm, curr
  cdef long best_a
  
  cdef long final_a
  cdef float final, final_bm
  
  cdef float q0, q1
  cdef double log_q0, log_q1
  
  with nogil:
    # First bin...
    for ci in range(1, x.shape[0]):
      cost[index[0] + ci - 1] = p_rat[0]
      cost_a[index[0] + ci - 1] = -1
      cost_bm[index[0] + ci - 1] = 2 * mass_start[index[0] + ci - 1] / (x[ci] - x[0])
    
    # Middle bins...
    for bi in range(1, x.shape[0]-1):
      for ci in range(bi+1, x.shape[0]):
        # Dummy values for best...
        best = INFINITY
        best_a = -1
        best_bm = 0.0
        
        # Loop and calculate how good each a value is, to find the best...
        for ai in range(bi):
          # Get the mass at points a and b, plus initalise curr with the relevent R() and inclusion prior ratio...
          am = cost_bm[index[ai] + bi - ai - 1]
          bm = mass_end[index[ai] + bi - ai - 1] + mass_start[index[bi] + ci - bi - 1]
          bm -= p[bi] * 0.5 * (x[bi+1] - x[bi-1]) # Central bin gets double counted
          bm /= 0.5 * (x[ci] - x[ai])
          curr = cost[index[ai] + bi - ai - 1] + p_rat[bi]
          
          # Loop linear segments from a to b, summing in their cross entropy term...
          q0 = am
          log_q0 = log(q0) if q0>=1e-64 else -150
          
          for i in range(ai, bi):
            t = (x[i+1] - x[ai]) / (x[bi] - x[ai])
            q1 = (1-t)*am + t*bm
            log_q1 = log(q1) if q1>=1e-64 else -150
          
            curr += samples * (x[i+1] - x[i]) * section_crossentropy(p[i], p[i+1], q0, q1, log_q0, log_q1)
            
            q0 = q1
            log_q0 = log_q1
          
          # If best thus far record...
          if curr<best:
            best = curr
            best_a = ai
            best_bm = bm
        
        # Record the best...
        cost[index[bi] + ci - bi - 1] = best
        cost_a[index[bi] + ci - bi - 1] = best_a
        cost_bm[index[bi] + ci - bi - 1] = best_bm
  
    # Last bin - for this we need to identify which bin comes before the last (last is always included)...
    final = INFINITY
    final_a = -1
    final_bm = 0.0
    
    bi = x.shape[0]-1
    for ai in range(bi):
      # Get the mass at points a and b, plus initalise curr with the relevent R() and inclusion prior ratio...
      am = cost_bm[index[ai] + bi - ai - 1]
      bm = 2 * mass_end[index[ai] + bi - ai - 1] / (x[bi] - x[ai])
      curr = cost[index[ai] + bi - ai - 1] + p_rat[bi]
      
      # Loop linear segments from a to b, summing in their cross entropy term...
      q0 = am
      log_q0 = log(q0) if q0>=1e-64 else -150
          
      for i in range(ai, bi):
        t = (x[i+1] - x[ai]) / (x[bi] - x[ai])
        q1 = (1-t)*am + t*bm
        log_q1 = log(q1) if q1>=1e-64 else -150
          
        curr += samples * (x[i+1] - x[i]) * section_crossentropy(p[i], p[i+1], q0, q1, log_q0, log_q1)
            
        q0 = q1
        log_q0 = log_q1
      
      # If best thus far record...
      if curr<final:
        final = curr
        final_a = ai
        final_bm = bm
      
  
  # Count how many bins are in the best solution...
  cdef long nlen = 1
  # bi = x.shape[0]-1
  ai = final_a
  
  with nogil:
    while True:
      nlen += 1
      ci = bi
      bi = ai
      if bi==0:
        break
      ai = cost_a[index[bi] + ci - bi - 1]
  
  # Extract the return information and package it all up for the return...
  cdef numpy.ndarray retx = numpy.empty(nlen, dtype=numpy.float32)
  cdef numpy.ndarray retp = numpy.empty(nlen, dtype=numpy.float32)
  cdef numpy.ndarray retk = numpy.zeros(x.shape[0], dtype=bool)
  
  cdef float[:] rx = retx
  cdef float[:] rp = retp
  cdef numpy.uint8_t[:] rk = numpy.frombuffer(retk, dtype=numpy.uint8)
  cdef float priorcost
  
  with nogil:
    bi = x.shape[0]-1
    ai = final_a
    
    i = rx.shape[0]-1
    rx[i] = x[bi]
    rp[i] = final_bm
    rk[bi] = True
    priorcost = p_rat[bi]

    while True:
      ci = bi
      bi = ai
      
      i -= 1
      rx[i] = x[bi]
      rp[i] = cost_bm[index[bi] + ci - bi - 1]
      rk[bi] = True
      priorcost += p_rat[bi]

      if bi==0:
        break
      ai = cost_a[index[bi] + ci - bi - 1]

  return retx, retp, retk, final, priorcost, priorall
