# Copyright 2023 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy

from libc.math cimport exp, log, fabs, INFINITY
from libc.stdlib cimport malloc, free, realloc
from libc.stdlib cimport qsort

from .xentropy cimport section_crossentropy



# A structure used internally to store the dyanmic programming state...
cdef packed struct Triangle:
  long ai 
  long atri # Index of specific a in the tri array, as there are multiple options
  long bi
  # ci is always implicitly avaliable from indexing when needed
  float cost # Cost upto b, including b prior ratio
  float prob # Probability at b


cdef int tricmp(const void * lhs, const void * rhs) nogil:
  cdef float cost_lhs = (<Triangle*>lhs).cost
  cdef float cost_rhs = (<Triangle*>rhs).cost
  
  if cost_lhs<cost_rhs:
    return -1
  
  if cost_lhs>cost_rhs:
    return 1
  
  return 0



cpdef tuple dp(float[:] x, float[:] p, float samples, float perbin):
  """Simplifies an orogram by picking a subset of bin centers to keep; uses dynamic programing to find the maximum a posteriori with cross entropy as a substitute for not having the actual data. Main input is x[:] and p[:], two algined arrays describing an orogram. Because this is normalised you also provide how many samples were used to generate the input PDF; you also need to provide the prior, as the expected number of samples in each bin of the output — this is converted into the parameter of an exponential distribution (lambda = 1/perbin). The return is (new x, new p, cost of output, cost with prior term only, cost of prior term if all bins kept, total number of dominant triangles that were stored). Note that the costs are negative log probability and includes ratios for including the points it does relative to the, not being included, meaning it is not directly comparable to the cost if calculated manually. Cost of input is the cost of the input, which is just the sum of ratios for all entries; for comparison really."""
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
  cdef long[:] to_old
  
  if bad>0:
    old_x = x
    old_p = p
    old_mass = mass
    
    x = numpy.empty(old_x.shape[0] - bad, dtype=numpy.float32)
    p = numpy.empty(x.shape[0], dtype=numpy.float32)
    mass = numpy.empty(x.shape[0], dtype=numpy.float32)
    to_old = numpy.empty(x.shape[0], dtype=int)
    
    with nogil:
      j = 0
      for i in range(old_x.shape[0]):
        if old_mass[i]>=1e-12:
          x[j] = old_x[i]
          p[j] = old_p[i]
          mass[j] = old_mass[i]
          to_old[j] = i
          j += 1


  # Use the mass to calculate the prior ratio, as in the change in negative log liklihood given that the indexed point is included...
  cdef float[:] p_rat = numpy.empty(x.shape[0], dtype=numpy.float32)
  cdef float priorall = 0.0
  
  with nogil:
    for i in range(p_rat.shape[0]):
      p_rat[i] = -log(exp(samples*mass[i]/perbin) - 1) if mass[i]>=1e-12 else 0.0
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
  
  
  # We need a data structure to store the set of dominant options for each (b-c), where the PDF is constructed of triangles a-b-c from the bin centers of the input. This is effectively an upper-triangular matrix with multiple values in each cell. We use a 1D array of these, that gets realloc-ed each time it's too small, with indexing of the range for each cell from a pair of 1D array that are tehmselves indexed by the above index array!..
  cdef long tri_size = 4 * pairs
  cdef long tri_next = 0
  
  cdef Triangle * tri = <Triangle*>malloc(tri_size * sizeof(Triangle))
  
  cdef long * tri_start = <long*>malloc(pairs * sizeof(long))
  cdef long * tri_count = <long*>malloc(pairs * sizeof(long))
  
  
  # Calculate set of dominant options at every b-c combination — involves calculating the cost of every a-b-c and then filtering out all values of a that are dominated...
  cdef long ai, bi, ci, atri
  cdef float p_a, p_b, p_max
  
  cdef float q0, q1, total
  cdef double log_q0, log_q1
  cdef Triangle last # The best solution in the final bin centre
  
  with nogil:
    # First bin - bi is implicitely 0...
    for ci in range(1, x.shape[0]):
      i = index[0] + ci - 0 - 1
      
      if tri_next==tri_size:
        tri_size += pairs
        tri = <Triangle*>realloc(tri, tri_size * sizeof(Triangle))
      
      tri_start[i] = tri_next
      tri_count[i] = 1
      
      tri[tri_next].ai = -1
      tri[tri_next].atri = -1
      tri[tri_next].bi = 0
      tri[tri_next].cost = p_rat[0]
      tri[tri_next].prob = 2 * mass_start[i] / (x[ci] - x[0])
      
      tri_next += 1
    
    # Middle bin centres...
    for bi in range(1, x.shape[0]-1):
      for ci in range(bi+1, x.shape[0]):
        i = index[bi] + ci - bi - 1
        tri_start[i] = tri_next

        # Loop ai and generate every triangle cost, dumping them onto the end of the tri array; note that we have to consider every dominate from ai as a possible previous cost...
        for ai in range(bi):
          j = index[ai] + bi - ai - 1
          for atri in range(tri_start[j], tri_start[j]+tri_count[j]):
            # Check we have storage...
            if tri_next==tri_size:
              tri_size += pairs
              tri = <Triangle*>realloc(tri, tri_size * sizeof(Triangle))
            
            # Need probability at a and b of the triangle...
            p_a = tri[atri].prob
            p_b = mass_end[j] + mass_start[i]
            p_b -= p[bi] * 0.5 * (x[bi+1] - x[bi-1]) # Central bin gets double counted
            p_b /= 0.5 * (x[ci] - x[ai])
            
            # Prepare cost to be the cost thus far plus the prior ratio...
            total = tri[atri].cost + p_rat[bi]
            
            # Loop linear segments from a to b, summing in their cross entropy term...
            q0 = p_a
            log_q0 = log(q0) if q0>=1e-64 else -150
          
            for k in range(ai, bi):
              t = (x[k+1] - x[ai]) / (x[bi] - x[ai])
              q1 = (1-t)*p_a + t*p_b
              log_q1 = log(q1) if q1>=1e-64 else -150
          
              total += samples * (x[k+1] - x[k]) * section_crossentropy(p[k], p[k+1], q0, q1, log_q0, log_q1)
            
              q0 = q1
              log_q0 = log_q1
            
            # Record and update...
            tri[tri_next].ai = ai
            tri[tri_next].atri = atri
            tri[tri_next].bi = bi
            tri[tri_next].cost = total
            tri[tri_next].prob = p_b
            tri_next += 1
        
        # Identify the dominant set and shrink it back down to only contain that — dominance is defined as lower cost and higher probability, so first sort by cost then loop keeping track of highest probability thus far, deleting all that are less...
        tri_count[i] = tri_next - tri_start[i]
        qsort(&tri[tri_start[i]], tri_count[i], sizeof(Triangle), &tricmp)
        
        p_max = tri[tri_start[i]].prob
        tri_next = tri_start[i] + 1
        
        for atri in range(tri_start[i] + 1, tri_start[i] + tri_count[i]):
          if tri[atri].prob > p_max:
            p_max = tri[atri].prob
            
            tri[tri_next] = tri[atri]
            tri_next += 1

        tri_count[i] = tri_next - tri_start[i]

    # Last bin - can just grab the best as there are no future decisions to be made...
    last.cost = INFINITY
    last.bi = x.shape[0] - 1
    
    bi = last.bi
    for ai in range(bi):
      j = index[ai] + bi - ai - 1
      for atri in range(tri_start[j], tri_start[j]+tri_count[j]):
        # Need probability at a and b of the triangle...
        p_a = tri[atri].prob
        p_b = 2 * mass_end[j] / (x[bi] - x[ai])
            
        # Prepare cost to be the cost thus far plus the prior ratio...
        total = tri[atri].cost + p_rat[bi]
            
        # Loop linear segments from a to b, summing in their cross entropy term...
        q0 = p_a
        log_q0 = log(q0) if q0>=1e-64 else -150
          
        for k in range(ai, bi):
          t = (x[k+1] - x[ai]) / (x[bi] - x[ai])
          q1 = (1-t)*p_a + t*p_b
          log_q1 = log(q1) if q1>=1e-64 else -150
          
          total += samples * (x[k+1] - x[k]) * section_crossentropy(p[k], p[k+1], q0, q1, log_q0, log_q1)
            
          q0 = q1
          log_q0 = log_q1
        
        # Check if it's the best thus far...
        if total < last.cost:
          last.ai = ai
          last.atri = atri
          last.cost = total
          last.prob = p_b
  
  
  # Count how many bins are in the best solution...
  cdef long nlen = 0
  cdef Triangle * targ = &last
  
  with nogil:
    while True:
      nlen += 1
      if targ.ai<0:
        break
      targ = &tri[targ.atri]
  
  
  # Extract the return information and package it all up for the return...
  cdef numpy.ndarray retx = numpy.empty(nlen, dtype=numpy.float32)
  cdef numpy.ndarray retp = numpy.empty(nlen, dtype=numpy.float32)
  cdef numpy.ndarray retk = numpy.zeros(old_x.shape[0] if bad>0 else x.shape[0], dtype=bool)
  
  cdef float[:] rx = retx
  cdef float[:] rp = retp
  cdef numpy.uint8_t[:] rk = numpy.frombuffer(retk, dtype=numpy.uint8)
  cdef float priorcost = 0.0
  
  with nogil:
    targ = &last
    i = rx.shape[0]-1

    while True:
      rx[i] = x[targ.bi]
      rp[i] = targ.prob
      rk[to_old[targ.bi] if bad>0 else targ.bi] = True
      priorcost += p_rat[targ.bi]
      
      if targ.ai<0:
        break
      
      i -= 1
      targ = &tri[targ.atri]


  # Clean up the main data structure then return...
  free(tri_count)
  free(tri_start)
  free(tri)

  return retx, retp, retk, last.cost, priorcost, priorall, tri_next + 1
