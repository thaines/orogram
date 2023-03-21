# Copyright 2023 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False



cpdef float above(float[:] x, float[:] p, float thresh) nogil:
  """Returns the amount of probability mass in the region where the pdf is above the given threshold. Assumes the function described by x -> p integrates to 1."""
  cdef long i
  cdef float t, total = 0.0
  cdef bint inside = p[0] >= thresh
  
  for i in range(x.shape[0]-1):
    if inside:
      if p[i+1] < thresh:
        t = (thresh - p[i]) / (p[i+1] - p[i])
        total += 0.5 * ((2-t)*p[i] + t*p[i+1]) * t * (x[i+1] - x[i])
        
        inside = False
      
      else:
        total += 0.5 * (p[i] + p[i+1]) * (x[i+1] - x[i])
    
    else:
      if p[i+1] >= thresh:
        t = (thresh - p[i]) / (p[i+1] - p[i])
        total += 0.5 * ((1-t)*p[i] + (1+t)*p[i+1]) * (1-t) * (x[i+1] - x[i])
        
        inside = True
  
  return total



cpdef list ranges(float[:] x, float[:] p, float thresh):
  """Collects the ranges of the given pdf which are above the given threshold. Returns a list of tuples, each being (start, end, mass), with the mass the total mass between start and end where every point in that range is above the given threshold."""
  cdef list ret = []
  
  cdef long i
  cdef float t, start = x[0], total = 0.0
  cdef bint inside = p[0] >= thresh
  
  with nogil:
    for i in range(x.shape[0]-1):
      if inside:
        if p[i+1] < thresh:
          t = (thresh - p[i]) / (p[i+1] - p[i])
          total += 0.5 * ((2-t)*p[i] + t*p[i+1]) * t * (x[i+1] - x[i])
        
          with gil:
            ret.append((start, (1-t)*x[i] + t*x[i+1], total))
        
          total = 0.0
          inside = False
      
        else:
          total += 0.5 * (p[i] + p[i+1]) * (x[i+1] - x[i])
    
      else:
        if p[i+1] >= thresh:
          t = (thresh - p[i]) / (p[i+1] - p[i])
          total += 0.5 * ((1-t)*p[i] + (1+t)*p[i+1]) * (1-t) * (x[i+1] - x[i])
          
          start = (1-t)*x[i] + t*x[i+1]
          inside = True
  
  if inside:
    ret.append((start, x[x.shape[0]-1], total))
  
  return ret



cpdef float credible(float[:] x, float[:] p, float mass, float tol = 1e-6):
  """Calculates the credible interval(s), in the sense of the smallest area of the range required to obtain a given probability mass. That is, it selects the ranges with the highest probability. Does a binary search on the threshold, so not the fastest. Just returns the threshold that is required to get the given quantity of mass - use ranges() to get them."""
  cdef long i
  cdef float low = 0.0
  cdef float high = p[0]
  cdef float half, amount
  
  with nogil:
    # Correct high to be the highest value...
    for i in range(p.shape[0]):
      if p[i]>high:
        high = p[i]
    
    # Do a binary search...
    while (high-low) > tol:
      half = 0.5 * (low+high)
      amount = above(x, p, half)
      
      if amount>mass:
        low = half
        
      else:
        high = half
  
  return 0.5 * (low+high)
