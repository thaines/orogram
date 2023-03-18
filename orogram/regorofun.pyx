# Copyright 2022 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy



cpdef int length(dic, long blocksize, long low, long high):
  """Counts how many parameters are needed to define the orogram, just the probability values, not including anything else."""
  
  # Variables...
  cdef long ret = 1 # This is pre-counting the zero after high.
  cdef bint lastnonzero = False, alreadycounted = False
  
  cdef long i = low
  cdef long blockindex = i // blocksize
  cdef long blockbase = blockindex * blocksize
  cdef float[:] block = dic[blockindex]
  
  # Loop the data and count the range...
  with nogil:
    while i <= high:
      # Count logic...
      if block[i - blockbase] > 1e-12:
        if not lastnonzero:
          if not alreadycounted:
            ret += 1
        ret += 1
        lastnonzero = True
    
      else:
        if lastnonzero:
          ret += 1
          alreadycounted = True
        else:
          alreadycounted = False
        lastnonzero = False
    
      # Move on after...
      i += 1
      
      while ((i - blockbase) >= blocksize) and (i <= high):
        blockindex += 1
        with gil:
          if blockindex in dic:
            blockbase = blockindex * blocksize
            block = dic[blockindex]
            break
      
          else:
            if lastnonzero:
              ret += 1
              lastnonzero = False
              alreadycounted = False
            i += blocksize

  return ret



cpdef void add(dic, long blocksize, long[:] base, float[:] w):
  cdef long i, bi
  
  cdef long blockindex = (base[0] // blocksize) - 1
  cdef long blockbase = 0
  cdef float[:] block = None
  
  with nogil:
    for i in range(base.shape[0]):

      bi = base[i] // blocksize
      if bi!=blockindex:
        blockindex = bi
        blockbase = blockindex * blocksize
        with gil:
          if bi not in dic:
            newblock = numpy.zeros(blocksize, dtype=numpy.float32)
            dic[bi] = newblock
            block = newblock
          else:
            block = dic[bi]
    
      block[base[i] - blockbase] += w[i]



cpdef void smoothadd(dic, long blocksize, long[:] base, float[:] t, float[:] w):
  cdef long i, bi
  
  cdef long blockindex = (base[0] // blocksize) - 1
  cdef long blockbase = 0
  cdef float[:] block = None
  
  with nogil:
    for i in range(base.shape[0]):
      # Low side...
      bi = base[i] // blocksize
      if bi!=blockindex:
        blockindex = bi
        blockbase = blockindex * blocksize
        with gil:
          if bi not in dic:
            newblock = numpy.zeros(blocksize, dtype=numpy.float32)
            dic[bi] = newblock
            block = newblock
          else:
            block = dic[bi]
    
      block[base[i] - blockbase] += (1 - t[i]) * w[i]
    
      # High side...
      bi = (base[i] + 1) // blocksize
      if bi!=blockindex:
        blockindex = bi
        blockbase = blockindex * blocksize
        with gil:
          if bi not in dic:
            block = numpy.zeros(blocksize, dtype=numpy.float32)
            dic[bi] = block
          else:
            block = dic[bi]
    
      block[base[i] + 1 - blockbase] += t[i] * w[i]



cpdef void binadd(dic, long blocksize, long base, float[:] density):
  """Adds values directly to the bins."""
  cdef long i, bi
  
  cdef long blockindex = (base // blocksize) - 1
  cdef long blockbase = 0
  cdef float[:] block = None
  
  with nogil:
    for i in range(density.shape[0]):
      bi = (base+i) // blocksize
      if bi!=blockindex:
        blockindex = bi
        blockbase = blockindex * blocksize
        with gil:
          if bi not in dic:
            newblock = numpy.zeros(blocksize, dtype=numpy.float32)
            dic[bi] = newblock
            block = newblock
          else:
            block = dic[bi]
      
      block[base + i - blockbase] += density[i]



cpdef void binweight(dic, long blocksize, long[:] index, float[:] out):
  """Returns weights for a series of bin indices."""
  cdef long i, bi
  
  cdef long blockindex = (index[0] // blocksize) - 1
  cdef long blockbase = 0
  cdef float[:] block = None
  
  with nogil:
    for i in range(index.shape[0]):
      bi = index[i] // blocksize
      
      if bi!=blockindex:
        blockindex = bi
        blockbase = blockindex * blocksize
        with gil:
          if bi not in dic:
            block = None
          else:
            block = dic[bi]
      
      if block is not None:
        out[i] = block[index[i] - blockbase]
      else:
        out[i] = 0.0



cpdef void weight(dic, long blocksize, long[:] base, float[:] t, float[:] out):
  """Returns weights for x positions, with linear interpolation."""
  cdef long i, bi
  
  cdef long blockindex = (base[0] // blocksize) - 1
  cdef long blockbase = 0
  cdef float[:] block = None
  
  with nogil:
    for i in range(base.shape[0]):
      # Low side...
      bi = base[i] // blocksize
      
      if bi!=blockindex:
        blockindex = bi
        blockbase = blockindex * blocksize
        with gil:
          if bi not in dic:
            block = None
          else:
            block = dic[bi]
      
      if block is not None:
        out[i] += (1 - t[i]) * block[base[i] - blockbase]
      
      # High side...
      bi = (base[i] + 1) // blocksize
      
      if bi!=blockindex:
        blockindex = bi
        blockbase = blockindex * blocksize
        with gil:
          if bi not in dic:
            block = None
          else:
            block = dic[bi]
      
      if block is not None:
        out[i] += t[i] * block[base[i] + 1 - blockbase]



cpdef modes(dic, long blocksize, long low, long high):
  """Returns the bins that are higher than their neighbours."""
  
  # Variables...
  cdef long i
  
  cdef long blockindex
  cdef long blockbase
  cdef float[:] block
  
  cdef long total = 0
  cdef float prev = 0.0
  cdef float curr
  cdef bint grew = False
  
  # First pass to count the number of modes...
  i = low
  blockindex = i // blocksize
  blockbase = blockindex * blocksize
  block = dic[blockindex]
    
  with nogil:
    while i <= high:
      curr = block[i - blockbase]
      if curr > prev:
        grew = True
    
      else:
        if grew==True:
          total += 1
      
        grew = False
    
      prev = curr
      i += 1
    
      while ((i - blockbase) >= blocksize) and (i <= high):
        blockindex += 1
        with gil:
          if blockindex in dic:
            blockbase = blockindex * blocksize
            block = dic[blockindex]
            break
      
          else:
            if grew==True:
              total += 1
        
            prev = 0.0
            grew = False
        
            i += blocksize

  # Create storage...
  ret = numpy.empty(total, dtype=int)
  cdef long[:] store = ret
  
  total = 0
  prev = 0.0
  grew = False
  
  # Second pass to record modes...
  i = low
  blockindex = i // blocksize
  blockbase = blockindex * blocksize
  block = dic[blockindex]
  
  with nogil:
    while i <= high:
      curr = block[i - blockbase]
      if curr > prev:
        grew = True
    
      else:
        if grew==True:
          store[total] = i - 1
          total += 1
      
        grew = False
    
      prev = curr
      i += 1
    
      while ((i - blockbase) >= blocksize) and (i <= high):
        blockindex += 1
        with gil:
          if blockindex in dic:
            blockbase = blockindex * blocksize
            block = dic[blockindex]
            break
      
          else:
            if grew==True:
              store[total] = i - 1
              total += 1
        
            prev = 0.0
            grew = False
        
            i += blocksize
  
  # Return...
  return ret



cpdef void cdf(dic, long blocksize, long low, float norm, float[:] out):
  """Fills out with the cdf, assuming position 0 is equivalent to bin 0. norm must be the total weight."""
  cdef int i, j, offset
  cdef double cumsum = 0.0, prev = 0.0, curr = 0.0
  
  cdef long blockindex = low // blocksize
  cdef long blockbase = blockindex * blocksize
  cdef float[:] block = dic[blockindex]

  with nogil:
    out[0] = 0.0
    i = 1
    while i < out.shape[0]:
      offset = low - 1 + i - blockbase
    
      if offset >= blocksize:
        while i < out.shape[0]:
          blockindex += 1
          blockbase += blocksize
          offset -= blocksize
          
          with gil:
            if blockindex in dic:
              block = dic[blockindex]
              break
            
            else:
              cumsum += 0.5*curr / norm
              out[i] = cumsum
              
              curr = 0.0
              i += 1
              
              j = 1
              while j < blocksize and i < out.shape[0]:
                out[i] = cumsum
                i += 1
                j += 1
                
              offset += blocksize

      if i>=out.shape[0]:
        break
      
      prev = curr
      curr = block[offset]
      
      cumsum += 0.5 * (prev + curr) / norm
      out[i] = cumsum

      i += 1
