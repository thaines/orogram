# Copyright 2022 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy

from libc.math cimport log, fabs, INFINITY



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



cpdef void add(dic, long blocksize, long[:] base, float[:] t, float[:] w):
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



cpdef float reg_entropy(float[:] prob, float spacing):
  """Calculates entropy of a regular orogram using the analytic equations I've derived."""
  cdef int i
  cdef float ret = 0.0
  cdef float plogp_last, plogp, pdelta
  
  plogp_last = prob[0] * log(prob[0]) if prob[0]>1e-12 else 0.0
  
  for i in range(1, prob.shape[0]):
    
    # Calculate the p*log(p) term for teh current value...
    plogp = prob[i] * log(prob[i]) if prob[i]>1e-12 else 0.0
    
    # Do the safe bit...
    ret -= 0.5 * spacing * (plogp_last + plogp)
    
    # Do the bit that might be unstable if the delta is too small...
    pdelta = prob[i] - prob[i-1]
    if fabs(pdelta)>1e-12:
      ret -= 0.25 * spacing * (prob[i-1]*prob[i-1] - prob[i]*prob[i] + 2 * (prob[i-1]*plogp - prob[i]*plogp_last)) / pdelta
    
    # Move to next...
    plogp_last = plogp
  
  return ret



cdef float section_crossentropy(float p0, float p1, float q0, float q1, float log_q0, float log_q1) nogil:
  # Early exit if it's zero...
  if p0<1e-12 and p1<1e-12:
    return 0.0
  
  # The first term...
  cdef float ret = -0.5 * (p0*log_q0 + p1*log_q1)
  
  # Second and third terms but only if not too close to the limit...
  cdef float qdelta = q1 - q0
  if fabs(qdelta)>1e-12:
    # Second term...
    ret += (p1*q0*q0 - p0*q1*q1) * (log_q1 - log_q0) / (2*qdelta*qdelta)
    
    # Third term...
    ret += ((3*p0 + p1)*q1 - (p0 + 3*p1)*q0) / (4*qdelta)
  
  # Return...
  return ret



cpdef float aligned_crossentropy(long blocksize, float spacing, int p_start, int p_end, p_dic, q_dic, float p_norm, float q_norm):
  """Calculates the cross entropy (nats) between two regularly spaced orograms whose bins are in alignment."""
  # Setup position information and return...
  cdef long i = p_start # Could +1, but then q_prev needs setting!
  cdef float p_prev = 0.0, p_curr
  cdef float q_prev = 0.0, q_curr
  cdef float log_q_prev = -150, log_q_curr # -150 ~= log(1e-64)
  cdef float ret = 0.0
  
  # Block indices and pointers...
  cdef long blockindex = i // blocksize
  cdef long blockbase = blockindex * blocksize
  
  if blockindex not in q_dic:
    return -INFINITY
  
  cdef float[:] p_block = p_dic[blockindex]
  cdef float[:] q_block = q_dic[blockindex]
  
  # Loop the linear sections of the two orograms...
  with nogil:
    while i < p_end:
      # Grab values for current position...
      p_curr = p_block[i - blockbase] / (p_norm * spacing)
      q_curr = q_block[i - blockbase] / (q_norm * spacing)
      log_q_curr = log(q_curr) if q_curr>=1e-64 else -150
      
      # Calculate crossentropy for section...
      ret += spacing * section_crossentropy(p_prev, p_curr, q_prev, q_curr, log_q_prev, log_q_curr)
      
      # Increment...
      i += 1
      p_prev = p_curr
      q_prev = q_curr
      log_q_prev = log_q_curr
      
      # Check if we have moved blocks...
      while (i - blockbase) >= blocksize:
        # Forwards!..
        blockindex += 1
        blockbase += blocksize
        
        with gil:
          # Update p pointer...
          if blockindex not in p_dic:
            # Last term of last block...
            ret += spacing * section_crossentropy(p_prev, 0.0, q_prev, 0.0, log_q_prev, -150)
            
            # Move on...
            i = blockbase + blocksize
            p_prev = 0.0
            q_prev = 0.0
            log_q_prev = -150 # -150 ~= log(1e-64)
            
            if i>= p_end:
              break
            else:
              continue
        
          p_block = p_dic[blockindex]
        
          # Update q pointer...
          if blockindex not in q_dic:
            return -INFINITY
        
          q_block = q_dic[blockindex]

  return ret



cpdef float misaligned_crossentropy(int p_start, int p_end, p_dic, q_dic, float p_norm, float q_norm, long p_blocksize, long q_blocksize, float p_spacing, float q_spacing):
  """Calculates the cross entropy (nats) between two regularly spaced orograms where the bins are not aligned."""
  cdef float ret = 0.0
  
  # Setup coordinates, right at the start...
  cdef long pi = p_start
  cdef float pt = 0.0
  
  cdef float qt = pi * p_spacing / q_spacing
  cdef long qi = long(qt)
  qt -= qi

  # Block indices and pointers...
  cdef long p_blockindex = pi // p_blocksize
  cdef long p_blockbase = p_blockindex * p_blocksize
  
  cdef long q_blockindex = qi // q_blocksize
  cdef long q_blockbase = q_blockindex * q_blocksize
  
  cdef float[:] p_block = p_dic[p_blockindex] if p_blockindex in p_dic else None
  cdef float[:] q_block = q_dic[q_blockindex] if q_blockindex in q_dic else None
  
  # Current values, as in the probability at pi and qi...
  cdef float p_curr = 0.0
  cdef float q_curr = (q_block[qi - q_blockbase] / (q_norm * p_spacing)) if q_block is not None else 0.0
  
  # Next values, with a dummy value so they get filled in...
  cdef float p_next = -1.0, q_next = -1.0
  
  # Variables needed within loop...
  cdef float pt_next, qt_next, width
  cdef float p0, p1 = -1, q0, q1 = 0, log_q0, log_q1 = -150
  
  
  # Loop each linear section of the two orograms, which will be irregular due to the misalignment...
  with nogil:
    while pi < p_end:
      # If p_next is a dummy value fill in...
      if p_next < 0.0:
        if pi + 1 - p_blockbase >= p_blocksize:
          p_blockindex += 1
          p_blockbase += p_blocksize
          with gil:
            p_block = p_dic[p_blockindex] if p_blockindex in p_dic else None

        p_next = (p_block[pi + 1 - p_blockbase] / (p_norm * p_spacing)) if p_block is not None else 0.0
      
      # If q_next is a dummy value fill in...
      if q_next < 0.0:
        if qi + 1 - q_blockbase >= q_blocksize:
          q_blockindex += 1
          q_blockbase += q_blocksize
          with gil:
            q_block = q_dic[q_blockindex] if q_blockindex in q_dic else None

        q_next = (q_block[qi + 1 - q_blockbase] / (q_norm * q_spacing)) if q_block is not None else 0.0

      # Work out what the next step is - either qi or pi is going to get incremented, the other will probably go part way, but represent the move by the t value of the ending position...
      if (pi + 1) * p_spacing < (qi + 1) * q_spacing:
        # Incrementing pi...
        pt_next = 1.0
        qt_next = (pi + 1) * p_spacing / q_spacing - qi
        width = p_spacing * (1 - pt)
      
      else:
        # Incrementing qi...
        pt_next = (qi + 1) * q_spacing / p_spacing - pi
        qt_next = 1.0
        width = q_spacing * (1 - qt)
      
      # Interpolate as needed to get parameters, reusing if possible...
      if p1 < 0.0:
        p0 = ((1-pt)*p_curr + pt*p_next) if pt>1e-12 else p_curr
        q0 = ((1-qt)*q_curr + qt*q_next) if qt>1e-12 else q_curr
        log_q0 = log(q0) if q0>=1e-64 else -150
      
      else:
        p0 = p1
        q0 = q1
        log_q0 = log_q1

      p1 = ((1-pt_next)*p_curr + pt_next*p_next) if pt_next<(1-1e-12) else p_next
      q1 = ((1-qt_next)*q_curr + qt_next*q_next) if qt_next<(1-1e-12) else q_next
      log_q1 = log(q1) if q1>=1e-64 else -150
      
      # Calculate cross entropy for this section...
      ret += width*section_crossentropy(p0, p1, q0, q1, log_q0, log_q1)
      
      # Move indices forwards to next linear section...
      if pt_next > (1.0 - 1e-12):
        pi += 1
        p_curr = p_next
        p_next = -1.0
        pt = 0.0
      else:
        pt = pt_next
        
      if qt_next > (1.0 - 1e-12):
        qi += 1
        q_curr = q_next
        q_next = -1.0
        qt = 0.0
      else:
        qt = qt_next
  
  return ret
