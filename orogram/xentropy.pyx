# Copyright 2022 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import numpy
cimport numpy

from libc.math cimport log, fabs, INFINITY



cpdef float reg_entropy(float[:] prob, float spacing):
  """Calculates entropy of a regular orogram using the analytic equations I've derived."""
  cdef int i
  cdef float ret = 0.0
  cdef float plogp_last, plogp, pdelta
  
  plogp_last = prob[0] * log(prob[0]) if prob[0]>1e-12 else 0.0
  
  for i in range(1, prob.shape[0]):
    
    # Calculate the p*log(p) term for the current value...
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



cpdef float irr_entropy(float[:] x, float[:] prob):
  """Calculates entropy of an irregular orogram using the analytic equations I've derived."""
  cdef int i
  cdef float ret = 0.0
  cdef float plogp_last, plogp, xdelta, pdelta

  plogp_last = prob[0] * log(prob[0]) if prob[0]>1e-12 else 0.0
  
  for i in range(1, prob.shape[0]):
    
    # Calculate the p*log(p) term for the current value...
    plogp = prob[i] * log(prob[i]) if prob[i]>1e-12 else 0.0
    
    # Do the safe bit...
    xdelta = x[i] - x[i-1]
    ret -= 0.5 * xdelta * (plogp_last + plogp)
    
    # Do the bit that might be unstable if the delta is too small...
    pdelta = prob[i] - prob[i-1]
    if fabs(pdelta)>1e-12:
      ret -= 0.25 * xdelta * (prob[i-1]*prob[i-1] - prob[i]*prob[i] + 2 * (prob[i-1]*plogp - prob[i]*plogp_last)) / pdelta
    
    # Move to next...
    plogp_last = plogp
  
  return ret



cdef float section_crossentropy(float p0, float p1, float q0, float q1, double log_q0, double log_q1) nogil:
  """Solves integral for a linear section as needed to calculate cross entropy. Internals carefully ordered to maximise numerical stability, but really don't think that's required - just being paranoid."""
  # Early exit if it's zero...
  if p0<1e-12 and p1<1e-12:
    return 0.0
  
  # Second and third terms but only if not too close to the limit...
  cdef float ret = 0
  cdef float qdelta = q1 - q0
  if fabs(qdelta)>1e-12:
    # Second term, partial division...
    ret = ((p1*q0*q0 - p0*q1*q1) * (log_q1 - log_q0)) / qdelta
    
    # Third term without division...
    ret += 0.5 * ((3*p0 + p1)*q1 - (p0 + 3*p1)*q0)
    
    # Remaining division required to get both terms correct...
    ret /= 2 * qdelta

  # The first term...
  ret -= 0.5 * (p0*log_q0 + p1*log_q1)
  
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
        p0 = p1# Interpolate as needed to get parameters, reusing if possible...
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



cpdef float irregular_crossentropy(float[:] p_x, float[:] p_y, float[:] q_x, float[:] q_y) nogil:
  """Calculates the cross entropy (nats) between two irregularly spaced orograms."""
  cdef float ret = 0.0
  
  # Position in sequence state (includes binary search for q)...
  cdef long pi = 0
  cdef float pt = 0.0
  
  cdef long low, high, half
  cdef long qi
  cdef float qt
  
  if p_x[pi] < q_x[0]:
    qi = 0
    qt = 0.0
  
  elif p_x[pi] > q_x[q_x.shape[0] - 1]:
    return -INFINITY
  
  else:
    low = 0
    high = q_x.shape[0] - 1
    
    while low+1 < high:
      half = (low + high) // 2
      if p_x[pi] < q_x[half]:
        high = half
      else:
        low = half
    
    qi = low
    if qi+1 < q_x.shape[0]:
      qt = (p_x[pi] - q_x[qi]) / (q_x[qi+1] - q_x[qi])
    else:
      qt = 0.0
  
  # Previous value, and next value, with -1 as dummy...
  cdef float p_curr = p_y[0]
  cdef float q_curr = (1-qt) * q_y[qi] + qt * q_y[qi+1] if qi+1 < q_x.shape[0] else q_y[qi]
  
  cdef float p_next = -1.0, q_next = -1.0
  
  # Further variables needed...
  cdef float p_x_next = 0.0, q_x_next = 0.0
  cdef float pt_next, qt_next, width
  cdef float p0, p1 = -1, q0, q1 = 0, log_q0, log_q1 = -150
  
  # Loop through the entirety of p, calculating for each segment in turn...
  while pi+1 < p_x.shape[0]:
    # If p_next is a dummy value fill in...
    if p_next < 0.0:
      if pi+1 < p_y.shape[0]:
        p_x_next = p_x[pi+1]
        p_next = p_y[pi+1]
      else:
        p_x_next = INFINITY
        p_next = 0.0
    
    # If q_next is a dummy value fill in...
    if q_next < 0.0:
      if qi+1 < q_y.shape[0]:
        q_x_next = q_x[qi+1]
        q_next = q_y[qi+1]
      else:
        q_x_next = INFINITY
        q_next = 0.0
    
    # Work out what the next step is - either qi or pi is going to get incremented, the other will probably go part way, but represent the move by the t value of the ending position...
    if p_x_next < q_x_next:
      pt_next = 1.0
      qt_next = (p_x_next - q_x[qi]) / (q_x_next - q_x[qi])
      width = (p_x_next - p_x[pi]) * (1 - pt)
      
    else:
      pt_next = (q_x_next - p_x[pi]) / (p_x_next - p_x[pi])
      qt_next = 1.0
      width = (q_x_next - q_x[qi]) * (1 - qt)
    
    # Interpolate as needed to get parameters, reusing if possible...
    if p1 < 0.0:
      p0 = ((1-pt)*p_curr + pt*p_next) if pt>1e-12 else p_curr
      q0 = ((1-qt)*q_curr + qt*q_next) if qt>1e-12 else q_curr
      log_q0 = log(q0) if q0>=1e-64 else -150
      
    else:
      p0 = p1# Interpolate as needed to get parameters, reusing if possible...
      q0 = q1
      log_q0 = log_q1

    p1 = ((1-pt_next)*p_curr + pt_next*p_next) if pt_next<(1-1e-12) else p_next
    q1 = ((1-qt_next)*q_curr + qt_next*q_next) if qt_next<(1-1e-12) else q_next
    log_q1 = log(q1) if q1>=1e-64 else -150
    
    # Calculate cross entropy for this section...
    ret += width*section_crossentropy(p0, p1, q0, q1, log_q0, log_q1)
    
    # Move indices forward to next linear section...
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
  
  # Return total...
  return ret
