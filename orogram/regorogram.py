# Copyright 2022 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys

import numpy


try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})

except:
  pass

from . import regorofun
from . import xentropy



class RegOrogram:
  """An orogram with regularly spaced bins - primarily for incremental data collection as regular spacing makes that easy. Note that this is exactly equivalent to the frequency polygon due to the regular spacing. Has an automatically expanding range, based on a dictionary of blocks, so the only parameter of consequence is the spacing between adjacent bins. Typical usage is to collect data in this with a very fine spacing then simplify to an Orogram with variable spacing that is data driven. Functionality is still pretty rich, just in case you want to set a data-suitable bin width and use it as a direct density estimate. This class is only robust to 10s of millions of samples; after that the numerical precision of float will start to bite. If that is the case fit an orogram to blocks, scale down (*=) and sum (+=)."""
  __slots__ = ('_spacing', '_blocksize', '_data', '_total', '_low', '_high', '_cdf')
  
  
  def __init__(self, spacing = 0.001, blocksize = 1024 * 16):
    """spacing is how wide the bins are, as in bin centre to bin centre; blocksize is how many bin values to record in each block of the backing storage (you can usually leave this alone)."""
    self._spacing = spacing
    self._blocksize = blocksize
    
    self._data = dict() # block # -> float numpy array of counts
    
    self._total = 0.0 # Total weight captured
    self._low = sys.maxsize # Index of lowest nonzero bin
    self._high = -sys.maxsize - 1 # Index of highest nonzero bin
    
    self._cdf = None # None if not calculated, otherwise an array indexed by bin starting with self._low - 1 at [0]
  
  
  def copy(self):
    """Makes a copy."""
    ret = RegOrogram(self._spacing, self._blocksize)
    
    for k, v in self._data.items():
      ret._data[k] = v.copy()
    
    ret._total = self._total
    ret._low = self._low
    ret._high = self._high
    
    ret._cdf = self._cdf # Never edited so safe
    
    return ret
  
  __copy__ = copy
  
  
  def __sizeof__(self):
    ret = super().__sizeof__()
    
    ret += sys.getsizeof(self._spacing)
    ret += sys.getsizeof(self._blocksize)
    ret += sys.getsizeof(self._blocksize)
    ret += sys.getsizeof(self._data)
    ret += sys.getsizeof(self._total)
    ret += sys.getsizeof(self._low)
    ret += sys.getsizeof(self._high)
    ret += sys.getsizeof(self._cdf)
    
    for k, v in self._data.items():
      ret += sys.getsizeof(k)
      ret += sys.getsizeof(v)
    
    return ret
  
  
  def __len__(self):
    """Returns a length that can be seen as the number of parameters, in the sense of being the number of values needed to define the heights for the range that is used. It includes the zeros either side in that count. Note what this excludes: start position and run length of each range."""
    return regorofun.length(self._data, self._blocksize, self._low, self._high)
  
  
  def spacing(self):
    """Returns the spacing this object was constructed with."""
    return self._spacing
  
  
  def blocksize(self):
    """Returns the blocksize this object was constructed with."""
    return self._blocksize
  
  
  def add(self, x, weight = 1.0, smooth=False):
    """Adds the set of samples x (single value or array) to the orogram. Can optionally include a weight, to count each sample as being some multiple of one (single value or array; broadcast as required). With the final parameter, smooth, if it is False you get the maximum likelihood estimate, i.e. it's the same as a histogram because it adds all of the weight to the closest bin, as that will increase the probability of the point the most. This is arguably the correct mode, hence being the default. If changed to True however it interpolates the weight between bins, which gives a smoother result but one that is no longer the most probable."""
    x = numpy.atleast_1d(x)
    weight = numpy.broadcast_to(weight, x.shape).astype(numpy.float32)
    
    x = x / self._spacing
    
    if smooth:
      base = numpy.floor(x).astype(int)
      t = (x - base).astype(numpy.float32)
      regorofun.smoothadd(self._data, self._blocksize, base, t, weight)
    
    else:
      base = numpy.floor(x+0.5).astype(int)
      regorofun.add(self._data, self._blocksize, base, weight)
      
    
    self._total += weight.sum()
    self._low = min(self._low, base.min())
    self._high = max(self._high, base.max() + (1 if smooth else 0))
    
    self._cdf = None
  
  
  def binadd(self, base, density, total = None):
    """Lets you add values directly to the bins, where those values can be seen as the number of samples worth of mass. The add function can be made to do the same, but this is faster if you already have data in this form. base is the bin index of density[0]. You can optionally provide total, the sum of the density array if you know it."""
    regorofun.binadd(self._data, self._blocksize, base, density.astype(numpy.float32))
    
    if total is None:
      total = density.sum()
    
    self._total += total
    self._low = min(self._low, base)
    self._high = max(self._high, base + density.shape[0] - 1)
    
    self._cdf = None
  
  
  def bake_cdf(self, cdf, start, end, weight = 1, vectorised = True):
    """Lets you 'bake' an arbitrary distribution into this representation, in terms of its cdf. You have to provide a function for evaluating the CDF of that distribution, plus the range to evaluate (it ensures the mass in the range sums to 1). By default it assumes that the CDF functon is vectorised but you can indicate it is not (vectorised parameter); that will be slow however. Solves the linear equation so that the areas between the bin centres match the mass of the CDF. Note that baking is additive, so the values will be summed on top of anything already within the model, by default as though this is a single samples worth of evidence, but the 'weight' parameter lets you change that to whatever they want."""
    
    # Calculate bin range...
    low = int(numpy.floor(start / self._spacing))
    high = int(numpy.ceil(end / self._spacing))
    
    # Generate array of relevant bin centres...
    centers = numpy.arange(low, high+1, dtype=numpy.float32)
    centers *= self._spacing
    
    # Evaluate cdf at each bin centre...
    if vectorised:
      cummass = cdf(centers)
      
    else:
      cummass = numpy.empty(centers.shape, dtype=numpy.float32)
      for i in range(cummass.shape[0]):
        cummass[i] = cdf(centers[i])
    
    # Convert to mass between bin centres, renormalise, then drop into bin centres - this does smooth slightly, but avoids the instability of attempting to optimise directly...
    between = (cummass[1:] - cummass[:-1]).astype(numpy.float32)
    between /= between.sum()
    
    density = numpy.zeros(cummass.shape, dtype=numpy.float32)
    density[:-1] += 0.5 * between
    density[1:] += 0.5 * between
    
    # cdf's often involve evaluating tricky transcendental functions, and numerical precision noise can dominate for tiny values, including taking it negative - clean that up...
    density[density<1e-12] = 0.0
    
    # Scale by weight...
    density *= weight
    
    # Update data structure by adding in these values...
    self.binadd(low, density, weight)
  
  
  def sum(self):
    """Returns the sum of all the bins, in effect the total weight recorded. Note that it's a float because fractional weights can be added (plus weight is distributed between bin centres)."""
    return self._total
  
  
  def min(self):
    """Returns the minimum value to have any weight assigned to it. Note that the probability at this point will be zero, but nonzero if you move up by some epsilon."""
    return (self._low - 1) * self._spacing
  
  
  def max(self):
    """Returns the maximum value to have any weight assigned to it. Note that the probability at this point will be zero, but nonzero if you move down by some epsilon."""
    return (self._high + 1) * self._spacing


  def binmin(self):
    """Returns the lowest index of a bin that contains weight minus one. The minus one means you have the same bin that min generates and it fully covers the range that actually contains probability mass."""
    return self._low - 1
  
  
  def binmax(self):
    """Returns the highest index of a bin that contains weight plus one. The plus one means you have the same bin that max generates and it fully covers the range that actually contains probability mass."""
    return self._high + 1


  def center(self, i):
    """Given the index of a bin converts it to the position of the centre of the bin. Obviously just a multiplication, but abstracted for consistency and to avoid mistakes. Vectorised."""
    return numpy.asarray(i, dtype=numpy.float32) * self._spacing


  def weight(self, i):
    """Given the index of a bin (or many - vectorised) returns how much weight has landed in that bin. This is often the number of samples that have landed in that bin."""
    if numpy.ndim(i)==0:
      block = i // self._blocksize
      if block in self._data:
        return self._data[block][i - block*self._blocksize]
      else:
        return 0.0
    
    else:
      i = numpy.asarray(i, dtype=int)
      ret = numpy.empty(i.shape, dtype=numpy.float32)
      regorofun.binweight(self._data, self._blocksize, i, ret)
      return ret
  
  
  def prob(self, i):
    """Given the index of a bin (or many - vectorised) returns the probability at the centre of that bin. Note that this is a pdf, not a pmf, and you need linear interpolation to get between bins (which is what you get for arbitrary values if you call this object)."""
    if numpy.ndim(i)==0:
      block = i // self._blocksize
      if block in self._data:
        return self._data[block][i - block*self._blocksize] / (self._spacing * self._total)
      else:
        return 0.0
    
    else:
      i = numpy.asarray(i, dtype=int)
      ret = numpy.empty(i.shape, dtype=numpy.float32)
      regorofun.binweight(self._data, self._blocksize, i, ret)
      ret /= self._spacing * self._total
      return ret
  
  
  def __call__(self, x):
    """Evaluates the probability at the given x, including linear interpolation between bin centres. Vectorised."""
    x = numpy.asarray(x) / self._spacing
    base = numpy.floor(x).astype(int)
    t = x - base
    
    if numpy.ndim(x)==0:
      ret = 0.0
      
      block1 = base // self._blocksize
      block1i = base - block*self._blocksize
      
      if block1 in self._data:
        dic = self._data[block1]
        ret += (1 - t) * dic[block1i]
        if block1i+1 < self._blocksize:
          ret += t * dic[block1i+1]
          
      elif (block1i+1 == self._blocksize) and (block1+1 in self._data):
        ret += t * self._data[block1+1][0]
      
      return ret / (self._spacing * self._total)
    
    else:
      ret = numpy.zeros(base.shape, dtype=numpy.float32)
      regorofun.weight(self._data, self._blocksize, base, t.astype(numpy.float32), ret)
      ret /= self._spacing * self._total
      return ret
  
  
  def modes(self):
    """Returns an array containing the position of every mode. Pretty useless under default use, but becomes useful if bin spacing is made large enough to regularise."""
    ret = regorofun.modes(self._data, self._blocksize, self._low, self._high)
    return ret.astype(numpy.float32) * self._spacing
  
  
  def binmodes(self):
    """Identical to modes() but returns the bin indices instead."""
    return regorofun.modes(self._data, self._blocksize, self._low, self._high)

  
  def highest(self):
    """Returns the position with the highest probability."""
    return self.binhighest() * self._spacing

  
  def binhighest(self):
    """Returns the bin with the highest probability."""
    best = self._low
    weight = 0.0
    
    for k, v in self._data.items():
      localbest = numpy.argmax(v)
      localweight = v[localbest]
      
      if localweight > weight:
        best = localbest + k*self._blocksize
        weight = localweight
    
    return best
  
  
  def graph(self, start = None, end = None):
    """Returns two aligned vectors, as a convenience function for generating the arrays to hand to a graph plotting function. Return is a tuple of (x, y), x being the bin centre and y the probability. You can give the start and end bin if you want (inclusive, exclusive), but it defaults to the range of the observed data (going one wider so the end points have probability zero)."""
    if start is None:
      start = self._low - 1
    
    if end is None:
      end = self._high + 2
    
    i = numpy.arange(start, end)
    retx = self.center(i)
    rety = self.prob(i)
    
    return retx, rety


  def __iadd__(self, other):
    """Allows you to add the data collected in another RegOrogram into this one (+=). Will be efficient and without data loss if the spacing and block size match, otherwise it will switch to interpolating bin positions. Note that the interpolation will have a smoothing effect."""
    if numpy.fabs(self._spacing - other._spacing) < 1e-12 and self._blocksize==other._blocksize:
      # Efficient version...
      for k, v in other._data.items():
        if k in self._data:
          self._data[k] += v
        
        else:
          self._data[k] = v.copy()
    
    else:
      # Inefficient version - go via add() to get interpolation...
      indices = numpy.arange(other._blocksize)
      for k, v in other._data.items():
        x = other.center(k + indices)
        send = numpy.nonzero(v)
        self.add(x[send], v[send], True)
    
    # Update the stats...
    self._total += other._total
    self._low = min(self._low, other._low)
    self._high = max(self._high, other._high)
    
    self._cdf = None
  
  
  def __imul__(self, other):
    """Allows you to scale the weight of all of the collected samples by a positive real number; matters mostly as an operation before +=."""
    for v in other._data.values():
      v[:] *= other
    self._total *= other
  
  
  def _ensure_cdf(self):
    """Ensures that the _cdf array exists, creating it if needed"""
    if self._cdf is None:
      self._cdf = numpy.empty(self._high + 3 - self._low, dtype=numpy.float32)
      regorofun.cdf(self._data, self._blocksize, self._low, self._total, self._cdf)
      self._cdf[-1] = 1.0
  
  
  def cdf(self, x):
    """Evaluates the cdf at the given x, which can be a vector. Note that the cdf is a cached value, recalculated and invalidated as required, so alternating this with an operation such as add() would be very slow."""
    self._ensure_cdf()
    
    x = numpy.asarray(x) / self._spacing - (self._low - 1)
    base = numpy.floor(x).astype(int)
    t = x - base
    
    ret = (1 - t) * self._cdf[numpy.clip(base, 0,  self._cdf.shape[0]-1)]
    ret += t * self._cdf[numpy.clip(base+1, 0,  self._cdf.shape[0]-1)]
    return ret
  
  
  def bincdf(self, i):
    """Evaluates the cdf at the given bin center, which can be a vector. Note that the cdf is a cached value, recalculated and invalidated as required, so alternating this with an operation such as add() would be very slow."""
    self._ensure_cdf()
    
    return self._cdf[numpy.clip(i - (self._low - 1), 0,  self._cdf.shape[0]-1)]
  
  
  def cdfgraph(self, start = None, end = None):
    """Returns two aligned vectors, as a convenience function for generating the arrays to hand to a graph plotting function to get the cdf. Return is a tuple of (x, y), x being the bin centre and y the cdf. You can give the start and end bin if you want (inclusive, exclusive), but it defaults to the range of the observed data (going one wider so the end points are 0 and 1). Note that the cdf is a cached value, recalculated and invalidated as required, so alternating this with an operation such as add() would be very slow."""
    if start is None:
      start = self._low - 1
    
    if end is None:
      end = self._high + 2
    
    i = numpy.arange(start, end)
    retx = self.center(i)
    rety = self.bincdf(i)
    
    return retx, rety
  
  
  def invcdf(self, uni):
    """Evaluates the inverse cdf for the given value in [0, 1); if the value was drawn from a uniform distribution then this is identical to draw(). Vectorised. Note that this uses the cdf which is a cached value, recalculated and invalidated as required, so alternating this with an operation such as add() would be very slow."""
    self._ensure_cdf()
    
    after = numpy.searchsorted(self._cdf, uni)
    t = (uni - self._cdf[after-1]) / (self._cdf[after] - self._cdf[after-1])
    
    return ((after - 1 + t) + (self._low - 1)) * self._spacing
    

  def draw(self, size = None, rng = None):
    """Draws samples from the distribution - first parameter is how many (defaults to None, which is one sample, not in an array; can be a tuple for a nD array), second something that numpy.random.default_rng() is happy to accept. Note that this uses the cdf (inverse cdf transform) which is a cached value, recalculated and invalidated as required, so alternating this with an operation such as add() would be very slow."""
    self._ensure_cdf()
    
    # Fetch the noise...
    rng = numpy.random.default_rng(rng)
    noise = rng.random(size, dtype=numpy.float32)
    
    # Do the inverse CDF dance...
    return self.invcdf(noise)


  def median(self):
    """Returns the median. Uses the cached cdf array, which is recalculated and invalidated as required, so alternating this with an operation such as add() would be very slow."""
    self._ensure_cdf()
    
    after = numpy.searchsorted(self._cdf, 0.5)
    t = (0.5 - self._cdf[after-1]) / (self._cdf[after] - self._cdf[after-1])
    
    return ((after - 1 + t) + (self._low - 1)) * self._spacing


  def mean(self):
    """Returns the mean. Calculation involves looping all data, so is not very efficient."""
    weight = 0.0
    mean = 0.0
    
    # This works because the triangles are symmetric...
    for k,v in self._data.items():
      # Block mean...
      bw = v.sum()
      m = (numpy.arange(self._blocksize) * v).sum() / bw
      bm = (k * self._blocksize + m) * self._spacing
      
      # Incrementally combine with mean thus far...
      weight += bw
      mean += (bm - mean) * bw / weight
    
    return mean
  
  
  def var(self):
    """Returns the variance. Calculation involves looping all data, so is not very efficient."""
    return self.meanvar()[1]
  
  
  def meanvar(self):
    """Returns a tuple containing (mean, variance). Calculation involves looping all data, so is not very efficient, but definitely more efficient than calling mean() and var() separately."""
    weight = 0.0
    mean = 0.0
    scatter = 0.0
    
    pos = numpy.arange(self._blocksize) * self._spacing
    
    for k,v in self._data.items():
      # Block weight, mean and scatter (variance * weight)...
      bw = v.sum()
      blm = (pos * v).sum() / bw
      bm = k * self._blocksize * self._spacing + blm
      bs = ((numpy.square(self._spacing) / 6 + numpy.square(pos)) * v).sum() - numpy.square(blm) * bw
      
      # Incrementally combine into statistics thus far...
      old_weight = weight
      weight += bw
      delta = bm - mean
      mean += delta * bw / weight
      scatter += bs + delta * delta * (old_weight * bw) / weight
    
    return mean, scatter / weight

  
  def entropy(self):
    """Calculates the entropy analytically, in nats."""
    # Evaluate probabilities...
    i = numpy.arange(self._low - 1, self._high + 2)
    prob = self.prob(i)
    
    # Entropy!..
    return xentropy.reg_entropy(prob, self._spacing)
  
  
  def entropynumint(self, samples=1024*1024, threshold=1e-12):
    """Numerical integration version of entropy(); for testing only as obviously much slower. In nats."""
    
    # Evaluate across range of distribution...
    x = numpy.linspace(self.center(self._low - 1), self.center(self._high + 2), samples)
    y = self(x)
    delta = x[1] - x[0]
    
    # Filter out the zeros - log gets very unhappy about them...
    y = y[y>=threshold]
    
    # Entropy!..
    return -delta * (y * numpy.log(y)).sum()
  
  
  def entropymc(self, samples=1024*1024, threshold=1e-12, rng=None):
    """Monte-Carlo integration version of entropy. Super slow of course so just for testing as it also doubles as a good sanity check of sampling. In nats."""
    
    # Draw and evaluate...
    x = self.draw(samples, rng)
    y = self(x)
    
    # Entropy!..
    return -numpy.log(numpy.maximum(y, 1e-12)).mean()
  
  
  def crossentropy(self, q):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding."""
    if numpy.fabs(self._spacing - q._spacing) < 1e-12 and self._blocksize==q._blocksize:
      # Efficient version - the change points are aligned...
      return xentropy.aligned_crossentropy(self._blocksize, self._spacing, self._low-1, self._high+2, self._data, q._data, self._total, q._total)
    
    else:
      # Inefficient version - the change points are not aligned...
      return xentropy.misaligned_crossentropy(self._low-1, self._high+2, self._data, q._data, self._total, q._total, self._blocksize, q._blocksize, self._spacing, q._spacing)


  def crossentropynumint(self, q, samples=1024*1024, threshold=1e-12):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding. This version uses numerical integration and exists for testing only - slow."""
    
    # Evaluate p across range of self distribution...
    x = numpy.linspace(self.center(self._low - 1), self.center(self._high + 2), samples)
    p = self(x)
    delta = x[1] - x[0]
    
    # Filter out the zeros - log gets very unhappy about them...
    good = p>=threshold
    x = x[good]
    p = p[good]
    
    # Evaluate log(q)... 
    qlog = numpy.log(numpy.maximum(q(x), 1e-32))
    
    # Crossentropy!..
    return -delta * (p * qlog).sum()


  def crossentropymc(self, q, samples=1024*1024, threshold=1e-12, rng=None):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding. This version uses Monte-Carlo integration and exists for testing only - super slow."""
    
    # Draw and evaluate...
    x = self.draw(samples, rng)
    y = q(x)
    
    # Calculate cross-entropy...
    return -numpy.log(numpy.maximum(y, 1e-32)).mean()
  
  
  def kl(self, q):
    """Calculates the Kullback—Leibler divergence between two distributions, i.e. the expected extra nats needed for encoding data with the distribution represented by this object when the encoder is optimised for the distribution of q, the first parameter to this method. Convenience method that uses the cross-entropy and entropy methods."""
    return self.crossentropy(q) - self.entropy()
