# Copyright 2022 Tom SF Haines

import sys
from collections import namedtuple
import numpy


try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})

except:
  pass

from . import xentropy
from . import simplify

from .regorogram import RegOrogram



# A named tuple that is returned by Orogram.simplify()...
SimplifyResult = namedtuple('SimplifyResult', ['solution', 'cost', 'kept', 'priorcost', 'priorall', 'dominant'])
# solution = The Orogram that is the solution
# cost - Cost of the solution
# kept - Boolean array indicating which bin centres from the input have been kept (True if included in solution, False if not)
# priorcost - Cost of the prior only; can get data term by subtracting this from cost
# priorall - Cost of prior if all bins are included, as for the input. This plus the entropy multiplied by the sample count is the cost of the input.
# dominant - Total number of dominant triangles generated, an indication of peak memory usage/efficiency as these are the triangles that have to be kept.



class Orogram:
  """An orogram with irregularly spaced bins - an arbitrary piecewise linear PDF in other words. The function is fixed on construction - this is immutable. Has a fairly rich interface."""
  __slots__ = ('_x', '_y', '_cdf')
  
  
  def __init__(self, x, y = None, cdf = None, norm=True, copy=True):
    """You construct it with two 1D arrays, containing x and y that define the values that make up the distribution. x must be increasing, all y's must be positive. By default it will renormalise if the area under the line is not one, but set norm=False if you can guarantee that is already the case. Note that it internally keeps a cdf, aligned with x/y - can provide if already calculated. Can alternatively provide a RegOrogram which it will automatically convert."""
    
    # Handle request to convert a regular orogram...
    if isinstance(x, RegOrogram):
      x, y = x.graph()
      norm = False
      copy = False
    
    # Record arrays...
    self._x = numpy.array(x, dtype=numpy.float32, copy=copy, order='A')
    self._y = numpy.array(y, dtype=numpy.float32, copy=copy, order='A')
    
    # Normalise if needed...
    if norm or cdf is None:
      mass = 0.5 * (self._y[:-1] + self._y[1:]) * (self._x[1:] - self._x[:-1])
    
    if norm:
      total = mass.sum()
      self._y /= total
    
    # Lil' safety...
    assert(len(self._x.shape)==1)
    assert(len(self._y.shape)==1)
    assert(self._x.shape[0]==self._y.shape[0])
    assert(numpy.all(numpy.diff(self._x) > 0.0))
    
    # Fill in the CDF array...
    if cdf is not None:
      self._cdf = numpy.array(cdf, dtype=numpy.float32, copy=copy, order='A')
    
    else:
      self._cdf = numpy.empty(x.shape, dtype=numpy.float32)
      self._cdf[0] = 0.0
      self._cdf[1:] = numpy.cumsum(mass)
      self._cdf[-1] = 1.0

  
  def copy(self):
    """Makes a copy."""
    return Orogram(self._x, self._y, cdf = self._cdf, norm=False)
  
  __copy__ = copy
  
  
  def __sizeof__(self):
    ret = super().__sizeof__()
    ret += sys.getsizeof(self._x)
    ret += sys.getsizeof(self._y)
    ret += sys.getsizeof(self._cdf)
    return ret


  def __len__(self):
    """Returns a length that can be seen as the number of parameters, in the sense of being the number of values needed to define the heights (y) for the range that is used. It doesn't include the actual positions (x) of these values in the count, so you may want to double this number for a more realistic count; done this way to be similar to RegOrogram() and because it doubles as the end value of a loop over all bins."""
    return self._x.shape[0]


  def min(self):
    """Returns the minimum value to have any weight assigned to it. Note that the probability at this point is likely to be be zero."""
    return self._x[0]
  
  
  def max(self):
    """Returns the maximum value to have any weight assigned to it. Note that the probability at this point is likely to be zero."""
    return self._x[-1]
  
  
  def _mass(self):
    """Returns the total probability mass - should always be 1, so this is for debugging only."""
    return 0.5 * ((self._y[:-1] + self._y[1:]) * (self._x[1:] - self._x[:-1])).sum()


  def center(self, i):
    """Given the index of a bin (or many - vectorised) returns the position of the center of that bin. The bins range from 0 to len(self)-1 inclusive; out of range indices will be clamped."""
    return numpy.take(self._x, i, mode='clip')


  def prob(self, i):
    """Given the index of a bin (or many - vectorised) returns the probability at the centre of that bin. Note that this is a pdf, not a pmf, and you need linear interpolation to get between bins (which is what you get for arbitrary values if you call this object). The bins range from 0 to len(self)-1 inclusive; out of range indices will return 0."""
    if numpy.ndim(i)==0:
      return self._y[i] if i>=0 and i<self._y.shape[0] else 0.0
    
    else:
      i = numpy.asarray(i, dtype=int)
      ret = numpy.take(self._y, i, mode='clip')
      ret[(i<0) | (i>=self._y.shape[0])] = 0.0
      return ret


  def __call__(self, x):
    """Evaluates the probability at the given x, including linear interpolation between bin centres. Vectorised."""
    return numpy.interp(x, self._x, self._y, 0.0, 0.0)
  
  
  def modes(self, threshold = 1e-6):
    """Returns an array containing the position of every mode."""
    keep = numpy.nonzero(((self._y[:-2] + threshold) < self._y[1:-1]) & ((self._y[2:] + threshold) < self._y[1:-1]))
    return self._x[keep[0] + 1]


  def binmodes(self, threshold = 1e-6):
    """Identical to modes() but returns the bin indices instead."""
    keep = numpy.nonzero(((self._y[:-2] + threshold) < self._y[1:-1]) & ((self._y[2:] + threshold) < self._y[1:-1]))
    return keep[0] + 1


  def highest(self):
    """Returns the position with the highest probability."""
    i = numpy.argmax(self._y)
    return self._x[i]
  
  
  def binhighest(self):
    """Returns the index of the bin with the highest probability."""
    return numpy.argmax(self._y)
  
  
  def graph(self):
    """Returns two aligned vectors, as a convenience function for generating the arrays to hand to a graph plotting function. Return is a tuple of (x, y), x being the bin centre and y the probability."""
    retx = self._x.view()
    rety = self._y.view()
    
    retx.flags.writeable = False
    rety.flags.writeable = False
    
    return retx, rety
  
  
  @staticmethod
  def _fetchx(orogram):
    if isinstance(orogram, RegOrogram):
      return ororam.center(numpy.arange(orogram._low-1, orogram._high+2))
    
    else:
      return orogram._x
  
  
  @staticmethod
  def mixture(orograms, weights):
    """Returns a new Orogram that is constructed as a mixture of orograms - inputs are a list of Orogram objects and a corresponding list of weights, matching up with each Orogram. Will handle any RegOrogram objects that are included. Note that the return value can have as many bin centres as all inputs combined (duplicates are merged), so doing this iteratively without some kind of simplification step is in general unwise."""

    # Extract the list of bin centres...
    centers = numpy.concatenate([Orogram._fetchx(og) for og in orograms])
    
    # Sort and dedup...
    centers.sort(kind='mergesort')
    
    unique = numpy.ones(centers.shape, dtype=bool)
    numpy.not_equal(centers[1:], centers[:-1], out=unique[1:])
    centers = centers[unique]
    
    # Sample mixture at each centre...
    values = numpy.zeros(centers.shape, dtype=numpy.float32)
    div = sum(weights)
    
    for og, w in zip(orograms, weights):
      values += (w / div) * og(centers)
    
    # Construct and return object...
    return Orogram(centers, values, norm=False, copy=False)
  
  
  @staticmethod
  def product(orograms):
    """Returns a new Orogram that is constructed as a product of ororams - input is a list of Orogram objects; it is renormalised. It can raise a ZeroDivisionError exception if the intersection of probability mass is null. Will handle any RegOrogram objects that are included. Note that the return value can have as many bin centres as all inputs combined (duplicates are merged), so doing this iteratively without some kind of simplification step is in general unwise."""
    
    # Extract the list of bin centres...
    centers = numpy.concatenate([Orogram._fetchx(og) for og in orograms])
    
    # Sort and dedup...
    centers.sort(kind='mergesort')
    
    unique = numpy.ones(centers.shape, dtype=bool)
    numpy.not_equal(centers[1:], centers[:-1], out=unique[1:])
    centers = centers[unique]
    
    # Sample mixture at each centre...
    values = numpy.ones(centers.shape, dtype=numpy.float32)
    
    for og in orograms:
      values *= og(centers)
    
    # Construct and return object...
    return Orogram(centers, values, copy=False)
  

  def cdf(self, x):
    """Evaluates the cdf at the given x. Vectorised."""
    return numpy.interp(x, self._x, self._cdf, 0.0, 0.0)


  def bincdf(self, i):
    """Evaluates the cdf at the given bin centre. Vectorised."""
    return numpy.take(self._cdf, i, mode='clip')


  def cdfgraph(self):
    """Returns two aligned vectors, as a convenience function for generating the arrays to hand to a graph plotting function to get the cdf. Return is a tuple of (x, y), x being the bin centre and y the cdf."""
    retx = self._x.view()
    rety = self._cdf.view()
    
    retx.flags.writeable = False
    rety.flags.writeable = False
    
    return retx, rety

  
  def invcdf(self, uni):
    """Evaluates the inverse cdf for the given value in [0, 1); if the value was drawn from a uniform distribution then this is identical to draw(). Vectorised."""
    after = numpy.searchsorted(self._cdf, uni)
    t = (uni - self._cdf[after-1]) / (self._cdf[after] - self._cdf[after-1])
    
    return (1-t) * self._x[after-1] + t * self._x[after]
  
  
  def draw(self, size = None, rng = None):
    """Draws samples from the distribution - first parameter is how many (defaults to None, which is one sample, not in an array; can be a tuple for a nD array), second something that numpy.random.default_rng() is happy to accept."""
    
    # Fetch the noise...
    rng = numpy.random.default_rng(rng)
    noise = rng.random(size, dtype=numpy.float32)
    
    # Do the inverse CDF dance...
    return self.invcdf(noise)


  def median(self):
    """Returns the median."""
    after = numpy.searchsorted(self._cdf, 0.5)
    t = (0.5 - self._cdf[after-1]) / (self._cdf[after] - self._cdf[after-1])
    
    return (1-t) * self._x[after-1] + t * self._x[after]


  def mean(self):
    """Returns the mean."""
    return numpy.average(self._x, weights=self._y)


  def var(self):
    """Returns the variance."""
    return numpy.cov(self._x, aweights=self._y).item()


  def meanvar(self):
    """Returns a tuple containing (mean, variance)."""
    return mean(), var()


  def entropy(self):
    """Calculates the entropy analytically, in nats."""
    return xentropy.irr_entropy(self._x, self._y)
  
  
  def entropynumint(self, samples=1024*1024, threshold=1e-12):
    """Numerical integration version of entropy(); for testing only as obviously much slower. In nats."""
    
    # Evaluate across range of distribution...
    x = numpy.linspace(self._x[0], self._x[-1], samples)
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
    return xentropy.irregular_crossentropy(self._x, self._y, q._x, q._y)


  def crossentropynumint(self, q, samples=1024*1024, threshold=1e-12):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding. This version uses numerical integration and exists for testing only - slow."""
    
    # Evaluate p across range of self distribution...
    x = numpy.linspace(self._x[0], self._x[-1], samples)
    p = self(x)
    delta = x[1] - x[0]
    
    # Filter out the zeros - log gets very unhappy about them...
    good = p>=threshold
    x = x[good]
    p = p[good]
    
    # Evaluate log(q)... 
    qlog = numpy.log(numpy.maximum(q(x), 1e-32))
    
    # Cross entropy!..
    return -delta * (p * qlog).sum()


  def crossentropymc(self, q, samples=1024*1024, threshold=1e-12, rng=None):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding. This version uses Monte-Carlo integration and exists for testing only - super slow."""
    
    # Draw and evaluate...
    x = self.draw(samples, rng)
    y = q(x)
    
    # Calculate cross-entropy...
    return -numpy.log(numpy.maximum(y, 1e-32)).mean()
  
  
  def kl(self, q):
    """Calculates the Kullback—Leibler divergence between two distributions, i.e. the expected extra nats needed for encoding data with the distribution represented by this object when the encoder is optimised for the distribution of q, the first parameter to this method. Often denoted D_{KL}(self || q) Convenience method that uses the cross-entropy and entropy methods, i.e. D_{KL}(self || q) = H(self, q) - H(self)"""
    return self.crossentropy(q) - self.entropy()
  
  
  def simplify(self, samples, perbin = 16, continuous = True):
    """Simplifies this orogram, returning a named tuple containing a new, simpler, orogram. Parameters are the number of samples that were used to generate this orogram (not recorded in Orogram object, hence having to pass in) and the expected number of data points per bin centre, which sets the prior. You can also set continuous to False if you don't mind the potential discontinuity of the output Orogram starting/ending with a value that is not zero. Returns a named tuple containing (solution - simplified Orogram, cost - it's cost (negative log probability, includes probability ratios for prior so not true cost), kept - boolean array aligned with input containing True when a point was included in the simplified output, priorcost - the cost of the prior ratio term only (subtract from cost to get cost of data term only), priorall - the cost of the prior ratio on the input, i.e. with all bins kept; add to the input entropy scaled by the sample count to get a comparable cost for the input). Depending on algorithm it may have further parameters."""
    retx, retp, retk, cost, priorcost, priorall, dominant = simplify.dp(self._x, self._y, samples, perbin, continuous)
    ret = Orogram(retx, retp, norm=False, copy=False)
    return SimplifyResult(solution=ret, cost=cost, kept=retk, priorcost=priorcost, priorall=priorall, dominant=dominant)
