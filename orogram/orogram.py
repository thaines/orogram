# Copyright 2022 Tom SF Haines

import sys

import numpy


try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})

except:
  pass


try:
  from . import xentropy

except ImportError:
  import xentropy


try:
  from .regorogram import RegOrogram
except ImportError:
  from regorogram import RegOrogram



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


  def center(self, i):
    """Given the index of a bin (or many - vectorised) returns the position of the center of that bin. The bins range from 0 to len(self)-1 inclusive; out of range indices will be clamped."""
    return numpy.take(self._x, i, mode='clip')


  def prob(self, i):
    """Given the index of a bin (or many - vectorised) returns the probability at the center of that bin. Note that this is a pdf, not a pmf, and you need linear interpolation to get between bins (which is what you get for arbitrary values if you call this object). The bins range from 0 to len(self)-1 inclusive; out of range indices will return 0."""
    if numpy.ndim(i)==0:
      return self._y[i] if i>=0 and i<self._y.shape[0] else 0.0
    
    else:
      i = numpy.asarray(i, dtype=int)
      ret = numpy.take(self._y, i, mode='clip')
      ret[(i<0) | (i>=self._y.shape[0])] = 0.0
      return ret


  def __call__(self, x):
    """Evaluates the probability at the given x, including linear interpolation between bin centers. Vectorised."""
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
    """Returns two aligned vectors, as a conveniance function for generating the arrays to hand to a graph plotting function. Return is a tuple of (x, y), x being the bin center and y the probability."""
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
    """Returns a new Orogram that is constructed as a mixture of orograms - inputs are a list of Orogram objects and a corresponding list of weights, matching up with each Orogram. Will handle any RegOrogram objects that are included. Note that the return value can have as many bin centers as all inputs combined (duplicates are merged), so doing this iteratively without some kind of simplification step is in general unwise."""

    # Extract the list of bin centers...
    centers = numpy.concatenate([Orogram._fetchx(og) for og in orograms])
    
    # Sort and dedup...
    centers.sort(kind='mergesort')
    
    unique = numpy.ones(centers.shape, dtype=bool)
    numpy.not_equal(centers[1:], centers[:-1], out=unique[1:])
    centers = centers[unique]
    
    # Sample mixture at each center...
    values = numpy.zeros(centers.shape, dtype=numpy.float32)
    div = sum(weights)
    
    for og, w in zip(orograms, weights):
      values += (w / div) * og(centers)
    
    # Construct and return object...
    return Orogram(centers, values, norm=False, copy=False)
  
  
  @staticmethod
  def product(orograms):
    """Returns a new Orogram that is constructed as a product of ororams - input is a list of Orogram objects; it is renormalised. It can raise a ZeroDivisionError exception if the intersection of probability mass is null. Will handle any RegOrogram objects that are included. Note that the return value can have as many bin centers as all inputs combined (duplicates are merged), so doing this iteratively without some kind of simplification step is in general unwise."""
    
    # Extract the list of bin centers...
    centers = numpy.concatenate([Orogram._fetchx(og) for og in orograms])
    
    # Sort and dedup...
    centers.sort(kind='mergesort')
    
    unique = numpy.ones(centers.shape, dtype=bool)
    numpy.not_equal(centers[1:], centers[:-1], out=unique[1:])
    centers = centers[unique]
    
    # Sample mixture at each center...
    values = numpy.ones(centers.shape, dtype=numpy.float32)
    
    for og in orograms:
      values *= og(centers)
    
    # Construct and return object...
    return Orogram(centers, values, copy=False)
  

  def cdf(self, x):
    """Evaluates the cdf at the given x. Vectorised."""
    return numpy.interp(x, self._x, self._cdf, 0.0, 0.0)


  def bincdf(self, i):
    """Evaluates the cdf at the given bin center. Vectorised."""
    return numpy.take(self._cdf, i, mode='clip')


  def cdfgraph(self):
    """Returns two aligned vectors, as a conveniance function for generating the arrays to hand to a graph plotting function to get the cdf. Return is a tuple of (x, y), x being the bin center and y the cdf."""
    retx = self._x.view()
    rety = self._cdf.view()
    
    retx.flags.writeable = False
    rety.flags.writeable = False
    
    return retx, rety


  def draw(self, size = None, rng = None):
    """Draws samples from the distribution - first parameter is how many (defaults to None, which is one sample, not in an array; can be a tuple for a nD array), second something that numpy.random.default_rng() is happy to accept."""
    
    # Fetch the noise...
    rng = numpy.random.default_rng(rng)
    noise = rng.random(size, dtype=numpy.float32)
    
    # Do the inverse CDF dance...
    after = numpy.searchsorted(self._cdf, noise)
    t = (noise - self._cdf[after-1]) / (self._cdf[after] - self._cdf[after-1])
    
    return (1-t) * self._x[after-1] + t * self._x[after]


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
