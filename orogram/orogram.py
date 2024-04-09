# Copyright 2022 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

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
from . import credible
from .bake import fitgapmass

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
  """An orogram with irregularly spaced bins - an arbitrary piecewise linear PDF in other words. There is a direct equivalence to a histogram with uneven bin sizes, in terms of the maximum liklihood solutions of both having the same weights (but note it is not a frequency polygon, because the bin centers of an Orogram are not at the centers of the equivalent uneven histogram bins). The distribution is fixed on construction - this is immutable. Has a fairly rich interface."""
  __slots__ = ('_x', '_y', '_cdf')
  
  
  def __init__(self, x, y = None, cdf = None, norm=True, copy=True):
    """You construct it with two 1D arrays, containing x and y that define the values that make up the distribution. x must be increasing, all y's must be positive. By default it will renormalise if the area under the line is not one, but set norm=False if you can guarantee that is already the case. Note that it internally keeps a cdf, aligned with x/y - can provide if already calculated. Can alternatively provide a RegOrogram which it will automatically convert. The copy parameter defaults to True, but if you are giving it arrays that you won't be using again you can set it to False to avoid wasteful memory allocations."""
    
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
      if total < 1e-12:
        raise ZeroDivisionError('No mass in Orogram')

      self._y /= total

      if cdf is None:
        mass /= total
    
    # Lil' safety...
    assert(len(self._x.shape)==1)
    assert(len(self._y.shape)==1)
    assert(self._x.shape[0]==self._y.shape[0])
    assert(numpy.all(numpy.diff(self._x) >= 0.0))
    
    # Fill in the CDF array...
    if cdf is not None:
      self._cdf = numpy.array(cdf, dtype=numpy.float32, copy=copy, order='A')
    
    else:
      self._cdf = numpy.empty(self._x.shape, dtype=numpy.float32)
      self._cdf[0] = 0.0
      self._cdf[1:] = numpy.cumsum(mass)
      self._cdf[-1] = 1.0


  def even(self, epsilon=1e-2, incerr = False):
    """Judges if there is an even amount of mass inbetween each of the bin centers, i.e. the bins are equally seperated in the space of the distributions CDF. Counts number of gaps (bins-1), divides one by that number and returns True if the mass in every case is within epsilon multiplied by that number, i.e. the default is within 1%. If incerr is true then it returns a tuple of whether it's even and the maximum delta observed, as scaled by the inverse goal density so it's comparable to epsilon."""
    goal = 1 / (len(self) - 1)

    amount = 0.5 * (self._y[:-1] + self._y[1:]) * (self._x[1:] - self._x[:-1])
    delta = numpy.fabs(amount - goal) / goal

    if incerr:
      return (delta < epsilon).all(), delta.max()
    else:
      return (delta < epsilon).all()


  @staticmethod
  def bake_cdf(cdf, start, end, resolution = 1024, epsilon = 1e-2, init = 8, maxiter = 128, vectorised = True):
    """Alternative constructor (static method, returns an Orogram) that bakes a cdf into an Orogram directly. You have to provide a function for evaluating the CDF of that distribution, plus the range to evaluate (it ensures the mass in the range sums to 1). By default it assumes that the CDF functon is vectorised but you can indicate if it is not (vectorised parameter); that will be slow however. This differs from the regular orogram object in that it dynamically distributes bins, i.e. each bin has the same mass in it, to within the given epsilon parameter multiplied by the expected mass within the bin (it defaults to being within 1%). This is done by first initalising the bin centers using a regular orogram, with the given init paramter being a multiplier for resolution to boost the resolution and get a more accurate initialisation. It then does a biased binary search to refine the bin positions until within the tolerance before constructing the orogram. Note that the even() method tests if an orogram is even; this method may not converge to within the given tolerance so it can be used to verify if it has. maxiter is the maximum number of binary search steps to do. Note that if you want an even sampling in x rather than CDF(x) you should just bake a regular orogram then convert it, and that simplifying that will be better if the distribution isn't that smooth."""

    # Define the goal, i.e. where the split points should be in terms of the cdf...
    splits = numpy.linspace(0.0, 1.0, resolution)

    # Create inital Orogram, with a resolution boost...
    first = RegOrogram((end-start) / (init * (resolution - 1)))
    first.bake_cdf(cdf, start, end, vectorised)
    
    # Convert splits to bin centers...
    centers = first.invcdf(splits).astype(numpy.float32)
    centers[0] = start
    centers[-1] = end
    
    # Some cleanup...
    del first
    
    # Refine bin centers via a biased binary search...
    low = cdf(centers[0])
    high = cdf(centers[-1])
    targ = numpy.linspace(low, high, resolution)
    
    for _ in range(maxiter):
      # Evaluate the CDF at each bin center...
      if vectorised:
        cdf_eval = cdf(centers)

      else:
        cdf_eval = numpy.empty(centers.shape, dtype=numpy.float32)
        for i in range(cdf_eval.shape[0]):
          cdf_eval[i] = cdf(centers[i])

      # Check for convergence...
      if numpy.fabs(cdf_eval - targ).max() < (epsilon / resolution):
        break

      # Identify the set that need to go down and the set that need to go up; the ends are excluded from both sets as they shouldn't move...
      down = targ<cdf_eval
      up = cdf_eval<targ

      down[0] = False
      down[-1] = False
      up[0] = False
      up[-1] = False

      # Do linear interpolation, both up and down, using the above masks to update the correct set each time...
      godown = (centers[:-1] - centers[1:]) * (cdf_eval[1:] - targ[1:]) / (cdf_eval[:-1] - cdf_eval[1:])
      goup = (centers[1:] - centers[:-1]) * (targ[:-1] - cdf_eval[:-1]) / (cdf_eval[1:] - cdf_eval[:-1])

      centers[down] -= godown[down[1:]]
      centers[up] += goup[up[:-1]]

    # Fit probabilities to the final set of bin centers...
    ## Evaluate the CDF at each bin center...
    if vectorised:
      cdf_eval = cdf(centers)

    else:
      cdf_eval = numpy.empty(centers.shape, dtype=numpy.float32)
      for i in range(cdf_eval.shape[0]):
        cdf_eval[i] = cdf(centers[i])
    
    ## Renormalise, for safety...
    cdf_eval -= cdf_eval[0]
    cdf_eval /= cdf_eval[-1]
    
    ## Calculate the target density for each gap...
    gapmass = (cdf_eval[1:] - cdf_eval[:-1]).astype(numpy.float32)
      
    ## Convert to a probability. This works by defining contribution factors, that define what percentage of the density contribution each probability is, then alternating updating those and updating the probabilities until convergence (the contribution factors enforce normalisation)...
    prob = numpy.empty(centers.shape, dtype=numpy.float32)
    fitgapmass(centers, gapmass, prob)

    # Create and return fitted model...
    ret = Orogram(centers, prob, norm=False, copy=False)
    return ret


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
  
  
  def histogram(self):
    """An Orogram can be interpreted as a histogram with variable bin widths, though note that the objective of simplify() is not this histogram, so doing that then using this is not really correct. But it still might be conveniant/useful hence providing an interface to grab it as it's not trivial to calculate (edges are not half way between bin centers unless it's regular; this is not a frequency polygon). If you accept the dodgyness and are using this as a hack to get an irregular histogram you propbably want to switch off the snapping to zero at the start/end when simplifying (continuous=False). Return is (edges, mass), where edges is a list of all bin edges and mass is the total probability mass between each pair of edges (not height!). The edges array is one longer than the mass array."""
    
    # Calculate the edges in the central region then extend...
    xm1 = numpy.pad(self._x[:-2], (1,0), 'edge')
    xp2 = numpy.pad(self._x[2:], (0,1), 'edge')
    
    t = xp2 - self._x[:-1]
    t /= xp2 + self._x[1:] - self._x[:-1] - xm1
    
    edges = (1-t)*self._x[:-1] + t*self._x[1:]
    edges = numpy.concatenate(([self._x[0]], edges, [self._x[-1]]), axis=0)
    
    # Mass simply requires scaling by the size of each triangle...
    # (the half is dropped because we normalise directly - safer for numerical precision)
    mass = numpy.pad(self._x[1:], (0, 1), 'edge')
    mass -= numpy.pad(self._x[:-1], (1, 0), 'edge')
    
    mass *= self._y
    mass /= mass.sum()
    
    # Return...
    return edges, mass
  
  
  @staticmethod
  def _fetchx(orogram):
    if isinstance(orogram, RegOrogram):
      return orogram.center(numpy.arange(orogram._low-1, orogram._high+2))
    
    else:
      return orogram._x
  
  
  @staticmethod
  def mixture(orograms, weights):
    """Returns a new Orogram that is constructed as a mixture of orograms - inputs are a list of Orogram objects and a corresponding list of weights (will be normalised), matching up with each Orogram. Will handle any RegOrogram objects that are included. Note that the return value can have as many bin centres as all inputs combined (duplicates are merged), so doing this iteratively without some kind of simplification step is in general unwise."""

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
    """Returns a new Orogram that is constructed as a product of orograms - input is a list of Orogram objects; it is renormalised. It can raise a ZeroDivisionError exception if the intersection of probability mass is null. Will handle any RegOrogram objects that are included. Note that the return value can have as many bin centres as all inputs combined (duplicates are merged), so doing this iteratively without some kind of simplification step is in general unwise."""
    
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
  

  def clip(self, low, high):
    """Returns a new Orogram clipped between the two given x values. Inclusive and renormalises."""

    low_y = numpy.interp(low, self._x, self._y, 0.0, 0.0)
    low_bin = numpy.searchsorted(self._x, low, 'left')
    high_y = numpy.interp(high, self._x, self._y, 0.0, 0.0)
    high_bin = numpy.searchsorted(self._x, high, 'right')

    new_x = numpy.concatenate(([low], self._x[low_bin:high_bin], [high]))
    new_y = numpy.concatenate(([low_y], self._y[low_bin:high_bin], [high_y]))

    return Orogram(new_x, new_y)


  def binclip(self, low, high):
    """Same as clip, but you give it bin indices instead. Inclusive and renormalises."""
    return Orogram(self._x[low:high+1], self._y[low:high+1])


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
    """Evaluates the inverse cdf for the given value in [0, 1]; if the value was drawn from a uniform distribution then this is identical to draw(). Vectorised."""
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
    
    # Individual triangular bins means aren't the centre values...
    m = self._x.copy()
    m[0] += self._x[0]
    m[1:] += self._x[:-1]
    m[-1] += self._x[-1]
    m[:-1] += self._x[1:]
    m *= 1 / 3
    
    # The weights are not the probabilities - need to scale by triangle area to get mass...
    w = numpy.empty(self._y.shape[0], dtype=numpy.float32)
    
    w[:-1] = self._x[1:]
    w[-1] = self._x[-1]
    w[0] -= self._x[0]
    w[1:] -= self._x[:-1]
    
    w *= self._y
    w /= w.sum()
    
    # Can now calculate the mean with the corrected means...
    return numpy.average(m, weights=w)


  def var(self):
    """Returns the variance."""
    return self.meanvar()[1]


  def meanvar(self):
    """Returns a tuple containing (mean, variance)."""
    
    # Calculate correct means for each bin...
    m = self._x.copy()
    m[0] += self._x[0]
    m[1:] += self._x[:-1]
    m[-1] += self._x[-1]
    m[:-1] += self._x[1:]
    m /= 3
    
    # Calculate correct weights for each bin...
    w = numpy.empty(self._y.shape[0], dtype=numpy.float32)
    
    w[:-1] = self._x[1:]
    w[-1] = self._x[-1]
    w[0] -= self._x[0]
    w[1:] -= self._x[:-1]
    
    w *= self._y
    w /= w.sum()
    
    # Calculate the variance for each bin...
    v = numpy.square(self._x)
    v[0] += numpy.square(self._x[0])
    v[1:] += numpy.square(self._x[:-1])
    v[-1] += numpy.square(self._x[-1])
    v[:-1] += numpy.square(self._x[1:])
    
    v[0] -= numpy.square(self._x[0])
    v[1:] -= self._x[:-1] * self._x[1:]
    
    v[0] -= self._x[0] * self._x[1]
    v[1:-1] -= self._x[:-2] * self._x[2:]
    v[-1] -= self._x[-2] * self._x[-1]
    
    v[:-1] -= self._x[:-1] * self._x[1:]
    v[-1] -= numpy.square(self._x[-1])
    
    v /= 18
    
    # Bring it all together...
    mean = numpy.average(m, weights=w)
    var = (w * (v + numpy.square(m))).sum() - numpy.square(mean)
    
    return mean, var
  
  
  def above(self, threshold, ranges = False):
    """Returns the amount of probability mass found in the region where the pdf evaluates as above the given threshold. If you set parameter ranges=True then it returns a tuple of (total mass above threshold, list of ranges, each as (start, end, mass in range)). Primarily an internal method, used for finding credible regions, but might prove useful for someone."""
    if ranges:
      mass = credible.above(self._x, self._y, threshold)
      r = credible.ranges(self._x, self._y, threshold)
      return mass, r
    
    else:
      return credible.above(self._x, self._y, threshold)
  
  
  def credible(self, amount = 0.95, tolerance=1e-6):
    """Calculates the credible interval of the distribution, in the sense of picking the ranges that add up to the given amount of probability mass while taking up the least amount of space. This means it selects the highest probability regions first. Returns a list of regions, where each region is represented by (start, end, mass). start to end is the actual range of the region, while mass is the total probability mass found within it."""
    threshold = credible.credible(self._x, self._y, amount, tolerance)
    return credible.ranges(self._x, self._y, threshold)


  def entropy(self):
    """Calculates the entropy analytically, in nats."""
    return xentropy.irr_entropy(self._x, self._y)
  
  
  def entropynumint(self, samples=1024*1024):
    """Numerical integration version of entropy(); for testing only as obviously much slower. In nats."""
    
    # Evaluate across range of distribution...
    x = numpy.linspace(self._x[0], self._x[-1], samples)
    y = self(x)
    
    # Safe log...
    log_y = numpy.log(numpy.maximum(y, 1e-64))

    # Average height, with tweaking end bins to half weight...
    heights = y * log_y
    heights[0] += heights[-1]
    heights[0] *= 0.5
    height = xentropy.mean(heights[:-1])

    # Scale by width and return...
    return -height * (x[-1] - x[0])
  
  
  def entropymc(self, samples=1024*1024, rng=None):
    """Monte-Carlo integration version of entropy. Super slow of course so just for testing as it also doubles as a good sanity check of sampling. In nats."""
    
    # Draw and evaluate...
    x = self.draw(samples, rng)
    y = self(x)
    
    # Entropy!..
    return -numpy.log(numpy.maximum(y, 1e-64)).mean()


  def crossentropy(self, q):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding."""
    return xentropy.irregular_crossentropy(self._x, self._y, q._x, q._y)


  def crossentropynumint(self, q, samples=1024*1024):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding. This version uses numerical integration and exists for testing only - slow."""
    
    # Evaluate p across range of self distribution...
    x = numpy.linspace(self._x[0], self._x[-1], samples)
    p = self(x)
    
    # Safe log...
    log_q = numpy.log(numpy.maximum(q(x), 1e-64))

    # Average height, with tweaking end bins to half weight...
    heights = p * log_q
    heights[0] += heights[-1]
    heights[0] *= 0.5
    height = xentropy.mean(heights[:-1])

    # Scale by width and return...
    return -height * (x[-1] - x[0])


  def crossentropymc(self, q, samples=1024*1024, rng=None):
    """Calculates the cross entropy, H(p=self, q=first parameter), outputting nats. If you're measuring the inefficiency of an encoding then p/self is the true distribution and q/first parameter the distribution used for encoding. This version uses Monte-Carlo integration and exists for testing only - super slow."""
    
    # Draw and evaluate...
    x = self.draw(samples, rng)
    y = q(x)
    
    # Calculate cross-entropy...
    return -numpy.log(numpy.maximum(y, 1e-64)).mean()
  
  
  def kl(self, q):
    """Calculates the Kullback—Leibler divergence between two distributions, i.e. the expected extra nats needed for encoding data with the distribution represented by this object when the encoder is optimised for the distribution of q, the first parameter to this method. Often denoted D_{KL}(self || q) Convenience method that uses the cross-entropy and entropy methods, i.e. D_{KL}(self || q) = H(self, q) - H(self)"""
    return self.crossentropy(q) - self.entropy()
  
  
  def simplify(self, samples, perbin = 16, continuous = True):
    """Simplifies this orogram, returning a named tuple containing a new, simpler, orogram. Parameters are the number of samples that were used to generate this orogram (not recorded in Orogram object, hence having to pass in) and the expected number of data points per bin centre, which sets the prior. You can also set continuous to False if you don't mind the potential discontinuity of the output Orogram starting/ending with a value that is not zero. Returns a named tuple containing (solution - simplified Orogram, cost - it's cost (negative log probability, includes probability ratios for prior so not true cost), kept - boolean array aligned with input containing True when a point was included in the simplified output, priorcost - the cost of the prior ratio term only (subtract from cost to get cost of data term only), priorall - the cost of the prior ratio on the input, i.e. with all bins kept; add to the input entropy scaled by the sample count to get a comparable cost for the input). Depending on algorithm it may have further parameters."""
    retx, retp, retk, cost, priorcost, priorall, dominant = simplify.dp(self._x, self._y, samples, perbin, continuous)
    ret = Orogram(retx, retp, norm=False, copy=False)
    return SimplifyResult(solution=ret, cost=cost, kept=retk, priorcost=priorcost, priorall=priorall, dominant=dominant)
