# Orogram

A library for working with 1D PDFs represented with piecewise linear functions.
The term _orogram_ is used to refer to any representation of a PDF with a piecewise linear function.
That may be read as _histogram_ where the _histo_ (poles, as in telephone poles) has been replaced with _oro_ (mountains, which are definitely triangular/piecewise-linear, at least when you're five).

This library was created for the paper
["The Cross-entropy of Piecewise Linear Probability Density Functions" by Tom S. F. Haines](https://openreview.net/forum?id=AoOi9Zgdsv)
(Published in Transactions on Machine Learning Research (04/2024))
It would be great to cite this paper if you make use of this code for a publication of your own.

The most recent version of this code can be found at [https://github.com/thaines/orogram](https://github.com/thaines/orogram).



## Requirements

Needs `numpy` and `cython` (at least version 3).
Whenever an array is expected it is expected to be some object compatible with a `numpy` array.



## Contents

The (installable) code provides the following:

* `RegOrogram` — An orogram with regularly spaced bins, much the same as your typical histogram. Feature rich — has everything you would expect of a 1D density estimate. Designed with data capture in mind.
* `Orogram` — An orogram with irregularly spaced bins. Reasonable set of features, but immutable and hence not for data capture. Simpler data structures so typically faster than `RegOrogram`. Otherwise does its best to match the `RegOrogram` feature set.
* `linear_crossentropy(length, p0, p1, q0, q1, fast = False)` — Direct access to the internal function that implements the key result of the cross-entropy paper. Calculates cross entropy for a linear segment with the given length where the two values for each distribution are the end point.

Beyond the provided code is

* `setup.py` — Standard `setuptools` code for making a package/installing.
* `test` - A folder of tests and demos for the library.
* `jax` — A folder of demos for the paper that needed automatic differentiation. Completely ignored by the actual installable library.



## `RegOrogram`

The _regular orogram_ object only supports an orogram with a regular spacing, but is designed for incremental data capture, i.e. data points can be added and it will expand its range as required.

This object provides the following stable methods/capabilities (other methods may be available, but are likely to change):

* `RegOrogram(spacing = 0.001, blocksize = 1024 * 16)` — Constructor, with the first parameter the spacing between bin centres. The blocksize is a trade off between memory/speed and has no effect on results; it's how much extra memory to allocate each time a data point lands outside the current range.
* `.copy()` — Copy constructor.
* `sizeof(self)` — Returns how many bytes of memory the object is currently using.
* `len(self)` — How many bins it has, including a zero mass bin either side.
* `.spacing()` — Spacing between bin centers it was constructed with.
* `.blocksize()` — Blocksize object was constructed with.
* `.add(x, weight = 1.0, smooth=False)` — Adds a single value or a 1D array of data points. Can optionally provide `weight` as something that broadcasts to each data point. If `smooth` is `False`, the default it does the maximum likelihood solution; if `True` it linearly interpolates between bin centers.
* `.binadd(base, density, total = None)` — Lets you add to bin centers directly. Base is the index of the bin center that maps to `density[0]` (0 gets you the bin center at 0). `total` is for if you know the sum of density, to save computation.
* `.bake_cdf(cdf, start, end, weight = 1, vectorised = True)` — Given a function for evaluating the CDF given `x` adds that distribution into the model, for the range `start` to `end`. `weight` is the total mass that the CDF should count as while `vectorised` indicates if the CDF function is vectorised or not.
* `.sum()` — Returns the total weight that the model has, i.e. how many samples have been received in typical usage.
* `.min()` — Minimum `x` where it hits zero.
* `.max()` — Maximum `x` where it hits zero.
* `.binmin()` — Index of minimum bin that is zero but adjacent to a nonzero bin.
* `.binmax()` — Index of maximum bin that is zero but adjacent to a nonzero bin.
* `.center(i)` — Given the index of a bin returns the x value of its center. Vectorised.
* `.weight(i)` — Given the index of a bin returns how much weight has been assigned to it (fractional sample count). Vectorised.
* `.prob(i)` — Given the index of a bin returns the probability at its bin center. Vectorised.
* `self(x)` — Evaluates the probability at the given x. Vectorised.
* `.modes()` — Returns an array containing every mode.
* `.highest()` — Returns the x with the highest probability.
* `.binhighest()` — Returns the index of the bin with the highest probability.
* `.graph(start = None, end = None)` — Convenience method that returns a tuple of `(x, y)` suitable for passing to a graphing tool. Can optionally provide a range.
* `self += other` — Adds the samples of another `RegOrogram` to this one. Much faster if the spacing matches but will interpolate if they do not. Matching blocksize also helps.
* `self *= scalar` — Lets you scale all of the weights (number of samples).
* `.cdf(x)` — Evaluates the CDF at any location. Vectorised.
* `.bincdf(i)` — Evaluates the CDF at an indexed bin's center. Vectorised.
* `.cdfgraph(start = None, end = None)` — Convenience method that returns a tuple of `(x, y)` suitable for passing to a graphing tool, this time outputting the CDF. Can optionally provide a range.
* `.invcdf(uni)` — Evaluates the inverse CDF for a value in `[0, 1)`. Vectorised.
* `.draw(size = None, rng = None)` — Samples from the represented PDF. Can optionally provide a `size` (integer for 1D, or tuple for a nD array) and/or a `rng`, as an object that `numpy.random.default_rng(rng)` knows what to do with.
* `.median()` — Returns the median.
* `.var()` — Returns the variance.
* `.meanvar()` — Returns a tuple of `(mean, variance)`.
* `.entropy()` — Calculates the entropy, in nats.
* `p.crossentropy(q)` — Calculates the cross entropy, `H(p,q)`, in nats.
* `p.kl(q)` — Calculates the Kullback–Leibler divergence, `KL(p||q)`, in nats.
