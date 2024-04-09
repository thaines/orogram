# Orogram

A library for working with 1D PDFs represented with piecewise linear functions.
The term _orogram_ is used to refer to any representation of a PDF with a piecewise linear function.
That may be read as _histogram_ where the _histo_ (poles, as in telephone poles) has been replaced with _oro_ (mountains, which are definitely triangular/piecewise-linear, at least when you're five).

This library was created for the paper
["The Cross-entropy of Piecewise Linear Probability Density Functions" by Tom S. F. Haines](https://openreview.net/forum?id=AoOi9Zgdsv)
(Published in Transactions on Machine Learning Research (04/2024))
It would be great to cite this paper if you make use of this code for a publication of your own.

The most recent version of this code can be found at [https://github.com/thaines/orogram](https://github.com/thaines/orogram).



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
* `sizeof(pdf)` — Returns how many bytes of memory the object is currently using.
