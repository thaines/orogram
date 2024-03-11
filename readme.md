# Orogram

A library for working with 1D PDFs represented with piecewise linear functions.
The term _orogram_ is used to refer to any representation of a PDF with a piecewise linear function.
That may be read as _histogram_ where the _histo_ (poles, as in telephone poles) has been replaced with _oro_ (mountains, which are definitely triangular/piecewise-linear, at least when you're five).

This library was created for the paper
["The Cross-entropy of Piecewise Linear Probability Density Functions" by Tom S. F. Haines](https://openreview.net/forum?id=AoOi9Zgdsv)
(currently unpublished)



## Contents

* `RegOrogram` — An orogram with regularly spaced bins, much the same as your typical histogram. Feature rich — has everything you would expect of a 1D density estimate. Designed with data capture in mind.
* `Orogram` — An orogram with irregularly spaced bins. Reasonable set of features, but immutable and hence not for data capture. Simpler data structures so typically faster than `RegOrogram`. Otherwise does its best to match the `RegOrogram` feature set.

There are demoes of how to use the libary in the `test` folder.

