#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os

import numpy

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import dists

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Setup a mixture model...
param = numpy.array([0.25, 0.25, 0.25, 0.25,
                     -2.25, -0.75, 0.75, 2.25,
                     0.6, 0.6, 0.3, 0.4])

pdf = lambda x: dists.mix4_pdf(x, param)
cdf = lambda x: dists.mix4_cdf(x, param)

x = numpy.linspace(-4, 4, 1024*8)
y = pdf(x)



# Bake...
model = orogram.Orogram.bake(cdf, -4, 4, 256)



# Report on success/not...
even, err = model.even(incerr=True)
if even:
  print('Converged')
else:
  print(f'Failed; error of {err}')
print()



# Visualise...
plt.figure(figsize=[6, 3])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.plot(x, y, linewidth=2)
plt.plot(*model.graph())

plt.savefig(f'bake_irregular.pdf', bbox_inches='tight')
