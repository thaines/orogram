#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

# Generates loads of random mixtures and finds which one has the largest difference between numerical integration and analytic entropy, before reporting on it...

import sys, os

import numpy

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import dists

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Some paramters...
low = -4
high = 4
samples_orogram = 2**6
samples_ni = 2**24

print(f'Orogram samples = {samples_orogram}')



# Generate random mixture model parameters...
params = dists.mix4_params(256, 42)



# Loop each, bake to Orogram, and compare entropy, storing index of largest difference...
worst_pi = -1
worst_delta = -1.0

for pi in range(params.shape[0]):
  print(f'\r{pi} of {params.shape[0]}', end='')
  # Bake to Orogram...
  pcdf = lambda x: dists.mix4_cdf(x, params[pi,:])
  base_model = orogram.RegOrogram((high-low) / (samples_orogram - 1))
  base_model.bake(pcdf, low, high)
  model = orogram.Orogram(base_model)

  # Compare...
  delta = numpy.fabs(model.entropynumint(samples_ni) - model.entropy())
  if delta>worst_delta:
    worst_pi = pi
    worst_delta = delta

print(f'\r{params.shape[0]} of {params.shape[0]}')
print(f'worst delta = {worst_delta}')
print()
