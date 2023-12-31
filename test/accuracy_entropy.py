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



# Some parameters...
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
worst_model = None

total_delta = 0.0

for pi in range(params.shape[0]):
  print(f'\r{pi} of {params.shape[0]}', end='')
  # Bake to Orogram...
  pcdf = lambda x: dists.mix4_cdf(x, params[pi,:])
  base_model = orogram.RegOrogram((high-low) / (samples_orogram - 1))
  base_model.bake(pcdf, low, high)
  model = orogram.Orogram(base_model)

  # Compare...
  delta = numpy.fabs(model.entropynumint(samples_ni) - model.entropy())
  total_delta += delta
  if delta>worst_delta:
    worst_pi = pi
    worst_delta = delta
    worst_model = model

print(f'\r{params.shape[0]} of {params.shape[0]}')
print(f'worst delta = {worst_delta}')
print(f'mean delta = {total_delta/params.shape[0]}')
print()



# Visualise the worst...
plt.figure(figsize=[8, 4])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.plot(*worst_model.graph())

plt.savefig(f'accuracy_entropy_worst.pdf', bbox_inches='tight')



# Calculate error for each linear segment (assume zero for no mass)...
errx = [worst_model.min()]
erry = []
for base in range(len(worst_model)-1):
  try:
    chunk = worst_model.binclip(base, base+1)
    delta = numpy.fabs(chunk.entropynumint(samples_ni // samples_orogram) - chunk.entropy())

    errx.append(chunk.max())
    erry.append(delta)

  except ZeroDivisionError:
    errx.append(worst_model.center(base+1))
    erry.append(0.0)

print(f'Maximum segment error = {max(erry)}')



# Plot segment error...
plt.figure(figsize=[8, 4])
plt.xlabel(r'$x$')
plt.ylabel(r'$\epsilon$')

plt.hist(errx[:-1], errx, weights=erry)

plt.savefig(f'accuracy_entropy_error.pdf', bbox_inches='tight')
