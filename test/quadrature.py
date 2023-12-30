#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os
import functools

import argparse

import numpy
import scipy

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

from dists import *

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Command line arguments...
parser = argparse.ArgumentParser(description='Compares quadrature for calculating entropy between piecewise linear analytic and numerical integration.')

parser.add_argument('-q', '--quick', action='store_true', help='Changes the default arguments for a fast set.')

parser.add_argument('-l', '--limit', default=1000, help='How high to go in samples for the graph it outputs.')
parser.add_argument('-d', '--dists', help='How many randomised mixture model to average the error over. Defaults to 1024, unless quick in which case it defaults to 32.')
parser.add_argument('-s', '--steps', help='How many steps to have on the x axis of the graph; defaults to 64 unless quick, in which case it drops down to 8.')

args = parser.parse_args()

if args.dists is None:
  args.dists = 32 if args.quick else 1024

if args.steps is None:
  args.steps = 8 if args.quick else 64



# Some parameters...
## Range to consider...
low = -4
high = 4



# Visualise the pdf for one where they are clearly seperated...
param = numpy.array([0.25, 0.25, 0.25, 0.25,
                     -2.25, -0.75, 0.75, 2.25,
                     0.6, 0.6, 0.3, 0.4])

sep_pdf = lambda x: mix4_pdf(x, param)
sep_cdf = lambda x: mix4_cdf(x, param)

x = numpy.linspace(low, high, 2048)
y = sep_pdf(x)
yc = sep_cdf(x)

base_model = orogram.RegOrogram(0.05)
base_model.bake(sep_cdf, low, high)
model = orogram.Orogram(base_model)

plt.figure(figsize=[6, 3])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.plot(x, y)
#plt.plot(*model.graph())
#plt.plot(x, yc, ':')
plt.savefig(f'quad_mixture.pdf', bbox_inches='tight')



# Generate a bank of randomised mixture parameters...
params = mix4_params(args.dists, 0)



# Array of sample counts, plus how far into the array to plot...
samples = numpy.geomspace(3, args.limit, args.steps)
samples = numpy.append(samples, [2**16 if args.quick else 2**24])
samples = numpy.unique(samples.astype(int))
show = numpy.searchsorted(samples, args.limit)



# Mega loop — for each parameter set and each sample count calculate entropy, with both numerical integration and my analytic approach, recording them all in a giant table...
entropy_ni = numpy.empty((params.shape[0], samples.shape[0]))
entropy = numpy.empty((params.shape[0], samples.shape[0]))

for pi in range(params.shape[0]):
  print(f'\r{pi} of {params.shape[0]}', end='')
  # Create cdf function with current parameters...
  pcdf = lambda x: mix4_cdf(x, params[pi,:])

  for si in range(samples.shape[0]):
    # Create orogram object with correct sample count from cdf...
    base_model = orogram.RegOrogram((high-low) / (samples[si] - 1))
    base_model.bake(pcdf, low, high)
    model = orogram.Orogram(base_model)

    # Calculate entropy both ways...
    entropy_ni[pi, si] = model.entropynumint(samples[si])
    entropy[pi, si] = model.entropy()

print(f'\r{params.shape[0]} of {params.shape[0]}')
print()



# Take the highest sample count of numerical integration as the ground truth and calculate the error throughout, for each of the given list of percentiles...
percentiles = numpy.array([50, 75, 95])

err_ni = numpy.fabs(entropy_ni - entropy_ni[:,-1,None])
pcent_ni = numpy.percentile(err_ni, percentiles, axis=0)

err = numpy.fabs(entropy - entropy_ni[:,-1,None])
pcent = numpy.percentile(err, percentiles, axis=0)



# Report difference between highest sample count of both numerical integration and analytic, as a sanity check...
zero_maybe = numpy.fabs(entropy[:,-1] - entropy_ni[:,-1])
print(f'mean converged delta = {zero_maybe.mean()}')
print(f'maximum converged delta = {zero_maybe.max()}')
print()



# Plot a graph of sample count vs error rate for both...
plt.figure(figsize=[8, 4])
plt.xlabel(r'$\textrm{sample count}$')
plt.ylabel(r'$\epsilon$')
plt.xscale('log')
plt.yscale('log')

plt.plot(samples[:show], pcent_ni[0,:show], 'C0-', label=r'$50^\textrm{th}$ percentile of numerical integration')
plt.plot(samples[:show], pcent_ni[1,:show], 'C0--', label=r'$75^\textrm{th}$ percentile of numerical integration')
plt.plot(samples[:show], pcent_ni[2,:show], 'C0:', label=r'$95^\textrm{th}$ percentile of numerical integration')

plt.plot(samples[:show], pcent[0,:show], 'C1-', label=r'$50^\textrm{th}$ percentile of analytic')
plt.plot(samples[:show], pcent[1,:show], 'C1--', label=r'$75^\textrm{th}$ percentile of analytic')
plt.plot(samples[:show], pcent[2,:show], 'C1:', label=r'$95^\textrm{th}$ percentile of analytic')

plt.legend(loc='upper right')
plt.savefig(f'quad_convergence.pdf', bbox_inches='tight')
