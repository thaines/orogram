#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os, time
import functools

import numpy
import scipy

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Some parameters...
## Range to consider...
low = -4
high = 4



# Need some distributions, with the pdf and cdf of each; two linear and two curvy to mix it up...
def uniform_pdf(x, centre, radius):
  ret = numpy.greater(x, centre-radius).astype(float)
  ret *= numpy.less_equal(x, centre+radius).astype(float)
  ret /= 2 * radius
  return ret

def uniform_cdf(x, centre, radius):
  return numpy.clip((x - centre + radius) / (2*radius), 0.0, 1.0)


def triangular_pdf(x, centre, radius):
  ret = 1.0 - numpy.fabs(x - centre) / radius
  ret = numpy.maximum(ret, 0.0)
  return ret / radius

def triangular_cdf(x, centre, radius):
  x = numpy.clip(x, centre-radius, centre+radius)
  upper = numpy.greater(x, centre).astype(float)
  sign = 2 * (upper - 0.5)

  ret = numpy.square(centre + sign*radius - x) / (2*radius*radius)
  ret *= -sign
  ret += upper

  return ret


def gaussian_pdf(x, mean, sd):
  ret = numpy.exp(-0.5 * numpy.square((x - mean) / sd))
  ret /= sd * numpy.sqrt(2*numpy.pi)
  return ret

def gaussian_cdf(x, mean, sd):
  return 0.5 * (1 + scipy.special.erf((x - mean) / (sd * numpy.sqrt(2))))


def laplace_pdf(x, mean, scale):
  return numpy.exp(-numpy.fabs(x - mean) / scale) / (2*scale)

def laplace_cdf(x, mean, scale):
  ret = 0.5 * numpy.exp(-numpy.fabs(x - mean) / scale)

  upper = numpy.greater(x, mean).astype(float)
  ret *= -2 * (upper - 0.5)
  ret += upper

  return ret



# Verify above by plotting pdf and cdf converted into pdf, to check they match and look good...
for name, pdf, cdf, param in [('uniform', uniform_pdf, uniform_cdf, {'centre' : 0.5, 'radius' : 1.2}),
                              ('triangular', triangular_pdf, triangular_cdf, {'centre' : -0.5, 'radius' : 1.4}),
                              ('gaussian', gaussian_pdf, gaussian_cdf, {'mean' : -0.8, 'sd' : 0.3}),
                              ('laplace', laplace_pdf, laplace_cdf, {'mean' : 0.3, 'scale' : 0.7})]:
  # Generate specific instance...
  pdf = functools.partial(pdf, **param)
  cdf = functools.partial(cdf, **param)

  # Evaluate pdf...
  x = numpy.linspace(low, high, 2048)
  y = pdf(x)

  # Evaluate cdf...
  yc = cdf(x)

  # Convert cdf to pdf...
  base_model = orogram.RegOrogram(0.05)
  base_model.bake(cdf, low, high)
  model = orogram.Orogram(base_model)

  # Plot and write graph...
  plt.figure(figsize=[6, 3])
  plt.xlabel(r'$x$')
  plt.ylabel(r'$P(x)$')

  plt.plot(x, y)
  plt.plot(*model.graph())
  plt.plot(x, yc, ':')
  plt.savefig(f'quad_test_{name}.pdf', bbox_inches='tight')



# Setup a parameterised mixture models, represented with vectorised pdf and cdf functions with a parameter vector...
## Parameter vector is:
## [4xmixture weight, 4xcenters, 4xscales] = length 12
## Mixture components in order of above functions
def pdf(x, param):
  return param[0]*uniform_pdf(x, param[4], param[8]) + param[1]*triangular_pdf(x, param[5], param[9]) + param[2]*gaussian_pdf(x, param[6], param[10]) + param[3]*laplace_pdf(x, param[7], param[11])

def cdf(x, param):
  return param[0]*uniform_cdf(x, param[4], param[8]) + param[1]*triangular_cdf(x, param[5], param[9]) + param[2]*gaussian_cdf(x, param[6], param[10]) + param[3]*laplace_cdf(x, param[7], param[11])



# Visualise the pdf for one where they are clearly seperated...
param = numpy.array([0.25, 0.25, 0.25, 0.25,
                     -2.25, -0.75, 0.75, 2.25,
                     0.6, 0.6, 0.6, 0.6])

sep_pdf = lambda x: pdf(x, param)
sep_cdf = lambda x: cdf(x, param)

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
plt.plot(*model.graph())
#plt.plot(x, yc, ':')
plt.savefig(f'quad_mixture.pdf', bbox_inches='tight')



# Generate a bank of randomised mixture parameters...
rng = numpy.random.default_rng(0)
params = numpy.empty((32, 12))

params[:,0:4] = rng.dirichlet(numpy.ones(4), params.shape[0])
params[:,4:8] = rng.uniform(-2.5, 2.5, (params.shape[0], 4))
params[:,8:12] = rng.uniform(0.1, 1.0, (params.shape[0], 4))



# Array of sample counts, plus how far into the array to plot...
samples = numpy.geomspace(3, 2**24, 16)
samples = numpy.unique(samples.astype(int))
show = numpy.searchsorted(samples, 10000)



# Mega loop — for each parameter set and each sample count calculate entropy, with both numerical integration and my analytic approach, recording them all in a giant table...
entropy_ni = numpy.empty((params.shape[0], samples.shape[0]))
entropy = numpy.empty((params.shape[0], samples.shape[0]))

for pi in range(params.shape[0]):
  print(f'\r{pi} of {params.shape[0]}', end='')
  # Create cdf function with current parameters...
  pcdf = lambda x: cdf(x, params[pi,:])

  for si in range(samples.shape[0]):
    # Create orogram object with correct sample count from cdf...
    base_model = orogram.RegOrogram((high-low) / samples[si])
    base_model.bake(pcdf, low, high)
    model = orogram.Orogram(base_model)

    # Calculate entropy both ways...
    entropy_ni[pi, si] = model.entropynumint(samples[si])
    entropy[pi, si] = model.entropy()

print(f'\r{params.shape[0]} of {params.shape[0]}')



# Take the highest sample count of numerical integration as the ground truth and calculate the error throughout, for each of the given list of percentiles...
percentiles = numpy.array([50, 75, 95])

err_ni = numpy.fabs(entropy_ni - entropy_ni[:,-1,None])
pcent_ni = numpy.percentile(err_ni, percentiles, axis=0)

err = numpy.fabs(entropy - entropy[:,-1,None])
pcent = numpy.percentile(err, percentiles, axis=0)



# Plot a graph of sample count vs error rate for both...
plt.figure(figsize=[8, 4])
plt.xlabel(r'$\textrm{sample count}$')
plt.ylabel(r'$\epsilon$')
plt.xscale('log')
plt.yscale('log')

plt.plot(samples[:show], pcent_ni[0,:show], 'C0-', label=r'numerical integration $50^\textrm{th}$ percentile')
plt.plot(samples[:show], pcent_ni[1,:show], 'C0--', label=r'numerical integration $75^\textrm{th}$ percentile')
plt.plot(samples[:show], pcent_ni[2,:show], 'C0:', label=r'numerical integration $95^\textrm{th}$ percentile')

plt.plot(samples[:show], pcent[0,:show], 'C2-', label=r'analytic $50^\textrm{th}$ percentile')
plt.plot(samples[:show], pcent[1,:show], 'C2--', label=r'analytic $75^\textrm{th}$ percentile')
plt.plot(samples[:show], pcent[2,:show], 'C2:', label=r'analytic $95^\textrm{th}$ percentile')

plt.legend()
plt.savefig(f'quad_convergence.pdf', bbox_inches='tight')
