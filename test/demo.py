#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os, time

import numpy
import matplotlib.pyplot as plt

from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orogram



# Generates the numerical validation results used in the linear cross entropy paper.



# First lets do entropy for a well known distribution...
print('Standard Gaussian:')

rv = norm(loc=0.0, scale=1.0)
base_model = orogram.RegOrogram(0.05)
base_model.bake(rv.cdf, -4.0, 4.0)

model = orogram.Orogram(base_model)


plt.figure(figsize=[6, 3])
plt.plot(*model.graph())
plt.savefig(f'demo_standard_gaussian.pdf')


print(f'  entropy:')
print(f'    true = {0.5*numpy.log(2*numpy.pi) + 0.5:.6f}')
print(f'    analytic = {model.entropy():.6f}')
print(f'    numerical integration = {model.entropynumint(samples=2**24):.6f}')
print(f'    monte-carlo = {model.entropymc(samples=2**24):.6f}')
print()



# Now KL-divergence as two standard width Gaussians pass each other on the street...
if not os.path.exists('demo_gaussian_street.pdf'):
  print('Gaussian street…')
  mean_p = numpy.linspace(-0.5, 0.5, 256+1)
  mean_q = -mean_p


  kl = numpy.zeros(mean_p.shape[0])
  kl_true = numpy.zeros(mean_p.shape[0])
  kl_ni = numpy.zeros(mean_p.shape[0])
  kl_mc = numpy.zeros(mean_p.shape[0])


  for i in range(kl.shape[0]):
    rv_p = norm(loc=mean_p[i], scale=1.0)
    base_model_p = orogram.RegOrogram(0.05)
    base_model_p.bake(rv_p.cdf, -4.0, 4.0)
    model_p = orogram.Orogram(base_model_p)

    rv_q = norm(loc=mean_q[i], scale=1.0)
    base_model_q = orogram.RegOrogram(0.05)
    base_model_q.bake(rv_q.cdf, -4.0, 4.0)
    model_q = orogram.Orogram(base_model_q)

    kl[i] = model_p.crossentropy(model_q) - model_p.entropy()
    kl_true[i] = numpy.log(1.0/1.0) + (numpy.square(1.0) + numpy.square(mean_p[i] - mean_q[i]))/(2*numpy.square(1.0)) - 0.5
    kl_ni[i] = model_p.crossentropynumint(model_q) - model_p.entropynumint()
    kl_mc[i] = model_p.crossentropymc(model_q) - model_p.entropymc()


  plt.figure(figsize=[6, 3])
  plt.xlabel(r'$\mu_Q - \mu_P$')
  plt.ylabel('KL-divergence')

  plt.plot(mean_q - mean_p, kl_mc, linewidth=6, label='Monte-Carlo integration')
  plt.plot(mean_q - mean_p, kl_ni, linewidth=4.5, label='Numerical integration')
  plt.plot(mean_q - mean_p, kl_true, linewidth=3, label='True')
  plt.plot(mean_q - mean_p, kl, linewidth=1.5, label='Analytic')

  plt.legend()
  plt.savefig('demo_gaussian_street.pdf', bbox_inches='tight')
  print()



# Cross entropy of a square and triangle switching positions, with a Gaussian in the background to avoid infinities — choosen to capture a key advantage of piecewise linear representations when it comes to stepped edges...
print('Square-triangle dance…')
slab = orogram.RegOrogram(0.02)
slab.bake(norm(loc=0.0, scale=1.0).cdf, -2.0, 2.0)

def generate(t):
  """Generates a mixture of three distributions — a slab, a uniform and a triangle, with t going from 0 to 1 and defining where the uniform and triangle go."""
  uniform_mid = (1-t) * -0.6 + t * 0.6
  tri_mid = (1-t) * 0.6 + t * -0.6

  uniform = orogram.Orogram([uniform_mid-0.5-1e-3, uniform_mid-0.5, uniform_mid+0.5, uniform_mid+0.5+1e-3], [0.0, 1.0, 1.0, 0.0])
  tri = orogram.Orogram([tri_mid-0.5, tri_mid, tri_mid+0.5], [0.0, 1.0, 0.0])

  return orogram.Orogram.mixture([slab, uniform, tri], [0.1, 1.0, 1.0])


model_p = generate(0)
plt.figure(figsize=[6, 3])
plt.plot(*model_p.graph())
plt.savefig(f'demo_square_tri_example.pdf', bbox_inches='tight')


ts = numpy.linspace(0.0, 1.0, 256+1)
xentropy = numpy.zeros(ts.shape[0])
xentropy_ni = numpy.zeros(ts.shape[0])
xentropy_mc = numpy.zeros(ts.shape[0])


for i in range(ts.shape[0]):
  model_q = generate(ts[i])

  xentropy[i] = model_p.crossentropy(model_q)
  xentropy_ni[i] = model_p.crossentropynumint(model_q)
  xentropy_mc[i] = model_p.crossentropymc(model_q)


plt.figure(figsize=[6, 3])
plt.xlabel(r'$t$')
plt.ylabel('Cross entropy')

plt.plot(ts, xentropy_mc, linewidth=4.5, label='Monte-Carlo integration')
plt.plot(ts, xentropy_ni, linewidth=3, label='Numerical integration')
plt.plot(ts, xentropy, linewidth=1.5, label='Analytic')

plt.legend()
plt.savefig('demo_square_tri_sweep.pdf', bbox_inches='tight')
print()