#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Demo of transforming draws from a Gaussian distribution into something else...

from functools import partial
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from regorojax import *
from nn import *

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_debug_nans', True)



# Cost function that measures how well a nn transforms a Gaussian draw to match a target orogram (using nn as offset); plus gradient of...
@partial(jax.jit, static_argnames=('batchsize',))
def cost(layers, key, target, low, high, batchsize):
  # Sample from the Gaussian to get a minibatch to send through...
  minibatch = jax.random.normal(key, (batchsize, 1))

  # Push it through the nn and make an orogram...
  tformed = (minibatch + mlp_gauss_dm(layers, minibatch))[:,0]
  ogram = orogram(tformed, low, high, target.shape[0])

  # Calculate the kl with reference to the target distribution...
  delta = spacing(low, high, target.shape[0])
  kl = crossentropy(ogram, target, delta) - crossentropy(ogram, ogram, delta)
  return kl


grad = jax.jit(jax.grad(cost), static_argnames=('batchsize',))



# Prepare a target orogram - mixture of several basic distributions...
low = -4.0
high = 4.0
edges = jnp.linspace(low, high, 65)

#target = jnp.zeros(edges.shape[0])
target = 0.01 * jnp.ones(edges.shape[0])

## Triangle...
target += jnp.maximum(1.25 - jnp.fabs(edges + 2.5), 0.0) / (0.5 * 1.25**2)

## Uniform/square...
target += (jnp.fabs(edges) < 0.8) / (0.8**2)

## Gaussian...
target += jnp.exp(-0.5 * jnp.square((edges - 2.5) / 0.35)) / (0.35*jnp.sqrt(2 * jnp.pi))

area = 0.5*(target[:-1] + target[1:]).sum() * (high - low) / (target.shape[0]+1)
target /= area



# Visualise target alongside input distribution...
plt.figure(figsize=[8, 3])
plt.xlabel(r'$z$')
plt.ylabel(r'$P(z)$')

gauss = jnp.exp(-0.5*jnp.square(edges)) / jnp.sqrt(2*jnp.pi)

plt.plot(edges, gauss, label='input pdf')
plt.plot(edges, target, label='output pdf')

plt.legend()
plt.savefig(f'nn_remap_initial.pdf', bbox_inches='tight')



# Initialise the parameters of a small NN...
rng = jax.random.key(0)

rng, key = jax.random.split(rng)
layers = random_init_glorot([1,32,32,1], key)

rng, key = jax.random.split(rng)
random_init_last(layers, 0.01, key)



# Measure how badly it solves the problem initially...
rng, cost_key = jax.random.split(rng)
initial_cost = cost(layers, cost_key, target, low, high, 1024)
print(f'initial cost = {initial_cost:.3f}')



# Train with Adam...
stepsize = 0.005
steps = 1024*8
batchsize = 256

beta1 = 0.9
beta2 = 0.999

moment1 = {k : jnp.zeros(v.shape) for k,v in layers.items()}
moment2 = {k : jnp.zeros(v.shape) for k,v in layers.items()}

print('optimising:')
for itr in range(steps):
  print(f'\r  {itr+1} of {steps}', end='')

  # Calculate gradient...
  rng, key = jax.random.split(rng)
  dx = grad(layers, key, target, low, high, batchsize)

  # Update roling averages...
  moment1 = {k : beta1*v + (1-beta1)*dx[k] for k,v in moment1.items()}
  moment2 = {k : beta2*v + (1-beta2)*jnp.square(dx[k]) for k,v in moment2.items()}

  # Move...
  layers = {k : v - (stepsize / (jnp.sqrt(moment2[k] / (1-beta2)) + 1e-6)) * moment1[k] / (1 - beta1) for k,v in layers.items()}

print()



# Report on final result...
final_cost = cost(layers, cost_key, target, low, high, 1024)
print(f'final cost = {final_cost:.3f}')



# Visualise target alongside distribution currently being generated...
minibatch = jax.random.normal(cost_key, (1024*32, 1))
tformed = (minibatch + mlp_gauss_dm(layers, minibatch))[:,0]
print(f'transformed range = [{tformed.min():.1f}, {tformed.max():.1f}]')
ogram = orogram(tformed, low, high, target.shape[0])

plt.figure(figsize=[8, 3])
plt.xlabel(r'$z$')
plt.ylabel(r'$P(z)$')

plt.plot(edges, ogram, label='transformed pdf')
plt.plot(edges, target, label='output pdf')

plt.legend()
plt.savefig(f'nn_remap_final.pdf', bbox_inches='tight')
