#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from regorojax import *

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_debug_nans', True)



# Generate some points from a distribution...
rng = jax.random.key(0)

rng, key = jax.random.split(rng)
x = jax.random.normal(key, (1024,))



# Setup and create an initial orogram...
low = -3.5
high = 3.5
bins = int(jnp.sqrt(2*x.shape[0]))
delta = spacing(low, high, bins)
print(f'low = {low:.3f}, high = {high:.3f}, bins = {bins}, delta = {delta:.3f}')

edges = jnp.linspace(low, high, bins)
px = orogram(x, low, high, bins)



# Generate a target distribution...
target = jnp.fabs(jnp.sin(jnp.clip(edges * jnp.pi / 3, -jnp.pi, jnp.pi)))
area = 0.5*(target[:-1] + target[1:]).sum() * (high - low) / (target.shape[0]+1)
target /= area



# Define cost and it's gradient, as required to move points towards target distribution...
@jax.jit
def cost(x, target, low, high):
  px = orogram(x, low, high, target.shape[0])

  delta = spacing(low, high, target.shape[0])
  return crossentropy(px, target, delta) - crossentropy(px, px, delta)

grad = jax.jit(jax.grad(cost))

initial = cost(x, target, low, high)
print(f'initial kl = {initial:.3f}')



# Visualise the initial state, target and gradient...
initial_grad = grad(x, target, low, high)
print('grad:')
print(f'  min = {initial_grad.min():.3f}')
print(f'  max = {initial_grad.max():.3f}')

for bi in jnp.where(jnp.logical_not(jnp.isfinite(initial_grad)))[0][:3]:
  print(f'  bad at {bi} = {initial_grad[bi]}')
  print(f'    x = {x[bi]:.3f}')

plt.figure(figsize=[5, 2.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.plot(edges, px, label='Start', color='C1')
plt.plot(edges, target, label='Goal', color='C2')

rng, key = jax.random.split(rng)
yrand = target.max() * (0.15 + 0.7*jax.random.uniform(key, (64,)))
for i in range(yrand.shape[0]):
  plt.annotate('', xy=(x[i]-100*initial_grad[i],yrand[i]), xytext=(x[i],yrand[i]), xycoords='data', textcoords='data', arrowprops=dict(width=0.1, headwidth=2, headlength=2))
  #plt.arrow(x[i], yrand[i], -initial_grad[i], 0.0, width=0.0005)

plt.legend()
plt.savefig(f'conform_initial.pdf', bbox_inches='tight')



# Use Nesterov to move points to match target distribution...
stepsize = 0.1
momentum = 0.75
steps = 1024*2

ngrad = jnp.zeros(x.shape)

print('optimising:')
for itr in range(steps):
  print(f'\r  {itr+1} of {steps}', end='')

  dx = grad(x - momentum*ngrad, target, low, high)
  ngrad = momentum*ngrad + stepsize*dx
  x -= ngrad

print()



# Report and graph...
final = cost(x, target, low, high)
end_grad = grad(x, target, low, high)
px = orogram(x, low, high, bins)

print(f'final kl = {final:.3f}')

plt.figure(figsize=[5, 2.5])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.plot(edges, px, label='End', color='C0')
plt.plot(edges, target, label='Goal', color='C2')

for i in range(yrand.shape[0]):
  plt.annotate('', xy=(x[i]-100*end_grad[i],yrand[i]), xytext=(x[i],yrand[i]), xycoords='data', textcoords='data', arrowprops=dict(width=0.1, headwidth=2, headlength=2))
  #plt.arrow(x[i], yrand[i], -end_grad[i], 0.0, width=0.0005)

plt.legend()
plt.savefig(f'conform_end.pdf', bbox_inches='tight')
