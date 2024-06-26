#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.



from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches

import jax
import jax.numpy as jnp

from regorojax import *

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_debug_nans', True)



# Basic parameters...
low = -3
high = 3
bins = 9
edges = jnp.linspace(low, high, bins)
delta = edges[1] - edges[0]



# Calculate and visualise an orogram as a single point sweeps across the range, to validate that code is behaving itself...
plt.figure(figsize=[6, 3])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.vlines(edges, 0.0, 0.1, colors='k')

for i, v in enumerate(jnp.linspace(-2, 2, 7)):
  og = orogram(jnp.array([v]), low, high, bins)

  plt.plot(edges, og, color=f'C{i}')
  plt.scatter([v], [0.5], color=f'C{i}')

plt.savefig('regorojax_sweep.pdf', bbox_inches='tight')



# Functions for generating a suitable family of regular orograms...
def segment(edges, slope):
  ret = jnp.maximum(0.5 + slope * edges, 0)
  ret = ret.at[:edges.shape[0]//4].set(0.0)
  ret = ret.at[(edges.shape[0]*3)//4:].set(0.0)

  area = 0.5 * (ret[:-1] + ret[1:]).sum() * (edges[-1] - edges[0]) / (edges.shape[0]-1)

  return ret / area


def clipped_stdgauss(edges):
  ret = jnp.exp(-0.5*jnp.square(edges)) / jnp.sqrt(2*jnp.pi)

  area = 0.5 * (ret[:-1] + ret[1:]).sum() * (edges[-1] - edges[0]) / (edges.shape[0]-1)

  return ret / area



# Define four variants...
orogramA = segment(edges, 0.5)
orogramB = segment(edges, 0.25)
orogramC = segment(edges, 0.0)
orogramD = clipped_stdgauss(edges)



# Visualise...
plt.figure(figsize=[6, 3])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.plot(edges, orogramA, label='A')
plt.plot(edges, orogramB, label='B')
plt.plot(edges, orogramC, label='C')
plt.plot(edges, orogramD, label='D')

plt.legend()
plt.savefig('regorojax_dists.pdf', bbox_inches='tight')



# Calculate and report entropy and cross entropy...
print(f'    H(A) = {crossentropy(orogramA, orogramA, delta):.3f}')
print(f'  H(A,B) = {crossentropy(orogramA, orogramB, delta):.3f}')
print(f'  H(A,C) = {crossentropy(orogramA, orogramC, delta):.3f}')
print(f'  H(A,D) = {crossentropy(orogramA, orogramD, delta):.3f}')
print()

print(f'  H(B,A) = {crossentropy(orogramB, orogramA, delta):.3f}')
print(f'    H(B) = {crossentropy(orogramB, orogramB, delta):.3f}')
print(f'  H(B,C) = {crossentropy(orogramB, orogramC, delta):.3f}')
print(f'  H(B,D) = {crossentropy(orogramB, orogramD, delta):.3f}')
print()

print(f'  H(C,A) = {crossentropy(orogramC, orogramA, delta):.3f}')
print(f'  H(C,B) = {crossentropy(orogramC, orogramB, delta):.3f}')
print(f'    H(C) = {crossentropy(orogramC, orogramC, delta):.3f}')
print(f'  H(C,D) = {crossentropy(orogramC, orogramD, delta):.3f}')
print()

print(f'  H(D,A) = {crossentropy(orogramD, orogramA, delta):.3f}')
print(f'  H(D,B) = {crossentropy(orogramD, orogramB, delta):.3f}')
print(f'  H(D,C) = {crossentropy(orogramD, orogramC, delta):.3f}')
print(f'    H(D) = {crossentropy(orogramD, orogramD, delta):.3f}')
print()



# Define cost and it's gradient, as required to move points towards a target distribution...
@jax.jit
def cost(x, target, low, high):
  px = orogram(x, low, high, target.shape[0])

  delta = spacing(low, high, target.shape[0])
  return crossentropy(px, target, delta) - crossentropy(px, px, delta)

grad = jax.jit(jax.grad(cost))



# Put an evenly spaced sample of points through and calculate their gradient relative to each distribution, as in the initial gradient of the transform to move them to be a sample from each target distribution...
uniform = jnp.linspace(-2.8, 2.8, 65)
height = 0.2 + 0.6 * 0.5 * (jnp.asin(jnp.sin(uniform * jnp.pi)) * 2 / jnp.pi + 1.0)

for name, target in [('A', orogramA), ('B', orogramB), ('C', orogramC), ('D', orogramD)]:
  dx = grad(uniform, target, low, high)
  y = height * target.max()

  plt.figure(figsize=[5, 2.5])
  plt.xlabel(r'$x$')
  plt.ylabel(r'$P(x)$')

  plt.plot(edges, target, color='C2', label='$Q(x)$')
  plt.plot([-3, -2.8, -2.8, 2.8, 2.8, 3.0], [0.0,0.0, 1/5.6, 1/5.6, 0.0, 0.0], label='$P(x)$', color='C1')

  for i in range(uniform.shape[0]):
    plt.annotate('', xy=(uniform[i]-dx[i],y[i]), xytext=(uniform[i],y[i]), xycoords='data', textcoords='data', arrowprops=dict(width=0.1, headwidth=2, headlength=2))

  plt.vlines(edges, 0.0, 0.03, colors='k', linewidths=0.5)

  plt.legend(loc='upper left')
  plt.savefig(f'regorojax_{name}.pdf', bbox_inches='tight')
