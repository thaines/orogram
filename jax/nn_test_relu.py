#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from functools import partial
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from nn import *

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_debug_nans', True)



# Super simple data set — 2D, Gaussian distributed, with binary classification of inside/outside a circle...
rng = jax.random.key(0)

rng, key = jax.random.split(rng)
x = jax.random.normal(key, (1024,2))
y = (jnp.sqrt(jnp.square(x).sum(axis=1)) < 1.2).astype(float)
print(f'percentage in class 1 = {100*y.sum()/y.shape[0]:.1f}%')



# Simple rmse objective, even though it's classification...
def cost(layers, x, ytrue):
  yguess = mlp_relu_dm(layers, x)[:,0]
  return jnp.sqrt(jnp.square(ytrue - yguess).sum())

grad = jax.jit(jax.grad(cost))



# Also want ability to calculate error rate...
def error(layers, x, ytrue):
  yguess = (mlp_relu_dm(layers, x)[:,0] > 0.5).astype(float)
  rate = jnp.fabs(yguess - ytrue).sum() /  ytrue.shape[0]
  return rate



# Initalise model and confirm it doesn't work...
rng, key = jax.random.split(rng)
layers = random_init_he([2,32,32,32,1], key)

initial_cost = cost(layers, x, y)
initial_error = error(layers, x, y)

print(f'initial cost = {initial_cost:.3f}')
print(f'initial error rate = {initial_error*100:.1f}%')



# Train with Nesterov...
stepsize = 0.01
momentum = 0.75
steps = 512

ngrad = {k : jnp.zeros(v.shape) for k,v in layers.items()}

print('optimising:')
for itr in range(steps):
  print(f'\r  {itr+1} of {steps}', end='')

  momlayers = {k : v - momentum*ngrad[k] for k,v in layers.items()}
  dx = grad(momlayers, x, y)
  ngrad = {k : momentum*v + stepsize*dx[k] for k,v in ngrad.items()}
  layers = {k : v - ngrad[k] for k,v in layers.items()}

print()



# How well does it work now...
final_cost = cost(layers, x, y)
final_error = error(layers, x, y)

print(f'final cost = {final_cost:.3f}')
print(f'final error rate = {final_error*100:.1f}%')
