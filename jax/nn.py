#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import jax
import jax.numpy as jnp



def random_init_glorot(widths, rng):
  """Generates random weights for a NN - using the Glorot initialisation techneque, uniform version, as suitable for activations with a mean of zero. widths is a list of layer widths, with the length of the input feature vector in position [0] and the length of the output feature vector in position [-1]. Adds a plus one to the number of columns each time to account for a bias constant. Returns a dictionary of the right format to pass as layers to mlp_relu() etc."""
  ret = dict()

  for i, (c, r, key) in enumerate(zip(widths[:-1], widths[1:], jax.random.split(rng, len(widths)-1))):
    scale = jnp.sqrt(6 / (c+r))
    ret[i] = jax.random.uniform(key, (r,c+1), minval=-scale, maxval=scale)

  return ret



def random_init_he(widths, rng):
  """Generates random weights for a NN - using the He initialisation techneque, uniform version, as suitable for ReLU. widths is a list of layer widths, with the length of the input feature vector in position [0] and the length of the output feature vector in position [-1]. Adds a plus one to the number of columns each time to account for a bias constant. Returns a dictionary of the right format to pass as layers to mlp_relu() etc."""
  ret = dict()

  for i, (c, r, key) in enumerate(zip(widths[:-1], widths[1:], jax.random.split(rng, len(widths)-1))):
    scale = jnp.sqrt(6 / c)
    ret[i] = jax.random.uniform(key, (r,c+1), minval=-scale, maxval=scale)

  return ret



@jax.jit
def mlp_relu(layers, vec):
  """A simple neural network with ReLU non-linearities on each layer except the last. The layers parameter is a dictionary where each layer appears as an integer key, starting at 0. The value attached to the key is hence a matrix, such that it multiplies the vector from the previous layer. Appends a constant (value 1) to the vector before sending it into each layer to handle biases. vec is a vector representing a single data point. Obviously all the shapes better line up!"""
  i = 0
  while i in layers:
    # ReLU unless first...
    if i!=0:
      vec = jnp.maximum(vec, 0)

    # Apply weights...
    vec = (layers[i][:,:-1] @ vec) + layers[i][:,-1]

    # To next layer...
    i += 1

  return vec



## Takes a data matrix, i.e. second parameter is indexed [exemplar, feature]...
mlp_relu_dm = jax.jit(jax.vmap(mlp_relu, in_axes=(None, 0)))



@jax.jit
def mlp_gauss(layers, vec):
  """A simple neural network with Gaussian activations on each layer except the last. The layers parameter is a dictionary where each layer appears as an integer key, starting at 0. The value attached to the key is hence a matrix, such that it multiplies the vector from the previous layer. Appends a constant (value 1) to the vector before sending it into each layer to handle biases. vec is a vector representing a single data point. Obviously all the shapes better line up!"""
  i = 0
  while i in layers:
    # Gaussian unless first...
    if i!=0:
      vec = jnp.exp(-jnp.square(vec))

    # Apply weights...
    vec = (layers[i][:,:-1] @ vec) + layers[i][:,-1]

    # To next layer...
    i += 1

  return vec



## Takes a data matrix, i.e. second parameter is indexed [exemplar, feature]...
mlp_gauss_dm = jax.jit(jax.vmap(mlp_gauss, in_axes=(None, 0)))
