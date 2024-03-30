# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

from functools import partial

import jax
import jax.numpy as jnp



def spacing(low, high, bins):
  """Returns the spacing between bin centers of a regular orogram covering the given range with the given number of bin centers."""
  return (high - low) / (bins -1)



@partial(jax.jit, static_argnames=('bins',))
def orogram(x, low, high, bins):
  """Given an array of data points (x) this constructs a regular orogram using them, with linear interpolation of weights so there is a gradient. The lowest bin centre is parameter low, the highest is parameter high, and there are as many bin centers as the parameter bins, which must be a natural number of at least 2. Returns the bin heights, such that if you integrate the area underneath you get 1."""
  
  # The change points and the bin that's higher than each data point...
  changepoints = jnp.linspace(low, high, bins)
  above = jnp.digitize(x, changepoints)
  
  # Weight assigned to above-1 and above...
  delta = spacing(low, high, bins)
  below_weight = jnp.clip((changepoints[above] - x) / delta, 0.0, 1.0)
  above_weight = 1 - below_weight
  
  # Sum to create an orogram...
  density = jnp.bincount(above-1, below_weight, length=bins)
  density += jnp.bincount(above, above_weight, length=bins)
  
  # Normalise to integrate to 1 and return...
  area = 0.5 * (density[:-1] + density[1:]).sum() * (high - low) / (density.shape[0]-1)
  return density / area



@jax.jit
def crossentropy(p, q, delta):
  """Returns the cross entropy between two regular orograms with aligned and evenly spaced bin centers, given as p and q. delta is the spacing between bins. Will be approximate in some situations, as it dodges around infinities and singularities to remain stable whatever you give it."""
  
  halved_ends = jnp.ones(p.shape[0])
  halved_ends = halved_ends.at[0].set(0.5)
  halved_ends = halved_ends.at[-1].set(0.5)
  
  qsqr = jnp.square(q)
  log_q = jnp.log(jnp.maximum(q,1e-32))
  
  pdelta = p[1:] - p[:-1]
  qdelta = q[1:] - q[:-1]
  qsum = jnp.maximum(q[:-1] + q[1:], 1e-5)
  
  inner = qdelta / qsum
  top = p[1:]*qsqr[:-1] - p[:-1]*qsqr[1:]
  
  # Do the stable parts...
  ret = -(halved_ends * p * log_q).sum()
  ret += 0.25 * (pdelta * inner).sum()
  
  # Do the two branches, with stability hacks for the unstable one...
  abs_qdelta = jnp.fabs(qdelta)
  sign_qdelta = -2 * (jnp.signbit(qdelta) - 0.5)

  qdelta_sqr_safe = jnp.maximum(jnp.square(qdelta), 1e-5)
  qdelta_qsum_safe = sign_qdelta * jnp.maximum(abs_qdelta * qsum, 1e-5)
  
  ret_unstable = top * (0.5 * (log_q[1:] - log_q[:-1]) / qdelta_sqr_safe - 1 / qdelta_qsum_safe)
  ret_approx = (inner / 3 + jnp.square(inner)*inner / 5) * top / jnp.square(qsum)
  
  # Select for each segment and sum into the return...
  ret += jax.lax.select(abs_qdelta>1e-5, ret_unstable, ret_approx).sum()
  
  return delta*ret