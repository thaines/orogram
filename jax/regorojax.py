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
  above = jnp.clip(above, 1, bins-1)
  
  # Weight assigned to above-1 and above...
  delta = spacing(low, high, bins)
  below_weight = jnp.clip((changepoints[above] - x) / delta, 0.0, 1.0)
  above_weight = 1 - below_weight
  
  # Sum to create an orogram...
  density = jnp.bincount(above-1, below_weight, length=bins)
  density += jnp.bincount(above, above_weight, length=bins)

  # Correct for end bins being half the size...
  density = density.at[0].mul(2)
  density = density.at[-1].mul(2)
  
  # Normalise to integrate to 1 and return...
  area = 0.5 * (density[:-1] + density[1:]).sum() * (high - low) / (density.shape[0]-1)
  return density / jnp.maximum(area, 1e-10)



@jax.jit
def crossentropy(p, q, delta):
  """Returns the cross entropy between two regular orograms with aligned and evenly spaced bin centers, given as p and q. delta is the spacing between bins. Will be approximate in some situations, as it dodges around infinities and singularities to remain stable whatever you give it."""
  
  # First term requires what is effectively the relative area, which is what this represents...
  halved_ends = jnp.ones(p.shape[0])
  halved_ends = halved_ends.at[0].set(0.5)
  halved_ends = halved_ends.at[-1].set(0.5)

  # Assorted basic terms...
  log_q = jnp.log(jnp.maximum(q,1e-32))

  pdelta = p[1:] - p[:-1]
  qdelta = q[1:] - q[:-1]
  qsum = q[:-1] + q[1:]

  qsqr = jnp.square(q)
  top = p[1:]*qsqr[:-1] - p[:-1]*qsqr[1:]

  # Inner term of infinite loop (used elsewhere), done in a stable way, plus variant with extra qsum...
  notzero = qsum>1e-5
  qsum_safe = jnp.maximum(qsum, 1e-5)
  inner = qdelta / qsum_safe
  inner_ds2 = qdelta / (jnp.square(qsum_safe)*qsum_safe)

  # Do the stable parts...
  ret = -(halved_ends * p * log_q).sum()
  ret += 0.25 * (pdelta * inner).sum()

  # Do the two branches, with stability hacks for the unstable one...
  ## Unstable but accurate when qdelta is high...
  abs_qdelta = jnp.fabs(qdelta)
  sign_qdelta = -2 * (jnp.signbit(qdelta) - 0.5)

  qdelta_sqr_safe = jnp.maximum(jnp.square(qdelta), 1e-10)
  qdelta_qsum_safe = sign_qdelta * jnp.maximum(abs_qdelta * qsum, 1e-10)

  ret_unstable = top * (0.5 * (log_q[1:] - log_q[:-1]) / qdelta_sqr_safe - 1 / qdelta_qsum_safe)

  ## Stable but only accurate when qdelta is low...
  ret_approx = top * (1 / 3 + jnp.square(inner) / 5) * inner_ds2

  ## Pick the right branch for each and sum in...
  ret += jax.lax.select(abs_qdelta>1e-5, ret_unstable, ret_approx).sum()

  return delta*ret



@jax.jit
def crossentropy_safe(p, q, delta):
  """Same as crossentropy() but it is super safe, as in does everything in a numerically safe but slow way."""

  log_q = jnp.log(jnp.maximum(q,1e-32))

  pdelta = p[1:] - p[:-1]
  qdelta = q[1:] - q[:-1]
  qsum = jnp.maximum(q[:-1] + q[1:], 1e-8)

  plog_q = p * log_q
  ret = -0.5 * (plog_q[:-1] + plog_q[1:]).sum()

  ret += 0.25 * (pdelta * qdelta / qsum).sum()

  # Do this term brute force; isn't great if inner=1 however...
  inner = qdelta / qsum
  scales = (p[1:]*jnp.square(q[:-1]) - p[:-1]*jnp.square(q[1:]))  / jnp.square(qsum)

  for n in range(1, 1024, 2):
    ret += (scales * jnp.power(inner, n)).sum() / (n + 2)

  return delta*ret



grad_crossentropy = jax.jit(jax.grad(crossentropy, (0,1)))
grad_crossentropy_safe = jax.jit(jax.grad(crossentropy_safe, (0,1)))
