#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Compares many calls to crossentropy() and crossentropy_safe() and finds the parameters that result in the greatest difference, to validate the fast version is behaving.

import jax
import jax.numpy as jnp

from regorojax import *

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_debug_nans', True)



# Record of worst mismatch found...
worst = {}
things = ('xentropy', 'dp[0]', 'dp[1]', 'dq[0]', 'dq[1]')
for thing in things:
  worst[thing] = {'err' : -1.0, 'fast' : None, 'safe' : None,
                  'p' : None, 'q' : None,
                  ('fast','dp') : None, ('fast','dq') : None,
                  ('safe','dp') : None, ('safe','dq') : None}



# Grid search for worst mismatch...
zeroish = 0
resolution = 5

for i, p_area in enumerate(jnp.linspace(zeroish, 1.0-zeroish, resolution)):
  print(f'\r{i} of {resolution}', end='')
  for q_area in jnp.linspace(zeroish, 1.0-zeroish, resolution):
    for p_bias in jnp.linspace(zeroish, 1.0-zeroish, resolution):
      for q_bias in jnp.linspace(zeroish, 1.0-zeroish, resolution):
        # Generate p and q arrays...
        p = 2.0*p_area*jnp.array([1-p_bias, p_bias])
        q = 2.0*q_area*jnp.array([1-q_bias, q_bias])

        # Query both versions of the cross entropy, including the gradients...
        fast = crossentropy(p, q, 1)
        safe = crossentropy_safe(p, q, 1)

        gfast = grad_crossentropy(p, q, 1)
        gsafe = grad_crossentropy_safe(p, q, 1)

        # If worst mismatch thus far record...
        for thing, delta in [('xentropy', fast - safe),
                             ('dp[0]', gfast[0][0] - gsafe[0][0]),
                             ('dp[1]', gfast[0][1] - gsafe[0][1]),
                             ('dq[0]', gfast[1][0] - gsafe[1][0]),
                             ('dq[1]', gfast[1][1] - gsafe[1][1])]:
          err = jnp.fabs(delta)
          if err > worst[thing]['err']:
            worst[thing]['err'] = err
            worst[thing]['fast'] = fast
            worst[thing]['safe'] = safe
            worst[thing]['p'] = p
            worst[thing]['q'] = q
            worst[thing]['fast','dp'] = gfast[0]
            worst[thing]['fast','dq'] = gfast[1]
            worst[thing]['safe','dp'] = gsafe[0]
            worst[thing]['safe','dq'] = gsafe[1]


print(f'\r{resolution} of {resolution}')
print()



# Report on worst...
for thing in things:
  print(f'Worst of {thing}:')
  print(f'  p = [{worst[thing]["p"][0]:.4f}, {worst[thing]["p"][1]:.4f}]')
  print(f'  q = [{worst[thing]["q"][0]:.4f}, {worst[thing]["q"][1]:.4f}]')
  print(f'  err = {worst[thing]["err"]}')

  print(f'  fast xentropy = {worst[thing]["fast"]}')
  print(f'  safe xentropy = {worst[thing]["safe"]}')
  print(f'  fast dp = [{worst[thing]["fast","dp"][0]:.4f}, {worst[thing]["fast","dp"][1]:.4f}]')
  print(f'  safe dp = [{worst[thing]["safe","dp"][0]:.4f}, {worst[thing]["safe","dp"][1]:.4f}]')
  print(f'  fast dq = [{worst[thing]["fast","dq"][0]:.4f}, {worst[thing]["fast","dq"][1]:.4f}]')
  print(f'  safe dq = [{worst[thing]["safe","dq"][0]:.4f}, {worst[thing]["safe","dq"][1]:.4f}]')

  print()
