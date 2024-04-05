#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Compares the two versions of linear_crossentropy()

import sys
import os

import numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Record of worst mismatch found...
worst_err = -1.0
worst_fast = None
worst_safe = None
worst_p = None
worst_q = None



# Grid search for worst mismatch...
zeroish = 0
resolution = 5

for i, p_area in enumerate(numpy.linspace(zeroish, 1.0-zeroish, resolution)):
  print(f'\r{i} of {resolution}', end='')
  for q_area in numpy.linspace(zeroish, 1.0-zeroish, resolution):
    for p_bias in numpy.linspace(zeroish, 1.0-zeroish, resolution):
      for q_bias in numpy.linspace(zeroish, 1.0-zeroish, resolution):
        # Generate p and q arrays...
        p = 2.0*p_area*numpy.array([1-p_bias, p_bias])
        q = 2.0*q_area*numpy.array([1-q_bias, q_bias])

        # Query both versions of the cross entropy, including the gradients...
        fast = orogram.linear_crossentropy(1.0, p[0], p[1], q[0], q[1], True)
        safe = orogram.linear_crossentropy(1.0, p[0], p[1], q[0], q[1], False)

        # If worst mismatch thus far record...
        err = numpy.fabs(fast - safe)
        if err > worst_err:
          worst_err = err
          worst_fast = fast
          worst_safe = safe
          worst_p = p
          worst_q = q

print(f'\r{resolution} of {resolution}')
print()



# Report on worst...
print('Worst:')
print(f'  p = [{worst_p[0]:.4f}, {worst_p[1]:.4f}]')
print(f'  q = [{worst_q[0]:.4f}, {worst_q[1]:.4f}]')
print(f'  err = {worst_err}')
print(f'  fast xentropy = {worst_fast}')
print(f'  safe xentropy = {worst_safe}')

print()
