#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys, os

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Consider a variety of pdfs and do a sweep over approximating the whole pdf with a linear section, ensuring the probability mass matches throughout...
tasks = [([0.0, 0.5, 1.0], [2.0, 0.0, 2.0], 'sawtooth'),
         ([0.0, 0.5, 1.0], [4.0, 0.0, 0.0], 'downhill'),
         ([0.0, 0.25, 0.5, 0.75, 1.0], [2.0, 0.0, 1.5, 1.5, 0.0], 'wiggle')]

t = numpy.linspace(0.0, 1.0, 256)
crossentropy = numpy.empty(t.shape[0])
crossentropy_fast = numpy.empty(t.shape[0])

for px, py, name in tasks:
  # Calculate cross entropy...
  crossentropy[:] = 0.0
  crossentropy_fast[:] = 0.0

  for i in range(crossentropy.shape[0]):
    for s in range(len(px)-1):
      pstart = (1 - px[s]) * (1 - t[i]) + px[s] * t[i]
      pend = (1 - px[s+1]) * (1 - t[i]) + px[s+1] * t[i]
      crossentropy[i] += orogram.linear_crossentropy(px[s+1] - px[s], py[s], py[s+1], pstart, pend, False)
      crossentropy_fast[i] += orogram.linear_crossentropy(px[s+1] - px[s], py[s], py[s+1], pstart, pend, True)

  # Plot a graph...
  fig, ax1 = plt.subplots(figsize=[6, 6])

  ax1.set_xlabel('x')
  ax1.set_xlim([0.0, 1.0])
  ax1.set_ylabel('probability', color='blue')
  ax1.set_ylim([0.0, max(py)*1.05])
  ax1.plot(px, py, color='blue')

  ax2 = ax1.twinx()
  ax2.set_ylabel('nats', color='red')
  ax2.set_ylim([0.0, crossentropy.max()*1.05])
  ax2.plot(t, crossentropy, color='red')

  ax2.plot(t, crossentropy_fast, color='green', linestyle=':')

  fig.savefig(f'vis_{name}.svg', bbox_inches='tight')
