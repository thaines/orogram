#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys, os

import numpy

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import dists

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Setup a mixture model...
param = numpy.array([0.25, 0.25, 0.25, 0.25,
                     -2.25, -0.75, 0.75, 2.25,
                     0.6, 0.6, 0.3, 0.4])

pdf = lambda x: dists.mix4_pdf(x, param)
cdf = lambda x: dists.mix4_cdf(x, param)

x = numpy.linspace(-4, 4, 1024*8)
y = pdf(x)



# Bake...
model = orogram.Orogram.bake_cdf(cdf, -4, 4, 256)



# Report on success/not...
even, err = model.even(incerr=True)
if even:
  print('Converged')
else:
  print(f'Not converged; error of {err}')
print()



# Visualise...
plt.figure(figsize=[6, 3])
plt.xlabel(r'$x$')
plt.ylabel(r'$P(x)$')

plt.plot(x, y, linewidth=2)
plt.plot(*model.graph())

plt.savefig(f'bake_irregular.pdf', bbox_inches='tight')
