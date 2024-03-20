#! /usr/bin/env python3
# Copyright 2024 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import matplotlib.pyplot as plt


# Plots the infinite series of the stable version around the singularity, and approximates it...



# Calculate...
x = numpy.linspace(-1e-4, 1e-4, 1025)

y = numpy.zeros(x.shape)
for n in range(1, 1024*4, 2):
  y += numpy.power(x, n+2) / (n+2)



# Fit a cubic function, as it has an odd structure...
a = numpy.stack((numpy.ones(x.shape), x, numpy.square(x), numpy.power(x,3)), axis=1)

sol,*_ = numpy.linalg.lstsq(a, y, rcond=None)

print(f'Solution: {sol[0]} + {sol[1]}*x + {sol[2]}*x^2 + {sol[3]}*x^3')



# Generate approximation graph...
y_approx1 = (a @ sol)



# Also do a simplified approximation, after noting that only one term is really non-zero...
y_approx2 = numpy.power(x,3) / 3



# Plot...
plt.figure(figsize=[12, 6])

plt.plot(x, y, linewidth=1, label='Correct', color='C0')
plt.plot(x, y_approx1, linewidth=1, label='Approximation 1', color='C1')
plt.plot(x, y_approx2, linewidth=1, label='Approximation 2', color='C2')

plt.legend()
plt.savefig('singularity.pdf', bbox_inches='tight')
