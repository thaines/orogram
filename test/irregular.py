#! /usr/bin/env python3
# Copyright 2022 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys, os

import numpy
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import orogram



# Construct a test model from a RegOrogram, as that's the usual use case...
rv = norm(loc=5, scale=0.5)
base_model = orogram.RegOrogram(0.1)
base_model.bake_cdf(rv.cdf, 0, 10)

model = orogram.Orogram(base_model)



# Run through basic features to check it at least runs and behaves sensibly...
print(f'sizeof = {sys.getsizeof(model)}')
print(f'parameters = {len(model)}')
print()

print(f'min = {model.min():.3f}')
print(f'max = {model.max():.3f}')
print()

print('Central pdf:')
print('  ' + ', '.join([f'{v:.3f}' for v in model([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])]))
print()

print('Modes:')
print('  by x: {}'.format(', '.join([f'{v:.3f}' for v in model.modes()])))
print('  by bin: {}'.format(', '.join([f'{v}' for v in model.binmodes()])))
for i in model.binmodes():
  print(f'    probablity around {i}: ' + ', '.join([f'{v:.3f}' for v in model.prob([i-1,i,i+1])]))
print(f'  highest = {model.highest()} (bin {model.binhighest()})')
print()

print('Above:')
thresh = model(model.highest()) * 0.75
print(f'  threshold = {thresh:.3f}')
mass, ranges = model.above(thresh, True)
print(f'  mass = {mass:.3f}')
print('  ranges:')
for start, end, mass in ranges:
  print(f'    {start:.3f} — {end:.3f} = {mass:.3f}')
print()

print('Credible region (95%):')
for start, end, mass in model.credible():
  print(f'    {start:.3f} — {end:.3f} = {mass:.3f}')
print()

print('CDF:')
print('  by x: {}'.format(', '.join([f'{v:.3f}' for v in model.cdf([4.25, 4.5, 5.0, 5.5, 5.75])])))
print('  by bin: {}'.format(', '.join([f'{v:.3f}' for v in model.bincdf([len(model)//2 - 10, len(model)//2, len(model)//2 + 10])])))
print()

print('Draw:')
print('  examples: {}'.format(', '.join([f'{v:.3f}' for v in model.draw(4)])))
more = model.draw(1024*8)
print(f'  mean of many: {more.mean():.3f}')
print(f'  var of many: {more.var():.3f}')
print()

print(f'median = {model.median()}')
print(f'mean = {model.mean()}')
print(f'variance = {model.var()}')
print()

print('Entropy:')
print(f'  analytic = {model.entropy():.6f}')
print(f'  numerical integration = {model.entropynumint():.6f}')
print(f'  monte-carlo = {model.entropymc():.6f}')
print()



# Need another model to test cross entropy...
rv = norm(loc=4, scale=2)
base_model = orogram.RegOrogram(0.1)
base_model.bake_cdf(rv.cdf, -2, 10)

model2 = orogram.Orogram(base_model)

print('Cross entropy, H(N(5,0.5), N(4,2)):')
print(f'  analytic = {model.crossentropy(model2):.6f}')
print(f'  numerical integration = {model.crossentropynumint(model2):.6f}')
print(f'  monte-carlo = {model.crossentropymc(model2):.6f}')
print()
