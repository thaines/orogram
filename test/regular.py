#! /usr/bin/env python3
# Copyright 2022 Tom SF Haines

import sys, os

import numpy
from scipy.stats import norm, uniform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orogram



# Run through basic features to check it at least runs and behaves sensibly...
model = orogram.RegOrogram(0.5)
rng = numpy.random.default_rng(0)
data = rng.standard_normal(1024*1024) * 2


print(f'sizeof before data = {sys.getsizeof(model)}')
model.add(data)
print(f'sizeof after data = {sys.getsizeof(model)}')
print(f'parameters = {len(model)}')
print()

print(f'sum = {model.sum():.1f}')
print(f'min = {model.min():.3f} (bin {model.binmin()})')
print(f'max = {model.max():.3f} (bin {model.binmax()})')
print()

print('Central weights:')
print('  ' + ', '.join([f'{v:.3f}' for v in model.weight([-3, -2, -1, 0, 1, 2, 3])]))
print()

print('Central pdf:')
print('  ' + ', '.join([f'{v:.3f}' for v in model([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])]))
print()

print('Modes:')
print('  by x: {}'.format(', '.join([f'{v:.3f}' for v in model.modes()])))
print('  by bin: {}'.format(', '.join([f'{v}' for v in model.binmodes()])))
for i in model.binmodes():
  print(f'    weight around {i}: ' + ', '.join([f'{v:.3f}' for v in model.weight([i-1,i,i+1])]))
print(f'  highest = {model.highest()} (bin {model.binhighest()})')
print()

print('CDF:')
print('  by x: {}'.format(', '.join([f'{v:.3f}' for v in model.cdf([-0.75, -0.5, 0.0, 0.5, 0.75])])))
print('  by bin: {}'.format(', '.join([f'{v:.3f}' for v in model.bincdf([-10, -1, 0, 1, 10])])))
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


model2 = orogram.RegOrogram(0.5)
data2 = rng.standard_normal(1024)*2 + 0.5
model2.add(data2)

print('Cross entropy, H(N(0,1), N(0.5,4)), same spacing:')
print(f'  analytic = {model.crossentropy(model2):.6f}')
print(f'  numerical integration = {model.crossentropynumint(model2):.6f}')
print(f'  monte-carlo = {model.crossentropymc(model2):.6f}')
print()


model3 = orogram.RegOrogram(0.4)
data3 = rng.standard_normal(1024)*3 + 1.0
model3.add(data3)

print('Cross entropy, H(N(0,1), N(1,9)), different spacing:')
print(f'  analytic = {model.crossentropy(model3):.6f}')
print(f'  numerical integration = {model.crossentropynumint(model3):.6f}')
print(f'  monte-carlo = {model.crossentropymc(model3):.6f}')
print()

print('KL-divergence, H(N(0,1), N(1,9)), different spacing:')
print(f'  analytic = {model.kl(model3):.6f}')
print()



# Now do tests with a gap to make sure those code paths don't falter...
rv1 = norm(loc=-2.5, scale=0.5)
rv2 = norm(loc=2.5, scale=0.5)

model4 = orogram.RegOrogram(0.1, blocksize=16)
model4.bake(rv1.cdf, -5, 5, weight=3)
model4.bake(rv2.cdf, -5, 5, weight=3)

model5 = orogram.RegOrogram(0.25, blocksize=24)
model5.add(model4.draw(1024*4, rng))

print('Cross entropy, between mixtures (with gap):')
print(f'  H(true,sampled) = {model4.crossentropy(model5):.6f}')
print(f'  H(true,sampled) = {model4.crossentropynumint(model5):.6f} (numerical integration)')
print(f'  H(true,sampled) = {model4.crossentropymc(model5):.6f} (monte carlo integration)')
print(f'  H(sampled,true) = {model5.crossentropy(model4):.6f}')
print(f'  H(sampled,true) = {model5.crossentropynumint(model4):.6f} (numerical integration)')
print(f'  H(sampled,true) = {model5.crossentropymc(model4):.6f} (monte carlo integration)')
print()
