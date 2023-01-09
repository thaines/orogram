#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os, time

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orogram

rng = numpy.random.default_rng(0)



# Quick triangle, because it's a super basic test with a known answer...
print('triangle:')
base_model = orogram.RegOrogram(0.2)
base_model.add([0.0])

model = orogram.Orogram(base_model)
res = model.simplify(1, 1)

print(f'  bins: {len(model)} -> {len(res.solution)}')
print(f'  cost: {model.entropy() + res.priorcost} -> {res.cost}')
print(f'  no prior kl loss = {model.kl(res.solution)}')
  
plt.figure(figsize=[12, 6])
plt.plot(*model.graph())
plt.plot(*res.solution.graph())
plt.savefig(f'triangle.svg')

print()



# Simplify a Gaussian for various parameters...
for samples, perbin in [(16,16), (64,16), (256,16), (1024,16), (1024,1), (1024,128), (1024*32,16), (1024*32,128), (1024*32,256)]:
  print(f'Gaussian(samples = {samples}; per bin = {perbin}):')
  
  data = rng.standard_normal(samples)
  
  base_model = orogram.RegOrogram(0.05)
  base_model.add(data)
  
  model = orogram.Orogram(base_model)
  start = time.time()
  res = model.simplify(samples, perbin)
  end = time.time()
  
  print(f'  bins: {len(model)} -> {len(res.solution)}')
  print(f'  cost: {model.entropy() + res.priorcost} -> {res.cost}')
  print(f'  no prior kl loss = {model.kl(res.solution)}')
  print(f'  time = {1000*(end - start):.3f}ms')
  
  plt.figure(figsize=[12, 6])
  plt.plot(*model.graph())
  plt.plot(*res.solution.graph())
  plt.savefig(f'gaussian,samples={samples},perbin={perbin}.svg')
  
  print()
