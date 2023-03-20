#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os, time

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orogram

rng = numpy.random.default_rng(0)



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
  print(f'  cost: {samples*model.entropy() + res.priorall:.3f} -> {res.cost:.3f}')
  print(f'  kl(model || solution) = {model.kl(res.solution):.6f}')
  print(f'  time = {1000*(end - start):.3f}ms')
  
  plt.figure(figsize=[12, 6])
  plt.plot(*model.graph())
  plt.plot(*res.solution.graph())
  plt.savefig(f'gaussian,samples={samples},perbin={perbin}.svg')
  
  print()
  
  print(f'  mean: {base_model.mean():.3f} = {model.mean():.3f} -> {res.solution.mean():.3f}')
  print(f'   var: {base_model.var():.3f} = {model.var():.3f} -> {res.solution.var():.3f}')
  print()
