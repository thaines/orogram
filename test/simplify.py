#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orogram

rng = numpy.random.default_rng(0)



# Simplify for various parameters...
for samples, perbin in [(16,16), (64,16), (256,16), (1024,16), (1024,1), (1024,128), (1024*32,16)]:
  print(f'samples = {samples}; per bin = {perbin}:')
  
  data = rng.standard_normal(samples)
  
  base_model = orogram.RegOrogram(0.05)
  base_model.add(data)
  
  model = orogram.Orogram(base_model)
  res = model.simplify(samples, perbin)
  
  print(f'  bins: {len(model)} -> {len(res.solution)}')
  print(f'  cost: {res.startcost} -> {res.cost}')
  print(f'  loss kl = {model.kl(res.solution)}')
  
  plt.figure(figsize=[12, 6])
  plt.plot(*model.graph())
  plt.plot(*res.solution.graph())
  plt.savefig(f'samples={samples},perbin={perbin}.svg')
  
  print()
