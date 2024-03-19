#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import sys, os, time

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
  
  hg_edges, hg_mass = res.solution.histogram()
  hgx = numpy.repeat(hg_edges, 2)[1:-1]
  hgy = numpy.repeat(hg_mass / (hg_edges[1:] - hg_edges[:-1]), 2)
  
  plt.figure(figsize=[12, 6])
  plt.plot(*model.graph())
  plt.plot(*res.solution.graph())
  plt.plot(hgx, hgy, ':',  linewidth=0.75)
  plt.savefig(f'gaussian,samples={samples},perbin={perbin}.svg')
  
  print()
  
  print(f'  mean: {base_model.mean():.3f} = {model.mean():.3f} -> {res.solution.mean():.3f}')
  print(f'   var: {base_model.var():.3f} = {model.var():.3f} -> {res.solution.var():.3f}')
  print()
