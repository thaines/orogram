#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os, time

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orogram

rng = numpy.random.default_rng(0)



# Three vertex triangle - a super basic test with a known answer...
print('Triangle with 3 vertices:')
base_model1 = orogram.RegOrogram(0.2)
base_model1.add([0.0])

model1 = orogram.Orogram(base_model1)
res1 = model1.simplify(base_model1.sum(), 1)

print(f'  bins: {len(model1)} -> {len(res1.solution)}')
print(f'  cost: {base_model1.sum()*model1.entropy() + res1.priorall:.3f} -> {res1.cost:.3f}')
print(f'  kl(model || solution) = {model1.kl(res1.solution):.6f}')
  
plt.figure(figsize=[12, 6])
plt.plot(*model1.graph())
plt.plot(*res1.solution.graph())
plt.savefig(f'triangle3.svg')

print()



# A 13 vertex triangle - like before, but with more bins...
print('Triangle with 13 vertices:')
base_model2 = orogram.RegOrogram(0.2)
for layer in range(6):
  base_model2.add(0.2 * numpy.arange(-layer, layer+1, dtype=float))

model2 = orogram.Orogram(base_model2)
res2 = model2.simplify(base_model2.sum(), 1)

print(f'  bins: {len(model2)} -> {len(res2.solution)}')
print(f'  cost: {base_model2.sum()*model2.entropy() + res2.priorall:.3f} -> {res2.cost:.3f}')
print(f'  kl(model || solution) = {model2.kl(res2.solution):.6f}')
  
plt.figure(figsize=[12, 6])
plt.plot(*model2.graph())
plt.plot(*res2.solution.graph())
plt.savefig(f'triangle13.svg')

print()



# A pyramid, so a stepped triangle...
print('Pyramid with 13 vertices:')
base_model3 = orogram.RegOrogram(0.05)
for layer in range(6):
  base_model3.add(0.05 * numpy.arange(-3*layer, layer*3+1, dtype=float))

model3 = orogram.Orogram(base_model3)
res3 = model3.simplify(base_model3.sum(), 5)

print(f'  bins: {len(model3)} -> {len(res3.solution)}')
print(f'  cost: {base_model3.sum()*model3.entropy() + res3.priorall:.3f} -> {res3.cost:.3f}')
print(f'  kl(model || solution) = {model3.kl(res3.solution):.6f}')
  
plt.figure(figsize=[12, 6])
plt.plot(*model3.graph())
plt.plot(*res3.solution.graph())
plt.savefig(f'pyramid33.svg')

print()
