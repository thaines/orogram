#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import sys, os

import numpy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import orogram

rng = numpy.random.default_rng(0)



# Generate a simple mixture model and two base models, one reflected...
data = numpy.append(rng.normal(-1.0, 2.0, 512), rng.normal(1.0, 1.0, 256))

base_model1 = orogram.RegOrogram(0.05)
base_model1.add(data)

base_model2 = orogram.RegOrogram(0.05)
base_model2.add(-data)



# Convert and simplify both models...
model1 = orogram.Orogram(base_model1)
res1 = model1.simplify(data.shape[0], 16)

model2 = orogram.Orogram(base_model2)
res2 = model2.simplify(data.shape[0], 16)

print(f'  bins: {len(res1.solution)} | {len(res2.solution)}')
print(f'  cost: {res1.cost:.3f} | {res2.cost:.3f}')
print(f'  mass: {res1.solution._mass():.6f} | {res2.solution._mass():.6f}')
print()

print('Indices kept (aligned):')
forwards = ''.join([('T' if v else 'F') for v in res1.kept])
backwards = ''.join([('T' if v else 'F') for v in res2.kept[::-1]])

for block in range(0, res1.kept.shape[0], 60):
  print('-> ' + forwards[block:block+60])
  print('<- ' + backwards[block:block+60])
  print()



# Generate graph, flipping second, such that they should perfectly match if all is well!..
plt.figure(figsize=[12, 6])
plt.plot(*model1.graph(), label='input')
plt.plot(*res1.solution.graph(), label='forwards')

x, y = res2.solution.graph()
plt.plot(-x, y, label='backwards')

plt.legend()
plt.savefig('symmetry.svg')