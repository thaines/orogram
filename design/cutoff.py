#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import time
import numpy
import matplotlib.pyplot as plt



# Generates various graphs of the error as the two q approach, for the version with a singularity and the stable-but-slow version...



def generate(p0, p1, qmean, qmax, fn, steps = 1024+1):
  """Sweeps q0 and q1 so the mean is as given and the """
  print(f'{fn}:')

  # Generate q values...
  q0 = numpy.linspace(qmean-qmax, qmean+qmax, steps)
  q1 = q0[::-1]

  qdelta = q1 - q0
  qsum = q0 + q1
  log_q0 = numpy.log(numpy.maximum(q0, 1e-64))
  log_q1 = numpy.log(numpy.maximum(q1, 1e-64))


  # Numerical integration...
  start = time.time()
  numint = numpy.zeros(steps)
  block0 = numpy.linspace(0.0, 1.0, 256+1)
  for i in range(block0.shape[0]-1):
    block1 = numpy.linspace(block0[i], block0[i+1], 256+1)
    numint1 = numpy.zeros(steps)
    for j in range(block1.shape[0]-1):
      t = numpy.linspace(block1[j],block1[j+1], 256, False)[:,None]
      numint1 -= numpy.mean(((1-t)*p0 + t*p1) * numpy.log((1-t)*q0 + t*q1), axis=0)
    numint1 /= block1.shape[0] - 1
    numint += numint1
  numint /= block0.shape[0] - 1
  end = time.time()
  print(f'  numerical integration = {end-start}s')

  # Do the stable evaluation, with a silly number of iterations to be sure...
  start = time.time()
  stable = -0.5 * (p0*log_q0 + p1*log_q1)
  stable += 0.25 * (p1 - p0) * qdelta / qsum

  mult = (p1*q0*q0 - p0*q1*q1) / (qsum*qsum)
  log_qinner = numpy.log(numpy.fabs(qdelta) / qsum)
  sign = numpy.sign(qdelta)

  n = numpy.arange(1, 256, 2)[:,None]
  stable += (mult * sign * numpy.exp(log_qinner*n) / (n + 2)).sum(axis=0)
  end = time.time()
  print(f'  stable = {end-start}s')

  # Do the unstable evaluation...
  start = time.time()
  unstable = ((p1*q0*q0 - p0*q1*q1) * (log_q1 - log_q0)) / qdelta
  unstable += 0.5 * ((3*p0 + p1)*q1 - (p0 + 3*p1)*q0)
  unstable /= 2 * qdelta
  unstable -= 0.5 * (p0*log_q0 + p1*log_q1)
  end = time.time()
  print(f'  fast = {end-start}s')

  # Generate a graph...
  plt.figure(figsize=[6, 3])
  plt.xlabel(r'$q_1 - q_0$')
  plt.ylabel('Segment cross entropy')

  plt.plot(q1 - q0, numint, linewidth=6, label='Numerical integration')
  plt.plot(q1 - q0, stable, linewidth=3, label='Stable')
  plt.plot(q1 - q0, unstable, linewidth=1.5, label='Fast')

  plt.legend()
  plt.savefig(fn, bbox_inches='tight')



epsilon = 1e-5
generate(0.1, 0.1, 0.1, epsilon, '0.1.pdf')
generate(0.1, 0.1, 0.2, epsilon, '0.2.pdf')
generate(0.1, 0.1, 0.4, epsilon, '0.4.pdf')
generate(0.1, 0.1, 0.8, epsilon, '0.8.pdf')
generate(0.1, 0.1, 1.6, epsilon, '1.6.pdf')
