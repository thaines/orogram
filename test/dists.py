#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

import numpy
import scipy



# Define a set of distributions with slightly unusual paramters...
def uniform_pdf(x, centre, radius):
  ret = numpy.greater(x, centre-radius).astype(float)
  ret *= numpy.less_equal(x, centre+radius).astype(float)
  ret /= 2 * radius
  return ret


def uniform_cdf(x, centre, radius):
  return numpy.clip((x - centre + radius) / (2*radius), 0.0, 1.0)



def triangular_pdf(x, centre, radius):
  ret = 1.0 - numpy.fabs(x - centre) / radius
  ret = numpy.maximum(ret, 0.0)
  return ret / radius


def triangular_cdf(x, centre, radius):
  x = numpy.clip(x, centre-radius, centre+radius)
  upper = numpy.greater(x, centre).astype(float)
  sign = 2 * (upper - 0.5)

  ret = numpy.square(centre + sign*radius - x) / (2*radius*radius)
  ret *= -sign
  ret += upper

  return ret



def gaussian_pdf(x, mean, sd):
  ret = numpy.exp(-0.5 * numpy.square((x - mean) / sd))
  ret /= sd * numpy.sqrt(2*numpy.pi)
  return ret


def gaussian_cdf(x, mean, sd):
  return 0.5 * (1 + scipy.special.erf((x - mean) / (sd * numpy.sqrt(2))))



def laplace_pdf(x, mean, scale):
  return numpy.exp(-numpy.fabs(x - mean) / scale) / (2*scale)


def laplace_cdf(x, mean, scale):
  ret = 0.5 * numpy.exp(-numpy.fabs(x - mean) / scale)

  upper = numpy.greater(x, mean).astype(float)
  ret *= -2 * (upper - 0.5)
  ret += upper

  return ret



if __name__=='__main__':
  import os, sys
  import functools

  import matplotlib.pyplot as plt
  plt.rcParams['text.usetex'] = True

  sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  import orogram


  # Verify above by plotting pdf and cdf converted into pdf, to check they match and look good...
  low = -4
  high = 4

  for name, pdf, cdf, param in [('uniform', uniform_pdf, uniform_cdf, {'centre' : 0.5, 'radius' : 1.2}),
                              ('triangular', triangular_pdf, triangular_cdf, {'centre' : -0.5, 'radius' : 1.4}),
                              ('gaussian', gaussian_pdf, gaussian_cdf, {'mean' : -0.8, 'sd' : 0.3}),
                              ('laplace', laplace_pdf, laplace_cdf, {'mean' : 0.3, 'scale' : 0.7})]:
    # Generate specific instance...
    pdf = functools.partial(pdf, **param)
    cdf = functools.partial(cdf, **param)

    # Evaluate pdf...
    x = numpy.linspace(low, high, 2048)
    y = pdf(x)

    # Evaluate cdf...
    yc = cdf(x)

    # Convert cdf to pdf...
    base_model = orogram.RegOrogram(0.05)
    base_model.bake(cdf, low, high)
    model = orogram.Orogram(base_model)

    # Plot and write graph...
    plt.figure(figsize=[6, 3])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$P(x)$')

    plt.plot(x, y)
    plt.plot(*model.graph())
    plt.plot(x, yc, ':')
    plt.savefig(f'dist_{name}.pdf', bbox_inches='tight')
