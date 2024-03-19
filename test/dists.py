#! /usr/bin/env python3
# Copyright 2023 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy
import scipy.special



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



# Setup a parameterised mixture models, with one each of the above distribtions, represented with vectorised pdf and cdf functions with a parameter vector...
## Parameter vector is:
## [4xmixture weight, 4xcenters, 4xscales] = length 12
## Mixture components in order of above functions
def mix4_pdf(x, param):
  return param[0]*uniform_pdf(x, param[4], param[8]) + param[1]*triangular_pdf(x, param[5], param[9]) + param[2]*gaussian_pdf(x, param[6], param[10]) + param[3]*laplace_pdf(x, param[7], param[11])


def mix4_cdf(x, param):
  return param[0]*uniform_cdf(x, param[4], param[8]) + param[1]*triangular_cdf(x, param[5], param[9]) + param[2]*gaussian_cdf(x, param[6], param[10]) + param[3]*laplace_cdf(x, param[7], param[11])



def mix4_params(count, rng = None):
  """Generates an array of parameters for the mix4 distribution; as a 2D array where each row is the parameters for a single randomised model. The first parameter is the number of rows, i.e. how many models to generate. Second a numpy rng, which supports everything that can be passed to default_rng(). Designed to work well if considering the range from -4 to 4."""
  rng = numpy.random.default_rng(rng)
  ret = numpy.empty((count, 12))

  ret[:,0:4] = rng.dirichlet(numpy.ones(4), count)
  ret[:,4:8] = rng.uniform(-2.5, 2.5, (count, 4))
  ret[:,8:12] = rng.uniform(0.1, 1.0, (count, 4))

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
