# Copyright 2023 Tom SF Haines

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import numpy

try:
  import pyximport
  pyximport.install(setup_args={'include_dirs': [numpy.get_include()]})

except:
  pass

from . import xentropy



def linear_crossentropy(length, p0, p1, q0, q1, fast = False):
  """Gives direct access to the internal calculation of the cross entropy of a single linear section. length is the length of the section, then {p,q}{0,1} give the end probabilities of the section. For testing/visualisation as it's very slow to do anything with this — have to call it too many times and it's not vectorised."""
  log_q0 = numpy.log(q0) if q0 >= 1e-64 else -150
  log_q1 = numpy.log(q1) if q1 >= 1e-64 else -150
  return length * xentropy.section_crossentropy(p0, p1, q0, q1, log_q0, log_q1, fast)
