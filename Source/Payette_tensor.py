# The MIT License

# Copyright (c) 2011 Tim Fuller

# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
import numpy.linalg as la
import scipy
from scipy import linalg

from Source.Payette_utils import *

di3 = [[0,1,2],[0,1,2]]
I = np.eye(3)

'''
NAME
   Payette_tensor.py

PURPOSE
   File contains several functions for creating and operating on tensors,
   represented by 3x3 matrices.

   For some functions, two levels of accuracy are present: full accuracy using
   scipy functions if the user specified --strict when calling runPayette, or
   (default) second order Taylor series approximations. With --strict, the
   answers are more accurate, but the slow down is substantial.

   The documentation for each individual function is currently very sparse, but
   will be updated as time allows.

AUTHORS
   Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
'''

def toMatrix(a):
    if len(a) == 6:
        return np.matrix([ [ a[0], a[3], a[5] ],
                           [ a[3], a[1], a[4] ],
                           [ a[5], a[4], a[2] ] ],dtype='double')
    elif len(a) == 9:
        return np.matrix([ [ a[0], a[3], a[5] ],
                           [ a[6], a[1], a[4] ],
                           [ a[8], a[7], a[2] ] ],dtype='double')
    else:
        msg = 'wrong size array of size [%i] sent to toMatrix'%(len(a))
        reportError(__file__,msg)
        return

def toArray(a,symmetric=True):
    shape = np.shape(a)
    if shape[0] != shape[1] or shape[0] != 3:
        reportError(__file__,'wrong shape of [%s] sent to toArray'%str(shape))
        return 1
    if not symmetric:
        return np.array([ a[0,0], a[1,1], a[2,2], a[0,1], a[1,2],
                          a[0,2], a[1,0], a[2,1], a[2,0] ],dtype='double')
    a = 0.5*(a + a.T)
    return np.array([ a[0,0], a[1,1], a[2,2],
                      a[0,1], a[1,2], a[0,2] ],dtype='double')

def powm(a,m,strict=False):
    if isDiagonal(a) and not strict: a[di3] = np.diag(a)**m
    else:
        w,v = la.eigh(a)
        a = np.dot(np.dot(v,np.diag(w**m)),v.T)
        pass
    return a

def expm(a,strict=False):
    if isDiagonal(a) and not strict: a[di3] = np.exp(np.diag(a))
    elif strict: a = np.real(scipy.linalg.expm(a))
    else: a = I + a + np.dot(a,a)/2.
    return a

def sqrtm(a,strict=False):
    if np.isnan(a).any() or np.isinf(a).any():
        msg = "Probably reaching the numerical limits for the " +\
              "magnitude of the deformation."
        reportError(__file__,msg)
    if isDiagonal(a) and not strict: a[di3] = np.sqrt(np.diag(a))
    elif strict: a = np.real(scipy.linalg.sqrtm(a))
    else: a = powm(a,0.5)
    return a

def logm(a,strict=False):
    if isDiagonal(a) and not strict: a[di3] = np.log(np.diag(a))
    elif strict: a = np.real(scipy.linalg.logm(a))
    else: a = (a-I)-np.dot(a-I,a-I)/2.+np.dot(a-I,np.dot(a-I,a-I))/3.
    return a

def isDiagonal(a):
    return bool(a[0,1] == 0. and a[1,2] == 0. and a[0,2] == 0. and
                a[1,0] == 0. and a[2,1] == 0. and a[2,0] == 0.)
