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
from math import sqrt

import Source.Payette_utils as pu

nvec = 3
nsym = 6
ntens = 9

tol = 1.e-16

di3 = [[0, 1, 2],[0, 1, 2]]

I3x3 = np.eye(3)
Z3x3 = np.zeros((3, 3))

Z3 = np.array([0., 0., 0.])
I3 = np.array([1., 1., 1.])
Z6 = np.array([0., 0., 0.,
                   0., 0.,
                       0.])

w = np.array([1., 1., 1., 2., 2., 2.])
I6 = np.array([1., 1., 1., 0., 0., 0.])
delta = np.array([1., 1., 1., 0., 0., 0.])

Z9 = np.array([0., 0., 0.,
               0., 0., 0.,
               0., 0., 0.])
I9 = np.array([1., 0., 0.,
               0., 1., 0.,
               0., 0., 1.])

tens_map = { 0:(0, 0), 1:(0, 1), 2:(0, 2),
             3:(1, 0), 4:(1, 1), 5:(1, 2),
             6:(2, 0), 7:(2, 1), 8:(2, 2) }

symtens_map = { 0:(0, 0), 3:(0, 1), 5:(0, 2),
                          1:(1, 1), 4:(1, 2),
                                    2:(2, 2) }

sym_map = {0: "11", 3: "12", 5: "13",
                    1: "22", 4: "23",
                             2: "33",}

vec_map = {0:(0,), 1:(1,), 2:(2,)}

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

def to_matrix(a):
    iam = "to_matrix"
    if len(a) == 6:
        return np.matrix([[a[0], a[3], a[5]],
                          [a[3], a[1], a[4]],
                          [a[5], a[4], a[2]]], dtype='double')
    elif len(a) == 9:
        return np.matrix([[a[0], a[1], a[2]],
                          [a[3], a[4], a[5]],
                          [a[6], a[7], a[8]]], dtype='double')
    else:
        msg = "wrong size array of size [{0:d}]".format(len(a))
        pu.reportError(iam, msg)
        return

def to_array(a, symmetric=True):
    iam = "to_array"
    shape = np.shape(a)
    if shape[0] != shape[1] or shape[0] != 3:
        pu.reportError(iam, 'wrong shape [{0}]'.format(str(shape)))
        return 1
    if not symmetric:
        return np.array([a[0, 0], a[0, 1], a[0, 2],
                         a[1, 0], a[1, 1], a[1, 2],
                         a[2, 0], a[2, 1], a[2, 2]], dtype='double')

    a = 0.5*(a + a.T)
    return np.array([a[0, 0], a[1, 1], a[2, 2],
                     a[0, 1], a[1, 2], a[0, 2]], dtype='double')

def powm(a, m, strict=False):
    if isdiag(a) and not strict:
        a[di3] = np.diag(a)**m
    else:
        l, v = la.eigh(a)
        a = np.dot(np.dot(v, np.diag(l ** m)), v.T)

    return a

def expm(a, strict=False):
    if isdiag(a) and not strict:
        a[di3] = np.exp(np.diag(a))
    elif strict:
        a = np.real(scipy.linalg.expm(a))
    else:
        a = I3x3 + a + np.dot(a, a)/2.
    return a

def sqrtm(a, strict=False):
    if np.isnan(a).any() or np.isinf(a).any():
        msg = "Probably reaching the numerical limits for the " +\
              "magnitude of the deformation."
        pu.reportError(__file__, msg)
    if isdiag(a) and not strict:
        a[di3] = np.sqrt(np.diag(a))
    elif strict:
        a = np.real(scipy.linalg.sqrtm(a))
    else:
        a = powm(a, 0.5)
    return a

def logm(a, strict=False):
    if isdiag(a) and not strict:
        a[di3] = np.log(np.diag(a))
    elif strict:
        a = np.real(scipy.linalg.logm(a))
    else:
        a = ((a-I3x3) - np.dot(a - I3x3, a - I3x3) / 2. +
             np.dot(a - I3x3, np.dot(a - I3x3, a - I3x3)) / 3.)
    return a

def isdiag(a):
    return all([abs(x) <= tol for x in [       a[0, 1], a[0, 2],
                                        a[1, 0],        a[1, 2],
                                        a[2, 0], a[2, 1]       ]])

def to_mig(a):
    if len(a) == 6 or len(a) == 3:
        return np.array(a)
    if len(a) == 9:
        return np.array([a[0], a[3], a[5], a[6], a[1], a[4], a[8], a[7], a[2]])

def ata(a):
    """ Compute Transpose(a).a

    Parameters
    ----------
    a: tensor a stored as 9x1 Voight array

    Returns
    -------
    ata : array_like
      Symmetric tensor defined by Transpose(a).a stored as 6x1 Voight array

    """
    return np.array([a[0] * a[0] + a[3] * a[3] + a[6] * a[6],
                     a[1] * a[1] + a[4] * a[4] + a[7] * a[7],
                     a[2] * a[2] + a[5] * a[5] + a[8] * a[8],
                     a[0] * a[1] + a[3] * a[4] + a[6] * a[7],
                     a[1] * a[2] + a[4] * a[5] + a[7] * a[8],
                     a[0] * a[2] + a[3] * a[5] + a[6] * a[8]])


def symleaf(farg):
    """Compute the 6x6 Mandel matrix (with index mapping {11, 22, 33, 12, 23,
    31}) that is the sym-leaf transformation of the input 3x3 matrix F.

    If A is any symmetric tensor, and if {A} is its 6x1 Mandel array, then
    the 6x1 Mandel array for the tensor B=F.A.Transpose[F] may be computed
    by
                      {B}=[FF]{A}

    If F is a deformation F, then B is the "push" (spatial) transformation
    of the reference tensor A If F is Inverse[F], then B is the "pull"
    (reference) transformation of the spatial tensor A, and therefore B
    would be Inverse[FF]{A}.

    If F is a rotation, then B is the rotation of A, and FF would be be a
    6x6 orthogonal matrix, just as is F

    Parameters
    ----------
    f : array_like
      any matrix (in conventional 3x3 storage)

    Returns
    -------
    symleaf : array_like
      6x6 Mandel matrix for the sym-leaf transformation matrix

    authors
    -------
    Rebecca Brannon:theory, algorithm, and code

    modification history
    --------------------
    yymmdd|who|what was done
    ------ --- -------------
    060915|rmbrann|created routine
    120514|tjfulle|conversion to f90

    """

    symleaf = np.zeros((6, 6))
    f = farg.reshape(3, 3)

    for i in range(3):
        for j in range(3):
            symleaf[i, j] = f[i, j] ** 2
        continue
        symleaf[i, 3] = sqrt(2.) * f[i, 0] * f[i, 1]
        symleaf[i, 4] = sqrt(2.) * f[i, 1] * f[i, 2]
        symleaf[i, 5] = sqrt(2.) * f[i, 2] * f[i, 0]
        symleaf[3, i] = sqrt(2.) * f[0, i] * f[1, i]
        symleaf[4, i] = sqrt(2.) * f[1, i] * f[2, i]
        symleaf[5, i] = sqrt(2.) * f[2, i] * f[0, i]
        continue

    symleaf[3, 3] = f[0, 1] * f[1, 0] + f[0, 0] * f[1, 1]
    symleaf[4, 3] = f[1, 1] * f[2, 0] + f[1, 0] * f[2, 1]
    symleaf[5, 3] = f[2, 1] * f[0, 0] + f[2, 0] * f[0, 1]

    symleaf[3, 4] = f[0, 2] * f[1, 1] + f[0, 1] * f[1, 2]
    symleaf[4, 4] = f[1, 2] * f[2, 1] + f[1, 1] * f[2, 2]
    symleaf[5, 4] = f[2, 2] * f[0, 1] + f[2, 1] * f[0, 2]

    symleaf[3, 5] = f[0, 0] * f[1, 2] + f[0, 2] * f[1, 0]
    symleaf[4, 5] = f[1, 0] * f[2, 2] + f[1, 2] * f[2, 0]
    symleaf[5, 5] = f[2, 0] * f[0, 2] + f[2, 2] * f[0, 0]

    return symleaf

def dd66x6(job, a, x):
    """Compute the product of a fourth-order tensor a times a second-order
    tensor x (or vice versa if job=-1)
                 a:x if job=1
                 x:a if job=-1

    Parameters
    ----------
    a : array_like
      6x6 Mandel matrix for a general (not necessarily major-sym)
      fourth-order minor-sym matrix
    x : array_like
      6x6 Voigt matrix

    returns
    -------
       a:x if JOB=1
       x:a if JOB=-1

    authors
    -------
    Rebecca Brannon:theory, algorithm, and code

    modification history
    --------------------
    yymmdd|who|what was done
    ------ --- -------------
    060915|rmb|created routine
    120514|tjfulle|conversion to f90

    """

    mandel = np.array([1., 1., 1., sqrt(2.), sqrt(2.), sqrt(2.)])
    res = np.zeros(6)

    # Construct the Mandel version of x
    t = mandel * x

    if job == 0:
        # Compute the Mandel form of A:X
        res = np.dot(a, t)

    elif job == 1:
        # Compute the Mandel form of X:A
        res = np.dot(t, a)

    else:
        pu.reportError(iam, "unknown job sent to dd66x6")

    # Convert result to Voigt form
    res = res / mandel

    return res


def push(a, f):
    """ Performs the "push" transformation

                1
               ---- f.a.Transpose(f)
               detf

    For example, if a is the Second-Piola Kirchoff stress, then the push
    transofrmation returns the Cauchy stress

    Parameters
    ----------
    x : array_like
      6x1 Voigt array
    f : array_like
      9x1 deformation gradient, stored as Voight array

    returns
    -------
    push : array_like
      6x1 Voight array

    modification history
    --------------------
    yymmdd|who|what was done
    ------ --- -------------
    120514|tjfulle|created subroutine

    """

    detf = (f[0] * f[4] * f[8] + f[1] * f[5] * f[6] + f[2] * f[3] * f[7]
          -(f[0] * f[5] * f[7] + f[1] * f[3] * f[8] + f[2] * f[4] * f[6]))
    return np.array(dd66x6(1, symleaf(f), a)) / detf

def unrot(a, r):
    """ unrotate a """
    return np.array(dd66x6(1, symleaf(inv(r)), a))

def rot(a, r):
    """ rotate a """
    return np.array(dd66x6(1, symleaf(r), a))

def inv(a):
    """ return the inverse of a, a stored as 9x1 Voight arrray

    Ordering is fortran ordering

    """
    ai = np.zeros(9)
    deta = (a[0] * a[4] * a[8] + a[1] * a[5] * a[6] + a[2] * a[3] * a[7]
          -(a[0] * a[5] * a[7] + a[1] * a[3] * a[8] + a[2] * a[4] * a[6]))
    ai[0] = (a[4] * a[8] - a[7] * a[5])
    ai[1] = (a[7] * a[2] - a[1] * a[8])
    ai[2] = (a[1] * a[5] - a[4] * a[2])
    ai[3] = (a[5] * a[6] - a[8] * a[3])
    ai[4] = (a[8] * a[0] - a[2] * a[6])
    ai[5] = (a[2] * a[3] - a[5] * a[0])
    ai[6] = (a[3] * a[7] - a[6] * a[4])
    ai[7] = (a[6] * a[1] - a[0] * a[7])
    ai[8] = (a[0] * a[4] - a[3] * a[1])
    return ai / deta

def ddp(a, b):
    """ double dot product of symmetric second order tensors a and b """
    return np.sum(w * a * b)

def mag(a):
    """ magnitude of symmetric second order tensor a """
    return sqrt(ddp(a, a))

def dev(a):
    """ deviatoric part of symmetric second order tensor a """
    return a  - iso(a)

def iso(a):
    """ isotropic part of symmetric second order tensor a """
    return ddp(a, delta) / 3. * delta
