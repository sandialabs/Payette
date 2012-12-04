# Copyright (2011) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
# rights in this software.

# The MIT License

# Copyright (c) Sandia Corporation

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

"""File contains several functions for creating and operating on tensors,
represented by 3x3 matrices.

For some functions, two levels of accuracy are present: full accuracy using
scipy functions if the user specified --strict when calling payette, or
(default) second order Taylor series approximations. With --strict, the
answers are more accurate, but the slow down is substantial.

The documentation for each individual function is currently very sparse, but
will be updated as time allows.

"""

import numpy as np
import numpy.linalg as la
import scipy
from scipy import linalg
from math import sqrt

import Source.Payette_utils as pu
import Source.__runopts__ as ro


# number of dimensions
NDIM = 3

# vector dimension and common vectors
NVEC = NDIM
Z3 = np.array([0.] * NVEC)
I3 = np.array([1.] * NVEC)
VEC_MAP = {0: "1", 1: "2", 2: "3"}

# symmetric tensor dimension and common symmetric tensors
NSYM = 6
Z6 = np.array([0.] * 6)
I6 = np.array([1., 1., 1., 0., 0., 0.])
SYM_MAP = {0: "11", 3: "12", 5: "13",
           1: "22", 4: "23",
           2: "33"}
SYMTENS_MAP = {0: (0, 0), 3: (0, 1), 5: (0, 2),
               1: (1, 1), 4: (1, 2),
               2: (2, 2)}

# tensor dimension and common tensors
NTENS = 9
Z9 = np.array([0.] * NTENS)
I9 = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
TENS_MAP = {0: "11", 1: "12", 2: "13",
            3: "21", 4: "22", 5: "23",
            6: "31", 7: "32", 8: "33"}
TENSC_MAP = {0: (0, 0), 1: (0, 1), 2: (0, 2),
             3: (1, 0), 4: (1, 1), 5: (1, 2),
             6: (2, 0), 7: (2, 1), 8: (2, 2)}

# common matrices
I3X3 = np.eye(NDIM)
Z3X3 = np.zeros((NDIM, NDIM))

# list to access diagonal components of 3x3 matrix
DI3 = [[0, 1, 2], [0, 1, 2]]
W = np.array([1., 1., 1., 2., 2., 2.])


def to_matrix(a):
    """convert a to a matrix"""
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
        pu.report_and_raise_error(msg)
        return


def to_array(a, sym=True):
    """convert a to an array"""
    shape = np.shape(a)
    if shape[0] != shape[1] or shape[0] != 3:
        pu.report_and_raise_error('wrong shape [{0}]'.format(str(shape)))
        return 1
    if not sym:
        return np.array([a[0, 0], a[0, 1], a[0, 2],
                         a[1, 0], a[1, 1], a[1, 2],
                         a[2, 0], a[2, 1], a[2, 2]],
                        dtype='double')

    a = 0.5 * (a + a.T)
    return np.array([a[0, 0], a[1, 1], a[2, 2],
                     a[0, 1], a[1, 2], a[0, 2]], dtype='double')


def powm(a, m):
    """return the matrix power of a"""
    if isdiag(a) and not ro.STRICT:
        a[DI3] = np.diag(a) ** m
    else:
        eig_val, eig_vec = la.eigh(a)
        a = np.dot(np.dot(eig_vec, np.diag(eig_val ** m)), eig_vec.T)

    return a


def expm(a):
    """return the matrix exponential of a"""
    if isdiag(a) and not ro.STRICT:
        a[DI3] = np.exp(np.diag(a))
    elif ro.STRICT:
        a = np.real(scipy.linalg.expm(a))
    else:
        a = I3X3 + a + np.dot(a, a) / 2.
    return a


def sqrtm(a):
    """return the matrix square root of a"""
    if np.isnan(a).any() or np.isinf(a).any():
        msg = "Probably reaching the numerical limits for the " +\
              "magnitude of the deformation."
        pu.report_and_raise_error(msg)
    if isdiag(a) and not ro.STRICT:
        a[DI3] = np.sqrt(np.diag(a))
    elif ro.STRICT:
        a = np.real(scipy.linalg.sqrtm(a))
    else:
        a = powm(a, 0.5)
    return a


def logm(a):
    """return the matrix log of a"""
    if isdiag(a) and not ro.STRICT:
        a[DI3] = np.log(np.diag(a))
    elif ro.STRICT:
        a = np.real(scipy.linalg.logm(a))
    else:
        a = ((a - I3X3) - np.dot(a - I3X3, a - I3X3) / 2. +
             np.dot(a - I3X3, np.dot(a - I3X3, a - I3X3)) / 3.)
    return a


def isdiag(a):
    """return True if a is diagonal else False"""
    tol = 1.e-16
    return all([abs(x) <= tol for x in [a[0, 1], a[0, 2],
                                        a[1, 0], a[1, 2],
                                        a[2, 0], a[2, 1]]])


def to_mig(a):
    """convert array a to mig ordering"""
    if len(a) == 6 or len(a) == 3:
        return np.array(a)
    if len(a) == 9:
        return np.array([a[0], a[3], a[5], a[6],
                         a[1], a[4], a[8], a[7], a[2]])


def ata(a):
    """ Compute Transpose(a).a

    Parameters
    ----------
    a: tensor a stored as 9x1 Voight array

    Returns
    -------
    ata : array_like
      Symmetric tensor defined by Transpose(a).a
      stored as 6x1 Voight array

    """
    return np.array([a[0] * a[0] + a[3] * a[3] + a[6] * a[6],
                     a[1] * a[1] + a[4] * a[4] + a[7] * a[7],
                     a[2] * a[2] + a[5] * a[5] + a[8] * a[8],
                     a[0] * a[1] + a[3] * a[4] + a[6] * a[7],
                     a[1] * a[2] + a[4] * a[5] + a[7] * a[8],
                     a[0] * a[2] + a[3] * a[5] + a[6] * a[8]])


def get_symleaf(x):
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
    x : array_like
      any matrix (in conventional 3x3 storage)

    Returns
    -------
    L : array_like
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

    L = np.zeros((6, 6))
    xm = x.reshape(3, 3)

    for i in range(3):
        for j in range(3):
            L[i, j] = xm[i, j] ** 2
            continue
        L[i, 3] = sqrt(2.) * xm[i, 0] * xm[i, 1]
        L[i, 4] = sqrt(2.) * xm[i, 1] * xm[i, 2]
        L[i, 5] = sqrt(2.) * xm[i, 2] * xm[i, 0]
        L[3, i] = sqrt(2.) * xm[0, i] * xm[1, i]
        L[4, i] = sqrt(2.) * xm[1, i] * xm[2, i]
        L[5, i] = sqrt(2.) * xm[2, i] * xm[0, i]
        continue

    L[3, 3] = xm[0, 1] * xm[1, 0] + xm[0, 0] * xm[1, 1]
    L[4, 3] = xm[1, 1] * xm[2, 0] + xm[1, 0] * xm[2, 1]
    L[5, 3] = xm[2, 1] * xm[0, 0] + xm[2, 0] * xm[0, 1]

    L[3, 4] = xm[0, 2] * xm[1, 1] + xm[0, 1] * xm[1, 2]
    L[4, 4] = xm[1, 2] * xm[2, 1] + xm[1, 1] * xm[2, 2]
    L[5, 4] = xm[2, 2] * xm[0, 1] + xm[2, 1] * xm[0, 2]

    L[3, 5] = xm[0, 0] * xm[1, 2] + xm[0, 2] * xm[1, 0]
    L[4, 5] = xm[1, 0] * xm[2, 2] + xm[1, 2] * xm[2, 0]
    L[5, 5] = xm[2, 0] * xm[0, 2] + xm[2, 2] * xm[0, 0]

    return L


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
    tens = mandel * x

    if job == 0:
        # Compute the Mandel form of A:X
        res = np.dot(a, tens)

    elif job == 1:
        # Compute the Mandel form of X:A
        res = np.dot(tens, a)

    else:
        pu.report_and_raise_error("unknown job sent to dd66x6")

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
    a : array_like
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

    detf = det(f)
    return np.array(dd66x6(1, get_symleaf(f), a)) / detf


def unrot(a, r):
    """ unrotate a """
    return np.array(dd66x6(1, get_symleaf(inv(r)), a))


def rot(a, r):
    """ rotate a """
    return np.array(dd66x6(1, get_symleaf(r), a))


def inv(a):
    """ return the inverse of a, a stored as 9x1 Voight arrray

    Ordering is fortran ordering

    """
    ainv = np.zeros(9)
    deta = det(a)
    ainv[0] = (a[4] * a[8] - a[7] * a[5])
    ainv[1] = (a[7] * a[2] - a[1] * a[8])
    ainv[2] = (a[1] * a[5] - a[4] * a[2])
    ainv[3] = (a[5] * a[6] - a[8] * a[3])
    ainv[4] = (a[8] * a[0] - a[2] * a[6])
    ainv[5] = (a[2] * a[3] - a[5] * a[0])
    ainv[6] = (a[3] * a[7] - a[6] * a[4])
    ainv[7] = (a[6] * a[1] - a[0] * a[7])
    ainv[8] = (a[0] * a[4] - a[3] * a[1])
    return ainv / deta


def ddp(a, b):
    """ double dot product of symmetric second order tensors a and b """
    return np.sum(W * a * b)


def mag(a):
    """ magnitude of symmetric second order tensor a """
    return sqrt(ddp(a, a))


def dev(a):
    """ deviatoric part of symmetric second order tensor a """
    return a - iso(a)


def iso(a):
    """ isotropic part of symmetric second order tensor a """
    return ddp(a, I6) / 3. * I6


def trace(a):
    """trace of the second order tensor a"""
    return ddp(a, I6)


def dot(a, b):
    """dot product of tensors a and b"""
    sa, sb = a.shape, b.shape
    if sa == sb == (6, ):
        return np.array([a[0] * b[0] + a[3] * b[3] + a[5] * b[5],  # 1, 1
                         a[1] * b[1] + a[3] * b[3] + a[4] * b[4],  # 2, 2
                         a[2] * b[2] + a[4] * b[4] + a[5] * b[5],  # 3, 3
                         a[3] * b[1] + a[0] * b[3] + a[5] * b[4],  # 1, 2
                         a[4] * b[2] + a[1] * b[4] + a[3] * b[5],  # 2, 3
                         a[5] * b[2] + a[3] * b[4] + a[0] * b[5]])  # 1, 3

    if sa == sb == (9, ):
        return np.array([a[0] * b[0] + a[1] * b[2] + a[2] * b[6],  # 1, 1
                         a[0] * b[1] + a[1] * b[4] + a[2] * b[7],  # 1, 2
                         a[0] * b[2] + a[1] * b[5] + a[2] * b[8],  # 1, 3
                         a[3] * b[0] + a[4] * b[3] + a[5] * b[6],  # 2, 1
                         a[3] * b[1] + a[4] * b[4] + a[5] * b[7],  # 2, 2
                         a[3] * b[2] + a[4] * b[5] + a[5] * b[8],  # 2, 3
                         a[6] * b[0] + a[7] * b[3] + a[8] * b[6],  # 3, 1
                         a[6] * b[1] + a[7] * b[4] + a[8] * b[7],  # 3, 2
                         a[6] * b[2] + a[7] * b[5] + a[8] * b[8]])  # 3, 3

    if sa == (9, ) and sb == (6, ):
        sa, sb = sb, sa
        a, b = np.array(b), np.array(a)

    if sa == (6, ) and sb == (9, ):
        return np.array([a[0] * b[0] + a[3] * b[3] + a[5] * b[6],  # 1, 1
                         a[0] * b[1] + a[3] * b[4] + a[5] * b[7],  # 1, 2
                         a[0] * b[2] + a[3] * b[5] + a[5] * b[8],  # 1, 3
                         a[3] * b[0] + a[1] * b[3] + a[4] * b[6],  # 2, 1
                         a[3] * b[1] + a[1] * b[4] + a[4] * b[7],  # 2, 2
                         a[3] * b[2] + a[1] * b[5] + a[4] * b[8],  # 2, 3
                         a[5] * b[0] + a[4] * b[3] + a[2] * b[6],  # 3, 1
                         a[5] * b[1] + a[4] * b[4] + a[2] * b[7],  # 3, 2
                         a[5] * b[2] + a[4] * b[5] + a[2] * b[8]])  # 3, 3

    pu.report_and_raise_error("Bad tensors sent to dot")
    return


def det(f):
    """Determine the determinant of the array f

    Parameters
    ----------
    f : array_like
        The array

    Returns
    -------
    det(f) : float
        The determinant of the array f

    Notes
    -----

    """
    fs = f.shape
    if fs == (3, 3):
        f = f.reshape(9)
    elif fs == (6,):
        f = to_matrix(f).reshape(9)
    if f.shape != (9, ):
        pu.report_and_raise_error("Bad array sent to det")
    return (f[0] * f[4] * f[8] + f[1] * f[5] * f[6] + f[2] * f[3] * f[7] -
            (f[0] * f[5] * f[7] + f[1] * f[3] * f[8] + f[2] * f[4] * f[6]))


def cross(a, b):
    """Cross product of a and b

    Parameters
    ----------
    a : array_like
    b : array_like

    Returns
    -------
    c : array_like

    """
    if len(a) != len(b) != NVEC:
        pu.report_and_raise_error("Bad length vector sent to cross")

    return np.array([-a[2] * b[1] + a[1] * b[2],
                     a[2] * b[0] - a[0] * b[2],
                     -a[1] * b[0] + a[0] * b[1]])
