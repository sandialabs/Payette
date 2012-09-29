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


DI3 = [[0, 1, 2], [0, 1, 2]]

I3X3 = np.eye(3)
Z3X3 = np.zeros((3, 3))

Z3 = np.array([0., 0., 0.])
I3 = np.array([1., 1., 1.])
Z6 = np.array([0., 0., 0.,
                   0., 0.,
                       0.])

I6 = np.array([1., 1., 1., 0., 0., 0.])

Z9 = np.array([0., 0., 0.,
               0., 0., 0.,
               0., 0., 0.])
I9 = np.array([1., 0., 0.,
               0., 1., 0.,
               0., 0., 1.])

TENS_MAP = {0: (0, 0), 1: (0, 1), 2: (0, 2),
            3: (1, 0), 4: (1, 1), 5: (1, 2),
            6: (2, 0), 7: (2, 1), 8: (2, 2)}

SYMTENS_MAP = {0: (0, 0), 3: (0, 1), 5: (0, 2),
                          1: (1, 1), 4: (1, 2),
                                     2: (2, 2)}

SYM_MAP = {0: "11", 3: "12", 5: "13",
                    1: "22", 4: "23",
                             2: "33"}

VEC_MAP = {0: (0,), 1: (1,), 2: (2,)}


def to_matrix(arga):
    """convert arga to a matrix"""
    if len(arga) == 6:
        return np.matrix([[arga[0], arga[3], arga[5]],
                          [arga[3], arga[1], arga[4]],
                          [arga[5], arga[4], arga[2]]], dtype='double')
    elif len(arga) == 9:
        return np.matrix([[arga[0], arga[1], arga[2]],
                          [arga[3], arga[4], arga[5]],
                          [arga[6], arga[7], arga[8]]], dtype='double')
    else:
        msg = "wrong size array of size [{0:d}]".format(len(arga))
        pu.report_and_raise_error(msg)
        return


def to_array(arga, symmetric=True):
    """convert arga to an array"""
    shape = np.shape(arga)
    if shape[0] != shape[1] or shape[0] != 3:
        pu.report_and_raise_error('wrong shape [{0}]'.format(str(shape)))
        return 1
    if not symmetric:
        return np.array([arga[0, 0], arga[0, 1], arga[0, 2],
                         arga[1, 0], arga[1, 1], arga[1, 2],
                         arga[2, 0], arga[2, 1], arga[2, 2]],
                        dtype='double')

    arga = 0.5 * (arga + arga.T)
    return np.array([arga[0, 0], arga[1, 1], arga[2, 2],
                     arga[0, 1], arga[1, 2], arga[0, 2]], dtype='double')


def powm(arga, pwm):
    """return the matrix power of arga"""
    if isdiag(arga) and not ro.STRICT:
        arga[DI3] = np.diag(arga) ** pwm
    else:
        eig_val, eig_vec = la.eigh(arga)
        arga = np.dot(np.dot(eig_vec, np.diag(eig_val ** pwm)), eig_vec.T)

    return arga


def expm(arga):
    """return the matrix exponential of arga"""
    if isdiag(arga) and not ro.STRICT:
        arga[DI3] = np.exp(np.diag(arga))
    elif ro.STRICT:
        arga = np.real(scipy.linalg.expm(arga))
    else:
        arga = I3X3 + arga + np.dot(arga, arga) / 2.
    return arga


def sqrtm(arga):
    """return the matrix square root of arga"""
    if np.isnan(arga).any() or np.isinf(arga).any():
        msg = "Probably reaching the numerical limits for the " +\
              "magnitude of the deformation."
        pu.report_and_raise_error(msg)
    if isdiag(arga) and not ro.STRICT:
        arga[DI3] = np.sqrt(np.diag(arga))
    elif ro.STRICT:
        arga = np.real(scipy.linalg.sqrtm(arga))
    else:
        arga = powm(arga, 0.5)
    return arga


def logm(arga):
    """return the matrix log of arga"""
    if isdiag(arga) and not ro.STRICT:
        arga[DI3] = np.log(np.diag(arga))
    elif ro.STRICT:
        arga = np.real(scipy.linalg.logm(arga))
    else:
        arga = ((arga - I3X3) - np.dot(arga - I3X3, arga - I3X3) / 2. +
             np.dot(arga - I3X3, np.dot(arga - I3X3, arga - I3X3)) / 3.)
    return arga


def isdiag(arga):
    """return True if arga is diagonal else False"""
    tol = 1.e-16
    return all([abs(x) <= tol for x in [arga[0, 1], arga[0, 2],
                                        arga[1, 0], arga[1, 2],
                                        arga[2, 0], arga[2, 1]]])


def to_mig(arga):
    """convert array arga to mig ordering"""
    if len(arga) == 6 or len(arga) == 3:
        return np.array(arga)
    if len(arga) == 9:
        return np.array([arga[0], arga[3], arga[5], arga[6],
                         arga[1], arga[4], arga[8], arga[7], arga[2]])


def ata(arga):
    """ Compute Transpose(arga).arga

    Parameters
    ----------
    arga: tensor arga stored as 9x1 Voight array

    Returns
    -------
    ata : array_like
      Symmetric tensor defined by Transpose(arga).arga
      stored as 6x1 Voight array

    """
    return np.array([arga[0] * arga[0] + arga[3] * arga[3] + arga[6] * arga[6],
                     arga[1] * arga[1] + arga[4] * arga[4] + arga[7] * arga[7],
                     arga[2] * arga[2] + arga[5] * arga[5] + arga[8] * arga[8],
                     arga[0] * arga[1] + arga[3] * arga[4] + arga[6] * arga[7],
                     arga[1] * arga[2] + arga[4] * arga[5] + arga[7] * arga[8],
                     arga[0] * arga[2] + arga[3] * arga[5] + arga[6] * arga[8]])


def get_symleaf(farg):
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
    farg : array_like
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
    fmat = farg.reshape(3, 3)

    for i in range(3):
        for j in range(3):
            symleaf[i, j] = fmat[i, j] ** 2
            continue
        symleaf[i, 3] = sqrt(2.) * fmat[i, 0] * fmat[i, 1]
        symleaf[i, 4] = sqrt(2.) * fmat[i, 1] * fmat[i, 2]
        symleaf[i, 5] = sqrt(2.) * fmat[i, 2] * fmat[i, 0]
        symleaf[3, i] = sqrt(2.) * fmat[0, i] * fmat[1, i]
        symleaf[4, i] = sqrt(2.) * fmat[1, i] * fmat[2, i]
        symleaf[5, i] = sqrt(2.) * fmat[2, i] * fmat[0, i]
        continue

    symleaf[3, 3] = fmat[0, 1] * fmat[1, 0] + fmat[0, 0] * fmat[1, 1]
    symleaf[4, 3] = fmat[1, 1] * fmat[2, 0] + fmat[1, 0] * fmat[2, 1]
    symleaf[5, 3] = fmat[2, 1] * fmat[0, 0] + fmat[2, 0] * fmat[0, 1]

    symleaf[3, 4] = fmat[0, 2] * fmat[1, 1] + fmat[0, 1] * fmat[1, 2]
    symleaf[4, 4] = fmat[1, 2] * fmat[2, 1] + fmat[1, 1] * fmat[2, 2]
    symleaf[5, 4] = fmat[2, 2] * fmat[0, 1] + fmat[2, 1] * fmat[0, 2]

    symleaf[3, 5] = fmat[0, 0] * fmat[1, 2] + fmat[0, 2] * fmat[1, 0]
    symleaf[4, 5] = fmat[1, 0] * fmat[2, 2] + fmat[1, 2] * fmat[2, 0]
    symleaf[5, 5] = fmat[2, 0] * fmat[0, 2] + fmat[2, 2] * fmat[0, 0]

    return symleaf


def dd66x6(job, arga, argx):
    """Compute the product of a fourth-order tensor arga times a second-order
    tensor argx (or vice versa if job=-1)
                 arga:argx if job=1
                 argx:arga if job=-1

    Parameters
    ----------
    arga : array_like
      6x6 Mandel matrix for a general (not necessarily major-sym)
      fourth-order minor-sym matrix
    argx : array_like
      6x6 Voigt matrix

    returns
    -------
       arga:argx if JOB=1
       argx:arga if JOB=-1

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

    # Construct the Mandel version of argx
    tens = mandel * argx

    if job == 0:
        # Compute the Mandel form of A:X
        res = np.dot(arga, tens)

    elif job == 1:
        # Compute the Mandel form of X:A
        res = np.dot(tens, arga)

    else:
        pu.report_and_raise_error("unknown job sent to dd66x6")

    # Convert result to Voigt form
    res = res / mandel

    return res


def push(arga, argf):
    """ Performs the "push" transformation

                1
               ---- argf.arga.Transpose(argf)
               detf

    For example, if arga is the Second-Piola Kirchoff stress, then the push
    transofrmation returns the Cauchy stress

    Parameters
    ----------
    arga : array_like
      6x1 Voigt array
    argf : array_like
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

    detf = (argf[0] * argf[4] * argf[8] +
            argf[1] * argf[5] * argf[6] +
            argf[2] * argf[3] * argf[7] -
           (argf[0] * argf[5] * argf[7] +
            argf[1] * argf[3] * argf[8] +
            argf[2] * argf[4] * argf[6]))
    return np.array(dd66x6(1, get_symleaf(argf), arga)) / detf


def unrot(arga, argr):
    """ unrotate arga """
    return np.array(dd66x6(1, get_symleaf(inv(argr)), arga))


def rot(arga, argr):
    """ rotate arga """
    return np.array(dd66x6(1, get_symleaf(argr), arga))


def inv(arga):
    """ return the inverse of arga, a stored as 9x1 Voight arrray

    Ordering is fortran ordering

    """
    ainv = np.zeros(9)
    deta = (arga[0] * arga[4] * arga[8] +
            arga[1] * arga[5] * arga[6] +
            arga[2] * arga[3] * arga[7] -
           (arga[0] * arga[5] * arga[7] +
            arga[1] * arga[3] * arga[8] +
            arga[2] * arga[4] * arga[6]))
    ainv[0] = (arga[4] * arga[8] - arga[7] * arga[5])
    ainv[1] = (arga[7] * arga[2] - arga[1] * arga[8])
    ainv[2] = (arga[1] * arga[5] - arga[4] * arga[2])
    ainv[3] = (arga[5] * arga[6] - arga[8] * arga[3])
    ainv[4] = (arga[8] * arga[0] - arga[2] * arga[6])
    ainv[5] = (arga[2] * arga[3] - arga[5] * arga[0])
    ainv[6] = (arga[3] * arga[7] - arga[6] * arga[4])
    ainv[7] = (arga[6] * arga[1] - arga[0] * arga[7])
    ainv[8] = (arga[0] * arga[4] - arga[3] * arga[1])
    return ainv / deta


def ddp(arga, argb):
    """ double dot product of symmetric second order tensors arga and argb """
    sym_fac = np.array([1., 1., 1., 2., 2., 2.])
    return np.sum(sym_fac * arga * argb)


def mag(arga):
    """ magnitude of symmetric second order tensor arga """
    return sqrt(ddp(arga, arga))


def dev(arga):
    """ deviatoric part of symmetric second order tensor arga """
    return arga - iso(arga)


def iso(arga):
    """ isotropic part of symmetric second order tensor arga """
    return ddp(arga, I6) / 3. * I6

def trace(arga):
    """trace of the second order tensor arga"""
    return ddp(arga, I6)
