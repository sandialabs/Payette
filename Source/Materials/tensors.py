import numpy as np
from math import sqrt

w = np.array([1., 1., 1., 2., 2., 2.])
delta = np.array([1., 1., 1., 0., 0., 0.])

def tada(a):
    """ Compute Transpose(a).a

    Parameters
    ----------
    a: tensor a stored as 9x1 Voight array

    Returns
    -------
    tada : array_like
      Symmetric tensor defined by Transpose(a).a stored as 6x1 Voight array

    """
    tada = np.zeros(6)
    tada[0] = a[0] * a[0] + a[3] * a[3] + a[6] * a[6]
    tada[1] = a[1] * a[1] + a[4] * a[4] + a[7] * a[7]
    tada[2] = a[2] * a[2] + a[5] * a[5] + a[8] * a[8]
    tada[3] = a[0] * a[1] + a[3] * a[4] + a[6] * a[7]
    tada[4] = a[1] * a[2] + a[4] * a[5] + a[7] * a[8]
    tada[5] = a[0] * a[2] + a[3] * a[5] + a[6] * a[8]
    return tada


def symleaf(farg):
    """Compute the 6x6 Mandel matrix (with index mapping {11,22,33,12,23,31})
    that is the sym-leaf transformation of the input 3x3 matrix F.

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

    symleaf = np.zeros((6,6))
    f = farg.reshape(3,3)

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
