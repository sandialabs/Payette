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

"""Module containing methods for computing kinematic quantities"""

import numpy as np
import numpy.linalg as la

import Source.Payette_iterative_solvers as piter
import Source.Payette_utils as pu
import Source.__runopts__ as ro

from Source.Payette_tensor import I3X3, powm, expm, logm, sqrtm, to_array


def velgrad_from_strain(simdat, matdat):
    """From the value of the Seth-Hill parameter kappa, current strain E, and
    strain rate dEdt, return the symmetric and skew parts of the velocity
    gradient

    Parameters
    ----------
    simdat : data container object
       simulation data container
    matdat : data container object
       material data container

    Returns
    -------
    None

    Updates
    -------
    matdat "rate of deformation"
    matdat "vorticity"

    Theory
    ------
    Velocity gradient L is given by

                 L = dFdt * Finv
                   = dRdt*I*Rinv + R*dUdt*Uinv*Rinv

    where F, I, R, U are the deformation gradient, identity, rotation, and
    right stretch tensor, respectively. d*dt and *inv are the rate and inverse
    or *, respectively,

    The stretch U is given by

                 if k != 0:
                     U = (k*E + I)**(1/k)

                 else:
                     U = exp(E)

    and its rate

                     dUdt = 1/k*(k*E + I)**(1/k - 1)*k*dEdt
                          = (k*E + I)**(1/k)*(k*E + I)**(-1)*dEdt
                          = U*X*dEdt

                     where X = (kE + I)**(-1)

       Then

                 d = sym(L)
                 w = skew(L)
    """

    # get data from data containers
    kappa = simdat.KAPPA
    dt = simdat.get("time step")

    # strain and rotation
    strain_0 = matdat.get("strain", form="Matrix")
    strain_f = matdat.get("prescribed strain", form="Matrix")
    rotation = matdat.get("rotation", form="Matrix")
    drotation = matdat.get("rotation rate", form="Matrix")

    # rate of strain
    dstrain = (strain_f - strain_0) / dt

    # stretch and its rate
    U = right_stretch(kappa, strain_f)

    # center X on half step
    X = 0.5 * (la.inv(kappa * strain_f + I3X3) +
               la.inv(kappa * strain_0 + I3X3))
    dU = U * X * dstrain

    # velocity gradient, sym, and skew parts
    L = (drotation * rotation.T +
         rotation * dU * la.inv(U) * rotation.T)
    D = .5 * (L + L.T)
    W = L - D

    # store updated data
    matdat.save("rate of deformation", D)
    matdat.save("vorticity", W)

    return


def velgrad_from_defgrad(simdat, matdat):
    """From the value of the deformation gradient F, return the symmetric and
    skew parts of the velocity gradient

    Parameters
    ----------
    simdat : data container object
       simulation data container
    matdat : data container object
       material data container

    Returns
    -------
    None

    Updates
    -------
    matdat "rate of deformation"
    matdat "vorticity"

    Theory
    ------
    Velocity gradient L is given by

                 L = dFdt * Finv

    Then

                 d = sym(L)
                 w = skew(L)
    """

    # get data from containers
    dt = simdat.get("time step")
    F0 = matdat.get("deformation gradient", form="Matrix")
    Ff = matdat.get("prescribed deformation gradient", form="Matrix")
    dF = (Ff - F0) / dt

    L = .5 * dF * (la.inv(Ff) + la.inv(F0))
    D = .5 * (L + L.T)
    W = L - D

    matdat.save("vorticity", W)
    matdat.save("rate of deformation", D)
    return


def velgrad_from_stress(material, simdat, matdat, prsig, V):
    """Seek to determine the unknown components of the symmetric part of
    velocity gradient d[v] satisfying

                               P(d[v]) = Ppres[:]                      (1)

    where P is the current stress, d the symmetric part of the velocity
    gradient, v is a vector subscript array containing the components for
    which stresses (or stress rates) are prescribed, and Ppres[:] are the
    prescribed values at the current time.

    Parameters
    ----------
    material : constitutive model instance
    simdat : data container object
       simulation data container
    matdat : data container object
       material data container

    Returns
    -------
    None

    Updates
    -------
    matdat "rate of deformation"
    matdat "vorticity"

    Approach
    --------
    Solution is found iteratively in (up to) 3 steps
      1) Call newton to solve 1, return stress, xtra, d if converged
      2) Call newton with d[v] = 0. to solve 1, return stress, xtra, d
         if converged
      3) Call simplex with d[v] = 0. to solve 1, return stress, xtra, d

    History
    -------
    This is a python implementation of a fortran subroutine of the same name
    written by Tom Pucick for his MMD material model driver.

    """
    depsdt_old = matdat.get("strain rate")
    matdat.stash("strain rate")
    nV = len(V)

    if not ro.PROPORTIONAL:

        converged = piter.newton(material, simdat, matdat, prsig, V)

        if converged:
            matdat.save("rate of deformation",
                        matdat.get("strain rate", copy=True))
            return

        # --- didn't converge, try Newton's method with initial
        # --- d[V]=0.
        dstrain = matdat.get("strain rate")
        dstrain[V] = np.zeros(nV)
        matdat.save("strain rate", dstrain)

        converged = piter.newton(material, simdat, matdat)

        if converged:
            matdat.save("rate of deformation",
                        matdat.get("strain rate", copy=True))
            return

    # --- Still didn't converge. Try downhill simplex method and accept
    #     whatever answer it returns:
    matdat.unstash("strain rate")
    piter.simplex(material, simdat, matdat, V)
    matdat.save("rate of deformation",
                matdat.get("strain rate", copy=True))
    return


def update_deformation(simdat, matdat):
    """From the value of the Seth-Hill parameter kappa, current strain E,
    deformation gradient F, and symmetric part of the velocit gradient d,
    update the strain and deformation gradient.

    Parameters
    ----------
    simdat : data container object
       simulation data container
    matdat : data container object
       material data container

    Returns
    -------
    None

    Updates
    -------
    matdat "strain"
    matdat "deformation gradient"
    matdat "equivalent strain"

    Theory
    ------
    Deformation gradient is given by

                 dFdt = L*F                                             (1)

    where F and L are the deformation and velocity gradients, respectively.
    The solution to (1) is

                 F = F0*exp(Lt)

    Solving incrementally,

                 Fc = Fp*exp(Lp*dt)

    where the c and p refer to the current and previous values, respectively.

    With the updated F, Fc, known, the updated stretch is found by

                 U = (trans(Fc)*Fc)**(1/2)

    Then, the updated strain is found by

                 E = 1/k * (U**k - I)

    where k is the Seth-Hill strain parameter.

    """

    # get data from containers
    kappa = simdat.KAPPA
    dt = simdat.get("time step")
    F0 = matdat.get("deformation gradient", form="Matrix")
    D = matdat.get("rate of deformation", form="Matrix")
    W = matdat.get("vorticity", form="Matrix")

    Ff = expm((D + W) * dt) * F0
    U = sqrtm((Ff.T) * Ff)

    if kappa == 0:
        strain = logm(U)
    else:
        strain = 1. / kappa * (powm(U, kappa) - I3X3)

    if np.linalg.det(Ff) <= 0.:
        pu.report_and_raise_error("negative Jacobian encountered")

    matdat.save("strain", strain)
    matdat.save("deformation gradient", Ff)

    # compute the equivalent strain
    eps = to_array(strain, symmetric=True)
    eqveps = np.sqrt(2. / 3. * (sum(eps[:3] ** 2) + 2. * sum(eps[3:] ** 2)))
    matdat.save("equivalent strain", eqveps)

    return


def right_stretch(kappa, strain):
    """Compute the symmtetric part U (right stretch) of the polar
    decomposition of the deformation gradient F = RU.

    Parameters
    ----------
    kappa : float
      Seth-Hill strain parameter
    strain : array_like
      current strain tensor

    Returns
    -------
    U : array_like
      the right stretch tensor

    Theory
    ------
    The right stretch tensor U, in terms of the strain tensor E is given by

                     U = (k*E + I)**(1/k)

    where k is the Seth-Hill parameter and I is the identity tensor.

    """
    if kappa == 0.:
        return expm(np.matrix(strain))
    else:
        return powm(kappa * np.matrix(strain) + I3X3, 1. / kappa)


def left_stretch(kappa, strain):
    """Compute the symmtetric part V (left stretch) of the polar decomposition
    of the deformation gradient F = VR.

    Parameters
    ----------
    kappa : float
      Seth-Hill strain parameter
    strain : array_like
      current spatial strain tensor

    Returns
    -------
    stretch : array_like
      the left stretch tensor

    Theory
    ------
    The left stretch tensor V, in terms of the strain tensor e is given by

                     V = (k*e + I)**(1/k)

    where k is the Seth-Hill parameter and I is the identity tensor.

    """
    if kappa == 0.:
        return expm(np.matrix(strain))
    else:
        return powm(kappa * np.matrix(strain) + I3X3, 1. / kappa)
