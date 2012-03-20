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

import math
import numpy as np
import numpy.linalg as la
import scipy
from scipy import linalg

import Source.Payette_iterative_solvers as piter
from Source.Payette_utils import *
from Source.Payette_tensor import *

# global identity matrix
I = np.eye(3)

def velGradCompFromE(simdat):
    '''
    NAME
       velGradCompFromE

    PURPOSE: from the value of the Seth-Hill parameter kappa, current strain
                E, and strain rate dEdt, return the symmetric and skew parts of the
                velocity gradient

    INPUT:
       simdat: simulation data container

    OUTPUT:
       d:     symmetric part of the velocity gradient
       w:     skew part of the velocity gradient

    THEORY:
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

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''
    # convert passed arrays to matrices
    k = simdat.KAPPA
    dt = simdat.getData("time step")
    E0 = simdat.getData("strain",form="Matrix")
    Ef = simdat.getData("prescribed strain",form="Matrix")
    dEdt = (Ef - E0)/dt
    R = simdat.getData("rotation",form="Matrix")
    dRdt = simdat.getData("rotation rate",form="Matrix")
    Ri = R.T

    # stretch and its rate
    U = rightStretch(k,Ef,simdat.STRICT)
    Ui = la.inv(U)
    X = 0.5*(la.inv(k*Ef + I) + la.inv(k*E0 + I)) # center X on half step
    dUdt = U*X*dEdt

    # velocity gradient, sym, and skew parts
    L = dRdt*Ri + R*dUdt*Ui*Ri
    d = 0.5*( L + L.T )
    w = L - d

    simdat.storeData("rate of deformation",d)
    simdat.storeData("vorticity",w)
    return

def velGradCompFromF(simdat):
    '''
    NAME
       velGradCompFromF

    PURPOSE
       from the value of the deformation gradient F, return the symmetric and
       skew parts of the velocity gradient

    INPUT:
       simdat: simulation data container

    OUTPUT:
       d:     symmetric part of velocity gradient
       w:     skew part of velocity gradient

    THEORY:
       Velocity gradient L is given by

                 L = dFdt * Finv

       Then

                 d = sym(L)
                 w = skew(L)

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    dt = simdat.getData("time step")
    F_beg = simdat.getData("deformation gradient",form="Matrix")
    F_end = simdat.getData("prescribed deformation gradient",form="Matrix")
    dFdt = (F_end - F_beg)/dt

    L = 0.5*dFdt*(la.inv(F_end) + la.inv(F_beg))
    d = 0.5*( L + L.T )
    w = L - d

    simdat.storeData("vorticity",w)
    simdat.storeData("rate of deformation",d)
    return

def velGradCompFromP(material,simdat,matdat):
    '''
    NAME
       velGradCompFromP

    PURPOSE
       Seek to determine the unknown components of the symmetric part of velocity
       gradient d[v] satisfying

                               P(d[v]) = Ppres[:]                      (1)

       where P is the current stress, d the symmetric part of the velocity
       gradient, v is a vector subscript array containing the components for
       which stresses (or stress rates) are prescribed, and Ppres[:] are the
       prescribed values at the current time.

    INPUT
       material:   constiutive model instance
       dEdt:   symmetric part of the velocity gradient
       dt:     timestep
       P:      current stress
       Ppres:  prescribed values of stress
       sv:     state variables
       v:      vector subscript array containing the components for which
               stresses are prescribed
       args:   not used
       kwargs: not used

    OUTPUT
       d:    symmetric part of velocity gradient at the half step
       w:    skew part of velocity gradient at the half step

    APPROACH
       Solution is found iteratively in (up to) 3 steps
          1) Call newton to solve 1, return P,sv,d if converged
          2) Call newton with d[v] = 0. to solve 1, return P,sv,d if converged
          3) Call simplex with d[v] = 0. to solve 1, return P,sv,d

    HISTORY
       This is a python implementation of a fortran subroutine of the same name
       written by Tom Pucick for his MMD material model driver.

    AUTHORS
       Tom Pucick, original fortran coding
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
                   python implementation
    '''

    depsdt_old = simdat.getData("strain rate")
    v = simdat.getData("prescribed stress components")
    nv = len(v)

    if not simdat.PROPORTIONAL:

        converged = piter.newton(material,simdat,matdat)

        if converged:
            return

        # --- didn't converge, try Newton's method with initial d[v]=0.
        dEdt = simdat.getData("strain rate")
        dEdt[v] = np.zeros(nv)
        simdat.storeData("strain rate",dEdt,old=True)

        converged = piter.newton(material,simdat,matdat)

        if converged:
            return

        pass

    # --- Still didn't converge. Try downhill simplex method and accept whatever
    #     answer it returns:
    simdat.restoreData("strain rate",depsdt_old)
    piter.simplex(material,simdat,matdat)
    return

def updateDeformation(simdat):
    '''
    NAME
       updateDeformation

    PURPOSE
       from the value of the Seth-Hill parameter kappa, current strain E,
       deformation gradient F, and symmetric part of the velocit gradient d,
       update the strain and deformation gradient.

    INPUT
       simdat: simulation data container

    OUTPUT
       E:    strain at end of step
       F:    deformation gradient at end of step

    THEORY
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

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''
    iam = "updateDeformation(simdat,matdat)"

    k = simdat.KAPPA
    dt = simdat.getData("time step")
    F0 = simdat.getData("deformation gradient",form="Matrix")
    d = simdat.getData("rate of deformation",form="Matrix")
    w = simdat.getData("vorticity",form="Matrix")

    Ff = expm((d + w)*dt,simdat.STRICT)*F0
    U = sqrtm((Ff.T)*Ff,simdat.STRICT)
    if k == 0: E = logm(U,simdat.STRICT)
    else: E = 1./k*(powm(U,k,simdat.STRICT) - I)

    if np.linalg.det(Ff) <= 0.:
        reportError(iam,"negative Jacobian encountered")

    simdat.storeData("strain",E)
    simdat.storeData("deformation gradient",Ff)

    # compute the equivalent strain
    eps = toArray(E,symmetric=True)
    eqveps = np.sqrt( 2./3.*( sum(eps[:3]**2) + 2.*sum(eps[3:]**2)) )
    simdat.storeData("equivalent strain",eqveps)

    return

def rightStretch(k,E,strict=False):
    '''
    NAME
       rightStretch

    PURPOSE

       Compute the symmtetric part U (right stretch) of the polar decomposition
       of the deformation gradient F = RU.

    INPUT
       k: Seth-Hill strain parameter
       E: current strain tensor

    OUTPUT
       U: the right stretch tensor

    THEORY
       The right stretch tensor U, in terms of the strain tensor E is given by

                     U = (k*E + I)**(1/k)

       where k is the Seth-Hill parameter and I is the identity tensor.

    AUTHORS
       Tim Fuller, Sandial National Laboratories, tjfulle@sandia.gov
    '''
    if k == 0.: return expm(np.matrix(E),strict)
    else: return powm(k*np.matrix(E) + I,1./k,strict)

def leftStretch(k,e,strict=False):
    '''
    NAME
       leftStretch

    PURPOSE

       Compute the symmtetric part V (left stretch) of the polar decomposition
       of the deformation gradient F = VR.

    INPUT
       k: Seth-Hill strain parameter
       e: current spatial strain tensor

    OUTPUT
       V: the left stretch tensor

    THEORY
       The left stretch tensor V, in terms of the strain tensor e is given by

                     V = (k*e + I)**(1/k)

       where k is the Seth-Hill parameter and I is the identity tensor.

    AUTHORS
       Tim Fuller, Sandial National Laboratories, tjfulle@sandia.gov
    '''
    if k == 0.: return expm(np.matrix(e),strict)
    else: return powm(k*np.matrix(e) + I,1./k,strict)

