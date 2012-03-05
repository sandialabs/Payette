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

def velGradCompFromE(dt,k,E0,Ef,dEdt,R,dRdt,strict,*args,**kwargs):
    '''
    NAME
       velGradCompFromE

    PURPOSE: from the value of the Seth-Hill parameter kappa, current strain
                E, and strain rate dEdt, return the symmetric and skew parts of the
                velocity gradient

    INPUT:
       dt:    timestep
       k:     Seth-Hill parameter
       E0:    strain at beginning of step
       Ef:    strain at end of step
       dEdt:  strain rate
       R:     rotation
       dRdt:  rate of rotation
       args:  not used
       kwargs:not used

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
    E0,Ef,dEdt = toMatrix(E0), toMatrix(Ef), toMatrix(dEdt)
    R,dRdt = toMatrix(R), toMatrix(dRdt)
    Ri = R.T

    # stretch and its rate
    U = rightStretch(k,Ef,strict)
    Ui = la.inv(U)
    X = 0.5*(la.inv(k*Ef + I) + la.inv(k*E0 + I)) # center X on half step
    dUdt = U*X*dEdt

    # velocity gradient, sym, and skew parts
    L = dRdt*Ri + R*dUdt*Ui*Ri
    d = 0.5*( L + L.T )
    w = L - d
    return toArray(d),toArray(w,symmetric=False)

def velGradCompFromF(F,dFdt,strict,*args,**kwargs):
    '''
    NAME
       velGradCompFromF

    PURPOSE
       from the value of the deformation gradient F, return the symmetric and
       skew parts of the velocity gradient

    INPUT:
       F:     deformation gradient
       dFdt:  deformation gradient rate
       args:  not used
       kwargs:not used

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

    F = toMatrix(F)
    Finv = la.inv(F)
    dFdt = toMatrix(dFdt)

    L = dFdt*Finv
    d = 0.5*( L + L.T )
    w = L - d
    return toArray(d),toArray(w,symmetric=False)

def velGradCompFromP(cmod,dt,dEdt,Fold,EF,P,Ppres,sv,v,strict,*args,**kwargs):
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
       cmod:   constiutive model instance
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
    proportional = kwargs["proportional"]
    nv,nsym = len(v),len(P)
    w = np.zeros(9)

    # save current values of dedt,P,sv
    dedtsave = np.array(dEdt)
    if not proportional:
        argv = [cmod,dt,np.array(dEdt),np.array(Fold),np.array(EF),
                np.array(P),np.array(Ppres),np.array(sv),np.array(v)]
        dEdt,converged = piter.newton(*argv)
        if converged: return dEdt,w

        # --- didn't converge, try Newton's method with initial d[v]=0.
        dEdt = np.array(dedtsave)
        dEdt[v] = np.zeros(nv)
        argv = [cmod,dt,np.array(dEdt),np.array(Fold),np.array(EF),
                np.array(P),np.array(Ppres),np.array(sv),np.array(v)]
        dEdt,converged = piter.newton(*argv)
        if converged: return dEdt,w

    # --- Still didn't converge. Try downhill simplex method and accept whatever
    #     answer it returns:
    dEdt = np.array(dedtsave)
    argv = [cmod,dt,np.array(dEdt),np.array(Fold),np.array(EF),
            np.array(P),np.array(Ppres),np.array(sv),np.array(v)]
    dEdt = piter.simplex(*argv,**kwargs)
    return dEdt,w

def updateDeformation(k,dt,d,w,F,strict):
    '''
    NAME
       updateDeformation

    PURPOSE
       from the value of the Seth-Hill parameter kappa, current strain E,
       deformation gradient F, and symmetric part of the velocit gradient d,
       update the strain and deformation gradient.

    INPUT
       k:    Seth-Hill parameter
       F:    deformation gradient at beginning of step
       d:    symmetric part of velocity gradient
       R:    rotation
       dRdt: rate of rotation

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
    F0,d,w = toMatrix(F), toMatrix(d), toMatrix(w)
    Ff = expm((d + w)*dt,strict)*F0
    U = sqrtm((Ff.T)*Ff,strict)
    if k == 0: E = logm(U,strict)
    else: E = 1./k*(powm(U,k,strict) - I)
    return toArray(E),toArray(Ff,symmetric=False)

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
    if k == 0.: return expm(E,strict)
    else: return powm(k*E + I,1./k,strict)

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
    if k == 0.: return expm(e,strict)
    else: return powm(k*e + I,1./k,strict)

