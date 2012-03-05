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
from __future__ import print_function
import math
import numpy as np
import scipy.optimize
import sys

from Source.Payette_utils import *
from Source.Payette_tensor import *

def newton(cmod,dt,d,Fold,EF,P,Ppres,sv,v,*args,**kwargs):
    '''
    NAME
       newton

    PURPOSE
       Seek to determine the unknown components of the symmetric part of velocity
       gradient d[v] satisfying

                               P(d[v]) = Ppres

       where P is the current stress, d the symmetric part of the velocity
       gradient, v is a vector subscript array containing the components for
       which stresses (or stress rates) are prescribed, and Ppres are the
       prescribed values at the current time.

    INPUT
       cmod:   constiutive model instance
       d:      symmetric part of the velocity gradient
       dt:     timestep
       P:      current stress
       Ppres:  prescribed values of stress
       sv:     state variables
       v:      vector subscript array containing the components for which
               stresses are prescribed
       args:   not used
       kwargs: not used

    OUTPUT
       P:      current stress
       d:      symmetric part of the velocity gradient at the half step
       sv:     current values of state variables
       converged: 0  did not converge
                  1  converged based on tol1 (more stringent)
                  2  converged based on tol2 (less stringent)

    THEORY:
       The approach is an iterative scheme employing a multidimensional Newton's
       method. Each iteration begins with a call to subroutine jacobian, which
       numerically computes the Jacobian submatrix

                                  Js = J[v,v]

       where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of d[v]
       is then updated according to

                            d[v] = d[v] - Jsi*Perr(d[v])/dt

       where

                             Perr(d[v]) = P(d[v]) - Ppres

       The process is repeated until a convergence critierion is satisfied. The
       argument converged is a flag indicat- ing whether or not the procedure
       converged:

    HISTORY
       This is a python implementation of the fortran routine of the same name in
       Tom Pucik's MMD driver.

    AUTHORS
       Tom Pucick, original fortran coding
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
                   python implementation
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
                   replaced computation of inverse of the Jacobian with the more
                   computationally robust and faster iterative solver
    '''

    # --- Local variables
    nv,nsv,nsym = len(v), len(sv), len(P)
    Perr = np.zeros(nv)
    tol1, tol2 = epsilon(), math.sqrt(epsilon())
    maxit1, maxit2, depsmax, converged = 20, 30, 0.2, 0
    F0 = np.array([ [ Fold[0], Fold[3], Fold[5] ],
                    [ Fold[6], Fold[1], Fold[4] ],
                    [ Fold[8], Fold[7], Fold[2] ] ],dtype='double')

    # --- Initialize
    if (depsmag(d,dt) > depsmax):
        return d,converged

    Psave, svsave = np.copy(P), np.copy(sv)
    Fnew = toArray( toMatrix(Fold) + np.dot(toMatrix(d),toMatrix(Fold)),
                    symmetric=False )*dt
    arg_us = (dt,np.array(d),np.array(Fold),np.array(Fnew),np.array(EF),
              np.array(P),np.array(sv))
    P,sv = cmod.updateState(*arg_us)[0:2]
    Perr = P[v] - Ppres

    # --- Perform Newton iteration

    for i in range(maxit2):
        P, sv = np.array(Psave), np.array(svsave)
        argv = (dt,np.array(d),np.array(Fold),np.array(Fnew),np.array(EF),
                np.array(P),np.array(sv),np.array(v))
        Js = cmod.jacobian(*argv)
        try:
            d[v] -= np.linalg.solve(Js,Perr)/dt
        except:
            d[v] -= np.linalg.lstsq(Js,Perr)[0]/dt
            msg = 'Using least squares approximation to matrix inverse'
            reportWarning(__file__,msg,limit=True)
        #---------------------------------------------------------------
        if (depsmag(d,dt) > depsmax):
            # restore mutable passed values
            P,sv = np.array(Psave),np.array(svsave)
            return d,converged

        Fnew = toArray( toMatrix(Fold) + np.dot(toMatrix(d),toMatrix(Fold)),
                        symmetric=False )*dt
        arg_us = (dt,np.array(d),np.array(Fold),np.array(Fnew),np.array(EF),
                  np.array(P),np.array(sv))
        P,sv = cmod.updateState(*arg_us)[0:2]
        Perr = P[v] - Ppres
        dnom = np.amax(np.abs(Ppres)) if np.amax(np.abs(Ppres)) > 2.e-16 else 1.
        relerr = np.amax(np.abs(Perr))/dnom

        if i <= maxit1:
            if relerr < tol1:
                converged = 1
                # restore mutable passed values
                P,sv = np.array(Psave),np.array(svsave)
                return d,converged
            pass
        else:
            if relerr < tol2:
                converged = 2
                # restore mutable passed values
                P,sv = np.array(Psave),np.array(svsave)
                return d,converged
        continue

    # restore mutable passed values
    P,sv = np.array(Psave),np.array(svsave)
    return d,converged

def depsmag(d,dt):
    '''
    NAME
       depsmag

    PURPOSE
       return the L2 norm of the rate of the strain increment

    INPUT
       d:  symmetric part of the velocity gradient at the half-step
       dt: time step

    OUTPUT
       depsmag: L2 norm of the strain increment

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''
    return math.sqrt(sum(d[:3]**2) + 2.*sum(d[3:]**2))*dt

def simplex(cmod,dt,dEdt,Fold,EF,P,Ppres,sv,v,*args,**kwargs):
    '''
    NAME
       simplex

    PURPOSE
       Perform a downhill simplex search to find d[v] such that
                        P(d[v]) = Ppres[v]

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''
    proportional = kwargs["proportional"]
    nv,nsv,nsym = len(v), len(sv), len(P)
    Psave, svsave = np.copy(P), np.copy(sv)

    # --- Perform the simplex search
    args = (cmod,dt,np.array(dEdt),np.array(Fold),np.array(EF),
            np.array(P),np.array(Ppres),np.array(sv),np.array(v))
    if proportional:
        dEdt[v] = scipy.optimize.fmin(proportionality_func,dEdt[v],
                                      args=args,maxiter=10,disp=False)
    else:
        dEdt[v] = scipy.optimize.fmin(func,dEdt[v],args=args,maxiter=20,disp=False)

    # restore mutable passed values
    P,sv = np.array(Psave),np.array(svsave)

    return dEdt

def func(dEdt_opt,cmod,dt,dEdt,Fold,EF,P,Ppres,sv,v):
    '''
    NAME
       func

    PURPOSE
       Objective function to minimize during simplex search

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''
    nsym,nsv = len(P), len(sv)

    # intialize
    Psave,svsave = np.copy(P), np.copy(sv)
    F0 = np.array([ [ Fold[0], Fold[3], Fold[5] ],
                    [ Fold[6], Fold[1], Fold[4] ],
                    [ Fold[8], Fold[7], Fold[2] ] ],dtype='double')

    dEdt[v] = dEdt_opt
    Fnew = toArray( toMatrix(Fold) + np.dot(toMatrix(dEdt),toMatrix(Fold)),
                    symmetric=False )*dt
    arg_us = (dt,np.array(dEdt),np.array(Fold),np.array(Fnew),np.array(EF),
              np.array(P),np.array(sv))
    P,sv = cmod.updateState(*arg_us)[0:2]
    j, error = 0, 0.
    for i in v:
        error += (P[i] - Ppres[j])**2
        j += 1
        continue

    # restore mutable passed values
    P,sv = np.array(Psave),np.array(svsave)

    return error

def proportionality_func(dEdt_opt,cmod,dt,dEdt,Fold,EF,P,Ppres,sv,v):
    '''
    NAME
       proportionality_func

    PURPOSE
       Objective function to minimize during simplex search. Stress
       states are a scalar multiple of the prescribed stres..

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''
    nsym,nsv = len(P), len(sv)

    # intialize
    Psave,svsave = np.copy(P), np.copy(sv)
    F0 = np.array([ [ Fold[0], Fold[3], Fold[5] ],
                    [ Fold[6], Fold[1], Fold[4] ],
                    [ Fold[8], Fold[7], Fold[2] ] ],dtype='double')

    dEdt[v] = dEdt_opt
    Fnew = toArray( toMatrix(Fold) + np.dot(toMatrix(dEdt),toMatrix(Fold)),
                    symmetric=False )*dt
    arg_us = (dt,np.array(dEdt),np.array(Fold),np.array(Fnew),np.array(EF),
              np.array(P),np.array(sv))
    P,sv = cmod.updateState(*arg_us)[0:2]
    j, error = 0, 0.

    Pv = []
    Pu = []
    for i in v:
        Pv.append(P[i])
        Pu.append(Ppres[j])
        j += 1
        continue
#    print(Pu,Pv)
    Pv = np.array(Pv)
    Pu = np.array(Pu)


    Pu_norm = np.linalg.norm(Pu)
    if Pu_norm != 0.0:
        dum = np.dot(Pu/Pu_norm,Pv)*Pu/Pu_norm
        error = np.linalg.norm(dum) + np.linalg.norm(Pv - dum)**2
    else:
        error = np.linalg.norm(Pv)
#    print(error)


    # restore mutable passed values
    P,sv = np.array(Psave),np.array(svsave)

    return error

