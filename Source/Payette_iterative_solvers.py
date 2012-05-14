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
import scipy.optimize
import sys

import Source.Payette_utils as pu
import Source.Payette_tensor as pt

def newton(material, simdat, matdat):
    '''
    NAME
       newton

    PURPOSE
       Seek to determine the unknown components of the symmetric part of velocity
       gradient depsdt[v] satisfying

                               sig(depsdt[v]) = prsig

       where P is the current stress, depsdt the symmetric part of the velocity
       gradient, v is a vector subscript array containing the components for
       which stresses (or stress rates) are prescribed, and prsig are the
       prescribed values at the current time.

    INPUT
       material:   constiutive model instance
       simdat: simulation data container
               depsdt: strain rate
               dt: timestep
               sig: current stress
               prsig: prescribed values of stress
               v: vector subscript array containing the components for which
                  stresses are prescribed

    OUTPUT
       simdat: simulation data container
               depsdt: strain rate at the half step
       converged: 0  did not converge
                  1  converged based on tol1 (more stringent)
                  2  converged based on tol2 (less stringent)

    THEORY:
       The approach is an iterative scheme employing a multidimensional Newton's
       method. Each iteration begins with a call to subroutine jacobian, which
       numerically computes the Jacobian submatrix

                                  Js = J[v,v]

       where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of
       depsdt[v] is then updated according to

                            depsdt[v] = depsdt[v] - Jsi*sig_dif(depsdt[v])/dt

       where

                             sig_dif(depsdt[v]) = P(depsdt[v]) - prsig

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
# dt,depsdt,Fold,EF,P,prsig,sv,v,*args,**kwargs):

    # --- Local variables
    dt = simdat.get_data("time step")
    depsdt = matdat.get_data("strain rate")
    Fold = matdat.get_data("deformation gradient",form="Matrix")
    dold = matdat.get_data("rate of deformation")
    prsig = matdat.get_data("prescribed stress")
    v = matdat.get_data("prescribed stress components")
    sig = matdat.get_data("stress")
    sv = matdat.get_data("extra variables")

    nv = len(v)
    sig_dif = np.zeros(nv)
    tol1, tol2 = pu.EPSILON, math.sqrt(pu.EPSILON)
    maxit1, maxit2, depsmax, converged = 20, 30, 0.2, 0

    # --- Check if strain increment is too large
    if (depsmag(depsdt,dt) > depsmax):
        return converged

    # --- Initialize
    Fnew = Fold + np.dot(pt.to_matrix(depsdt),Fold)*dt

    # replace deformation rate and gradient with current best guesses
    matdat.store_data("deformation gradient",Fnew,old=True)
    matdat.store_data("rate of deformation",depsdt,old=True)

    # update the material state
    material.update_state(simdat, matdat)

    sig = matdat.get_data("stress", cur=True)
    sig_dif = sig[v] - prsig

    # --- Perform Newton iteration
    for i in range(maxit2):
        Js = material.jacobian(simdat, matdat)
        try:
            depsdt[v] -= np.linalg.solve(Js,sig_dif)/dt

        except:
            print Js
            print sig_dif
            print dt
            print v
            print prsig
            sys.exit("here")
            depsdt[v] -= np.linalg.lstsq(Js,sig_dif)[0]/dt
            msg = 'Using least squares approximation to matrix inverse'
            pu.reportWarning(__file__, msg, limit=True)
            pass

        if (depsmag(depsdt,dt) > depsmax):
            # increment too large, restore changed data and exit
            matdat.restore_data("rate of deformation",dold)
            matdat.restore_data("deformation gradient",Fold)
            return converged

        Fnew = Fold + np.dot(pt.to_matrix(depsdt),Fold)*dt
        matdat.store_data("rate of deformation",depsdt,old=True)
        matdat.store_data("deformation gradient",Fnew,old=True)
        material.update_state(simdat, matdat)
        sig = matdat.get_data("stress",cur=True)
        sig_dif = sig[v] - prsig
        dnom = np.amax(np.abs(prsig)) if np.amax(np.abs(prsig)) > 2.e-16 else 1.
        relerr = np.amax(np.abs(sig_dif))/dnom

        if i <= maxit1:
            if relerr < tol1:
                converged = 1
                matdat.restore_data("rate of deformation",dold)
                matdat.restore_data("deformation gradient",Fold)
                matdat.store_data("strain rate",depsdt)
                return converged
            pass

        else:
            if relerr < tol2:
                converged = 2
                # restore changed data and store the newly found strain rate
                matdat.restore_data("rate of deformation",dold)
                matdat.restore_data("deformation gradient",Fold)
                matdat.store_data("strain rate",depsdt)
                return converged

        continue

    # didn't converge, restore restore data and exit
    matdat.restore_data("rate of deformation",dold)
    matdat.restore_data("deformation gradient",Fold)
    return converged

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

def simplex(material,simdat,matdat):
    '''
    NAME
       simplex

    PURPOSE
       Perform a downhill simplex search to find d[v] such that
                        P(d[v]) = prsig[v]

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''
    v = matdat.get_data("prescribed stress components")
    depsdt = matdat.get_data("strain rate")

    nv = len(v)

    # --- Perform the simplex search
    args = (material,simdat,matdat)
    depsdt[v] = scipy.optimize.fmin(func,depsdt[v],args=args,maxiter=20,disp=False)

    matdat.store_data("strain rate",depsdt)

    return

def func(depsdt_opt,material,simdat,matdat):
    '''
    NAME
       func

    PURPOSE
       Objective function to minimize during simplex search

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''

    # initialize
    dt = simdat.get_data("time step")
    v = matdat.get_data("prescribed stress components")
    prsig = matdat.get_data("prescribed stress")
    depsdt = matdat.get_data("strain rate")
    depsdt[v] = depsdt_opt
    Fold = matdat.get_data("deformation gradient",form="Matrix")
    dold = matdat.get_data("rate of deformation")
    Fnew = Fold + np.dot(pt.to_matrix(depsdt),Fold)*dt

    # store the best guesses
    matdat.store_data("rate of deformation",depsdt,old=True)
    matdat.store_data("deformation gradient",Fnew,old=True)
    material.update_state(simdat,matdat)

    sig = matdat.get_data("stress",cur=True)

    # check the error
    error = 0.
    if not simdat.PROPORTIONAL:
        for i, j in enumerate(v):
            error += (sig[j] - prsig[i])**2
            continue
        pass

    else:
        Pv, Pu = [], []
        for i, j in enumerate(v):
            Pu.append(prsig[i])
            Pv.append(sig[j])
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
            pass
        pass

    # restore data
    matdat.restore_data("rate of deformation",dold)
    matdat.restore_data("deformation gradient",Fold)

    return error

