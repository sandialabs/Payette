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

"""Module containing utilities for Payette's iterative stress solver"""

import math
import numpy as np
import scipy.optimize
import sys

import Source.Payette_utils as pu
import Source.Payette_tensor as pt
import Source.__runopts__ as ro


EPSILON = np.finfo(np.float).eps


def newton(material, simdat, matdat, prsig, V):
    '''
    NAME
       newton

    PURPOSE
       Seek to determine the unknown components of the symmetric part of velocity
       gradient depsdt[V] satisfying

                               sig(depsdt[V]) = prsig

       where P is the current stress, depsdt the symmetric part of the velocity
       gradient, V is a vector subscript array containing the components for
       which stresses (or stress rates) are prescribed, and prsig are the
       prescribed values at the current time.

    INPUT
       material:   constiutive model instance
       simdat: simulation data container
               depsdt: strain rate
               delt: timestep
               sig: current stress
               prsig: prescribed values of stress
               V: vector subscript array containing the components for which
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

                                  Js = J[V, V]

       where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of
       depsdt[V] is then updated according to

                depsdt[V] = depsdt[V] - Jsi*sig_dif(depsdt[V])/delt

       where

                   sig_dif(depsdt[V]) = P(depsdt[V]) - prsig

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
    delt = simdat.get("time step")
    depsdt = matdat.get("strain rate")
    sig = matdat.get("stress")

    nV = len(V)
    sig_dif = np.zeros(nV)
    tol1, tol2 = EPSILON, math.sqrt(EPSILON)
    maxit1, maxit2, depsmax, converged = 20, 30, 0.2, 0

    # --- Check if strain increment is too large
    if (depsmag(depsdt, delt) > depsmax):
        return converged

    # --- Initialize
    defgrad_new = defgrad_old + np.dot(pt.to_matrix(depsdt), defgrad_old) * delt

    # replace deformation rate and gradient with current best guesses
    matdat.save("deformation gradient", defgrad_new, "-", stash=True)
    matdat.save("rate of deformation", depsdt, "-", stash=True)

    # update the material state
    material.update_state(simdat, matdat)

    sig = matdat.get("stress", "+")
    sig_dif = sig[V] - prsig

    # --- Perform Newton iteration
    for i in range(maxit2):
        jac_sub = material.jacobian(simdat, matdat)
        try:
            depsdt[V] -= np.linalg.solve(jac_sub, sig_dif) / delt

        except:
            depsdt[V] -= np.linalg.lstsq(jac_sub, sig_dif)[0] / delt
            msg = 'Using least squares approximation to matrix inverse'
            pu.log_warning(msg, limit=True)

        if (depsmag(depsdt, delt) > depsmax):
            # increment too large, restore changed data and exit
            matdat.unstash("rate of deformation")
            matdat.unstash("deformation gradient")
            return converged

        defgrad_new = (defgrad_old +
                       np.dot(pt.to_matrix(depsdt), defgrad_old) * delt)
        matdat.save("rate of deformation", depsdt, "-")
        matdat.save("deformation gradient", defgrad_new, "-")
        material.update_state(simdat, matdat)
        sig = matdat.get("stress", "+")
        sig_dif = sig[V] - prsig
        dnom = np.amax(np.abs(prsig)) if np.amax(np.abs(prsig)) > 2.e-16 else 1.
        relerr = np.amax(np.abs(sig_dif)) / dnom

        if i <= maxit1:
            if relerr < tol1:
                converged = 1
                matdat.unstash("rate of deformation")
                matdat.unstash("deformation gradient")
                matdat.save("strain rate", depsdt)
                return converged

        else:
            if relerr < tol2:
                converged = 2
                # restore changed data and store the newly found strain rate
                matdat.unstash("rate of deformation")
                matdat.unstash("deformation gradient")
                matdat.save("strain rate", depsdt)
                return converged

        continue

    # didn't converge, unstash data and exit
    matdat.unstash("rate of deformation")
    matdat.unstash("deformation gradient")
    return converged


def depsmag(sym_velgrad, delt):
    '''
    NAME
       depsmag

    PURPOSE
       return the L2 norm of the rate of the strain increment

    INPUT
       sym_velgrad:  symmetric part of the velocity gradient at the half-step
       delt: time step

    OUTPUT
       depsmag: L2 norm of the strain increment

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''
    return math.sqrt(sum(sym_velgrad[:3] ** 2) +
                     2. * sum(sym_velgrad[3:] ** 2)) * delt


def simplex(material, simdat, matdat, prsig, V):
    '''
    NAME
       simplex

    PURPOSE
       Perform a downhill simplex search to find sym_velgrad[V] such that
                        P(sym_velgrad[V]) = prsig[V]

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''
    depsdt = matdat.get("strain rate")

    # --- Perform the simplex search
    args = (material, simdat, matdat, prsig, V)
    depsdt[V] = scipy.optimize.fmin(func, depsdt[V],
                                    args=args, maxiter=20, disp=False)

    matdat.save("strain rate", depsdt)

    return


def func(depsdt_opt, material, simdat, matdat, prsig, V):
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
    delt = simdat.get("time step")
    depsdt = matdat.get("strain rate")
    depsdt[V] = depsdt_opt
    defgrad_new = defgrad_old + np.dot(pt.to_matrix(depsdt), defgrad_old) * delt

    # store the best guesses (and stash current value)
    matdat.save("rate of deformation", depsdt, "-", stash=True)
    matdat.save("deformation gradient", defgrad_new, "-", stash=True)
    material.update_state(simdat, matdat)

    sig = matdat.get("stress", "+")

    # check the error
    error = 0.
    if not ro.PROPORTIONAL:
        for i, j in enumerate(V):
            error += (sig[j] - prsig[i]) ** 2
            continue

    else:
        stress_v, stress_u = [], []
        for i, j in enumerate(V):
            stress_u.append(prsig[i])
            stress_v.append(sig[j])
            continue
    #    print(stress_u, stress_v)
        stress_v = np.array(stress_v)
        stress_u = np.array(stress_u)

        stress_u_norm = np.linalg.norm(stress_u)
        if stress_u_norm != 0.0:
            dum = (np.dot(stress_u / stress_u_norm, stress_v) *
                   stress_u / stress_u_norm)
            error = np.linalg.norm(dum) + np.linalg.norm(stress_v - dum) ** 2
        else:
            error = np.linalg.norm(stress_v)

    # restore data
    matdat.unstash("rate of deformation")
    matdat.unstash("deformation gradient")

    return error
