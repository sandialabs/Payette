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


def newton(material, simdat, matdat, dt, Pt, V, dEdt):
    '''
    NAME
       newton

    PURPOSE
       Seek to determine the unknown components of the symmetric part of velocity
       gradient dEdt[V] satisfying

                               P(dEdt[V]) = Pt

       where P is the current stress, dEdt the symmetric part of the velocity
       gradient, V is a vector subscript array containing the components for
       which stresses (or stress rates) are prescribed, and Pt are the
       prescribed values at the current time.

    INPUT
       material:   constiutive model instance
       simdat: simulation data container
               dEdt: strain rate
               dt: timestep
               P: current stress
               Pt: prescribed values of stress
               V: vector subscript array containing the components for which
                  stresses are prescribed

    OUTPUT
       simdat: simulation data container
               dEdt: strain rate at the half step
       converged: 0  did not converge
                  1  converged based on tol1 (more stringent)
                  2  converged based on tol2 (less stringent)

    THEORY:
       The approach is an iterative scheme employing a multidimensional Newton's
       method. Each iteration begins with a call to subroutine jacobian, which
       numerically computes the Jacobian submatrix

                                  Js = J[V, V]

       where J[:,;] is the full Jacobian matrix J = dsig/deps. The value of
       dEdt[V] is then updated according to

                dEdt[V] = dEdt[V] - Jsi*Pd(dEdt[V])/dt

       where

                   Pd(dEdt[V]) = P(dEdt[V]) - Pt

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
    F0 = matdat.get("deformation gradient", copy=True)
    D0 = matdat.get("rate of deformation", copy=True)

    nV = len(V)
    Pd = np.zeros(nV)
    tol1, tol2 = EPSILON, math.sqrt(EPSILON)
    maxit1, maxit2, depsmax, converged = 20, 30, 0.2, 0

    # --- Check if strain increment is too large
    if (depsmag(dEdt, dt) > depsmax):
        return converged, dEdt

    # --- Initialize
    Ff = F0 + pt.dot(dEdt, F0) * dt

    # replace deformation rate and gradient with current best guesses
    matdat.store("deformation gradient", Ff)
    matdat.store("rate of deformation", dEdt)

    # update the material state to get the first guess at the new stress
    material.update_state(simdat, matdat)
    P = matdat.get("stress")
    Pd = P[V] - Pt
    matdat.restore()

    # --- Perform Newton iteration
    for i in range(maxit2):
        Js = material.jacobian(simdat, matdat, V)
        try:
            dEdt[V] -= np.linalg.solve(Js, Pd) / dt

        except:
            dEdt[V] -= np.linalg.lstsq(Js, Pd)[0] / dt
            msg = 'Using least squares approximation to matrix inverse'
            pu.log_warning(msg, limit=True)

        if (depsmag(dEdt, dt) > depsmax):
            # increment too large, restore changed data and exit
            matdat.restore("rate of deformation")
            matdat.restore("deformation gradient")
            return converged, D0

        Ff = F0 + pt.dot(dEdt, F0) * dt
        matdat.store("rate of deformation", dEdt)
        matdat.store("deformation gradient", Ff)
        material.update_state(simdat, matdat)
        P = matdat.get("stress")
        Pd = P[V] - Pt
        dnom = np.amax(np.abs(Pt)) if np.amax(np.abs(Pt)) > 2.e-16 else 1.
        relerr = np.amax(np.abs(Pd)) / dnom
        matdat.restore()

        if i <= maxit1:
            if relerr < tol1:
                converged = 1
                matdat.restore("rate of deformation")
                matdat.restore("deformation gradient")
                return converged, dEdt

        else:
            if relerr < tol2:
                converged = 2
                # restore changed data and store the newly found strain rate
                matdat.restore("rate of deformation")
                matdat.restore("deformation gradient")
                return converged, dEdt

        continue

    # didn't converge, restore restore data and exit
    matdat.restore("rate of deformation")
    matdat.restore("deformation gradient")
    return converged, dEdt


def depsmag(sym_velgrad, dt):
    '''
    NAME
       depsmag

    PURPOSE
       return the L2 norm of the rate of the strain increment

    INPUT
       sym_velgrad:  symmetric part of the velocity gradient at the half-step
       dt: time step

    OUTPUT
       depsmag: L2 norm of the strain increment

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''
    return math.sqrt(sum(sym_velgrad[:3] ** 2) +
                     2. * sum(sym_velgrad[3:] ** 2)) * dt


def simplex(material, simdat, matdat, dt, dEdt, Pt, V):
    '''
    NAME
       simplex

    PURPOSE
       Perform a downhill simplex search to find sym_velgrad[V] such that
                        P(sym_velgrad[V]) = Pt[V]

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
       M Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''
    # --- Perform the simplex search
    args = (material, simdat, matdat, dt, dEdt.copy(), Pt, V)
    dEdt[V] = scipy.optimize.fmin(func, dEdt[V],
                                  args=args, maxiter=20, disp=False)
    return dEdt


def func(x, material, simdat, matdat, dt, dEdt, Pt, V):
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
    dEdt[V] = x
    F0 = matdat.get("deformation gradient", copy=True)
    D0 = matdat.get("rate of deformation", copy=True)
    Ff = F0 + pt.dot(dEdt, F0) * dt

    # store the best guesses
    matdat.store("rate of deformation", dEdt)
    matdat.store("deformation gradient", Ff)
    material.update_state(simdat, matdat)
    P = matdat.get("stress", copy=True)
    matdat.restore()

    # check the error
    error = 0.
    if not ro.PROPORTIONAL:
        for i, j in enumerate(V):
            error += (P[j] - Pt[i]) ** 2
            continue

    else:
        stress_v, stress_u = [], []
        for i, j in enumerate(V):
            stress_u.append(Pt[i])
            stress_v.append(P[j])
            continue
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
    matdat.restore("rate of deformation")
    matdat.restore("deformation gradient")

    return error
