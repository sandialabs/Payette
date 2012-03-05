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

import os
import sys
import pickle
import numpy as np
import scipy
from scipy import optimize

import Source.Payette_iterative_solvers as citer
import Source.Payette_kinematics as pkin
from Source.Payette_utils import *

def runProblem(the_model,restart):
    '''
    NAME
       runProblem:

    PURPOSE
       Run the single element simulation for an instance of the main Payette class
       the_model

    INPUT
       the_model: Payette container class instance

    OUTPUT
       outfile: the_model.getOutputFile()

    PASSED AND LOCAL VARIABLES
       Variabls marked with a * are local copies of the_model data.

             variable  type      size    description
             --------  --------  ----    --------------------------------------
             cmod*     instance  1       constitutive model object
             d         array     (2,6)   symmetic part of velocity gradient
                                           d[0,:] at time n
                                           d[1,:] (to be) at time n + 1
             dflg      list      (?)     list containing unique values of
                                         prescribed [deformation,stress] types
                                         on each leg.
             dt        scalar    1       current timestep size
             E         array     (2,6)   strain
                                           E[0,:] at time n
                                           E[1,:] (to be) at time n + 1
             dEdt      array     (6)     rate of strain
             Epres     array     (3,6)   prescribed strain
                                           Epres[0,:] at tleg[0]
                                           Epres[1,:] at tleg[1]
                                           Epres[2,:] at t (interpolated)
             F         array     (2,9)   deformation gradient
                                           F[0,:] at time n
                                           F[1,:] (to be) at time n + 1
             dFdt      array     (9)     rate of deformation gradient
             Fpres     array     (3,9)   prescribed deformation gradient
                                           Fpres[0,:] at tleg[0]
                                           Fpres[1,:] at tleg[1]
                                           Fpres[2,:] at t (interpolated)
             EF        array     (2,3)   electric field
                                           EF[0,:] at time n
                                           EF[1,:] (to be) at time n + 1
             EFpres    array     (3,3)   prescribed electric field
                                           EFpres[0,:] at tleg[0]
                                           EFpres[1,:] at tleg[1]
                                           EFpres[2,:] at t (interpolated)
             J0        array     (6,6)   Initial (full) Jacobian matrix
             k*        scalar    1       Seth-Hill parameter
             leg*      list      var     input deformation leg.  size varies by
                                         prescribed [deformation,stress] type.
             P         array     (2,6)   stress conjugate to E
             ED        array     (2,3)   electric displacement
             dPdt      array     (6)     rate of stress
             Pdum      array     (2,6)   dummy holder for prescribed stress
                                           Pdum[0,:] at tleg[0]
                                           Pdum[1,:] at tleg[1]
             Ppres     array     (3,?)   array containing only those components
                                         of Pdum for which stresses were actually
                                         prescribed
                                           Ppres[0,:] at tleg[0]
                                           Ppres[1,:] at tleg[1]
                                           Ppres[2,:] at t (interpolated)
             nsv*      scalar    var     number of state variables
             R         array     (2,9)   rotation (=I for now)
                                           R[0,:] at time n
                                           R[1,:] (to be) at time n + 1
             dRdt      array     (9)     rate of rotation
             Rpres     array     (3,9)   prescribed rotation (not used)
                                           Rpres[0,:] at tleg[0]
                                           Rpres[1,:] at tleg[1]
                                           Rpres[2,:] at t (interpolated)
             sv*       array     (2,nsv) state variables
                                           sv[0,:] at time n
                                           sv[1,:] (to be) at time n + 1
             t         scalar    1       current time
             tleg*     array     (2)     let time
                                           t[0]: time at beginning of leg
                                           t[1]: time at end of leg
             v         array     var     vector subscript array containing the
                                         components for which stresses (or stress
                                         rates) are prescribed
             w         array     (2,6)   skew part of velocity gradient
                                           w[0,:] at time n
                                           w[1,:] (to be) at time n + 1
    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''
    if debug: pdb = __import__('pdb')

    # -------------------- initialize and copy passed values to local variables

    # --- passed variables
    verbose = the_model.verbosity() > 0
    diagonal = the_model.diagonal()
    strict = the_model.strict()
    sparseout = the_model.emit() == 'sparse'
    screenout = the_model.screenout()
    nprints = the_model.nprints()
    proportional = the_model.proportional()
    k = the_model.kappa()

    # constitutive model stuff
    cmod = the_model.constitutiveModel()
    multilevelfail = cmod.multi_level_fail_model
    has_efield = cmod.electric_field_model
    sqa = the_model.sqa()
    write_vandd_table = the_model.write_vandd_table
    sv = np.array((cmod.internalStateVariables(),
                   cmod.internalStateVariables()),
                  dtype='double')
    updateState = cmod.updateState
    initialJacobian = cmod.initialJacobian

    (ileg,legs,tleg,t,dt,E,Epres,dEdt,P,Ppres,dPdt,Pdum,vdum,
     R,Rpres,dRdt,F,Fpres,dFdt,d,w,EF,EFpres,ED,POL,
     failed) = the_model.getPayetteState()

    # ----------------------------------------------------------------------- #

    # -------------------------------------------------------- initialize model
    # --- start output file and Mathematica files
    msg = "starting calculations for simulation %s"%the_model.simname
    reportMessage(__file__,msg)

    setupOutputFile(the_model,restart) #.outfile,restart,cmod.isvKeys(),
#                    has_efield,write_vandd_table)

    # --- call the material model with zero state
    if ileg == 0:
        arg_us = ( 1., np.array(d[0,:]),np.array(F[0,:]),np.array(F[0,:]),
                   np.array(EF[0,:]),np.array(P[0,:]),np.array(sv[0,:]) )
        updatedState = updateState(*arg_us)
        if has_efield: P[0,:],sv[0,:],POL[0,:] = updatedState
        else: P[0,:],sv[0,:] = updatedState

        pass

    # write the mathematica files
    if the_model.plotable:
        writeMathPlot(the_model.math1,the_model.math2,the_model.outfile,
                      the_model.plotable,has_efield,the_model.paramkeys,
                      cmod.ui0,cmod.ui,t,P[0,:],dPdt,E[0,:],d[0,:],F[0,:],
                      sv[0,:],cmod.keya,cmod.namea,EF[0,:],POL[0,:],ED[0,:])
        pass

    # --- initial jacobian matrix
    v = np.array(range(6),dtype=int)
    J0 = initialJacobian()

    kwargs = { "time":t,
               "stress":P[0,:],
               "time rate of stress":dPdt,
               "strain":E[0,:],
               "time rate of strain":d[0,:],
               "deformation gradient":F[0,:],
               "extra variables":sv[0,:],
               "electric field":EF[0,:],
               "electric displacement":ED[0,:],
               "polarization":POL[0,:]}
    the_model.updateMaterialData(**kwargs)
    writeState(the_model)
    # ----------------------------------------------------------------------- #

    # --------------------------------------------------- begin{processing leg}
    for leg in legs:

        if the_model.test_restart and not restart:
            if ileg == int(len(legs)/2):
                print('\n\nStopping to test Payette restart capabilities.\n'
                      'Restart the simulation by executing\n\n'
                      '\t\trunPayette %s.prf\n\n'
                      %os.path.splitext(os.path.basename(the_model.outfile))[0])
                sys.exit(76)

        # read inputs and initialize for this leg
        lnum,tleg[1],nsteps,ltype,lcntrl = leg
        delt = tleg[1] - tleg[0]
        if delt == 0.: continue
        nv, dflg = 0, list(set(ltype))
        nprints = the_model.nprints() if the_model.nprints() else nsteps
        if sparseout: nprints = min(10,nsteps)
        print_interval = max(1,int(nsteps/nprints))

        if verbose:
            msg = 'leg %i, step %i, time %f, dt %f'%(lnum,1,t,dt)
            reportMessage(__file__,msg)
            pass

        # --- loop through components of lcntrl and compute:
        #       for ltype = 1,2: E at tleg[1]
        #       for ltype = 3,4: P at tleg[1]
        #       for ltype = 5:   F at tleg[1]
        #       for ltype = 6,7: EF at tleg[1]
        for i in range(len(lcntrl)):
            if ltype[i] == 0: continue
            elif i < 6 and ltype[i] == 1:                  # strain rate
                Epres[1,i] = Epres[0,i] + lcntrl[i]*delt
            elif i < 6 and ltype[i] == 2:      # strain
                Epres[1,i] = lcntrl[i]
            elif i < 6 and ltype[i] == 3:                # stress rate
                Pdum[1,i] = Pdum[0,i] + lcntrl[i]*delt
                vdum[nv] = i
                nv += 1
            elif i < 6 and ltype[i] == 4:                # stress
                Pdum[1,i] = lcntrl[i]
                vdum[nv] = i
                nv += 1
            elif ltype[i] == 5:                # deformation gradient
                Fpres[1,i] = lcntrl[i]
            elif i >= 9 and ltype[i] == 6:                # electric field rate
                EFpres[1,i-9] = EFpres[0,i-9] + lcntrl[i]*delt
            elif i >= 9 and ltype[i] == 7:                # electric field
                EFpres[1,i-9] = lcntrl[i]
            else:
                msg = ('\nERROR: Invalid load type (ltype) parameter\n'
                       'prescribed for leg %i\n'%lnum)
                reportError(__file__,msg)
                return 1
            continue

        v = vdum[0:nv]
        if len(v):
            Ppres = np.zeros((3,nv))
            Ppres[0,:],Ppres[1,:] = Pdum[0,v],Pdum[1,v]
            Js = J0[[[x] for x in v],v]
            pass

        t = tleg[0]
        dt = delt/nsteps

        # ---------------------------------------------- begin{processing step}
        for n in range(nsteps):

            # interpolate values of E, F, EF, and P for the current step
            a1 = float(nsteps - (n + 1))/nsteps
            a2 = float(n + 1)/nsteps

            Epres[2,:] = a1*Epres[0,:] + a2*Epres[1,:]
            dEdt = (Epres[2,:] - E[0,:])/dt

            Fpres[2,:] = a1*Fpres[0,:] + a2*Fpres[1,:]
            dFdt = (Fpres[2,:] - F[0,:])/dt

            EFpres[2,:] = a1*EFpres[0,:] + a2*EFpres[1,:]

            # --- initial guess for dEdt[v]
            Ppres[2,:] = a1*Ppres[0,:] + a2*Ppres[1,:]
            if len(v):
                Perr = Ppres[2,:] - P[0,v]
                try: dEdt[v] = np.linalg.solve(Js,Perr)/dt
                except: dEdt[v] -= np.linalg.lstsq(Js,Perr)[0]/dt

            # --- update electric field to the end of the step
            EF[1,:] = EFpres[2,:]

            # --- find d (symmetric part of velocity gradient)
            if not len(v):
                if dflg[0] == 5:
                    # --- deformation gradient prescribed
                    argv = [np.array(Fpres[2,:]),np.array(dFdt),strict]
                    d[1,:],w[1,:] = pkin.velGradCompFromF(*argv)
                else:
                    # --- strain or strain rate prescribed
                    argv = [dt,k,np.array(E[0,:]),np.array(Epres[2,:]),
                            np.array(dEdt),np.array(R[1,:]),np.array(dRdt),strict]
                    d[1,:],w[1,:] = pkin.velGradCompFromE(*argv)
            else:
                # --- One or more stresses prescribed
                argv = [cmod,dt,np.array(dEdt),np.array(F[0,:]),np.array(EF[1,:]),
                        np.array(P[0,:]),np.array(Ppres[2,:]),np.array(sv[1,:]),
                        np.array(v),strict]
                d[1,:],w[1,:] = pkin.velGradCompFromP(*argv,proportional=proportional)
                pass

            # --- update the deformation to the end of the step
            args = (k,dt,np.array(d[1,:]),np.array(w[1,:]),np.array(F[0,:]),strict)
            E[1,:],F[1,:] = pkin.updateDeformation(*args)
            if the_model.useTableVals():
                if dflg == [1] or dflg == [2] or dflg == [1,2]: E[1,:] = Epres[2,:]
                elif dflg[0] == 5: F[1,:] = Fpres[2,:]
                pass

            # update stress, internal state, and electric displacment if appl.
            arg_us = ( dt,np.array(d[1,:]),np.array(F[0,:]),np.array(F[1,:]),
                       np.array(EF[1,:]),np.array(P[0,:]),np.array(sv[0,:]) )
            updatedState = updateState(*arg_us)
            if has_efield: P[1,:],sv[1,:],POL[1,:] = updatedState
            else: P[1,:],sv[1,:] = updatedState
            dPdt = (P[1,:] - P[0,:])/dt

            # multilevel failure. since there is only one element, it really
            # isn't multilevel
            if multilevelfail:
                if sv[1,cmod.cflg_idx] == 1:
                    reportMessage(__file__,"multilevel failure beginning")
                    reportMessage(__file__,"Failure ratio is %12.5e"
                                  %sv[1,cmod.fratio_idx])
                    sv[1,cmod.cflg_idx] = 3
                elif sv[1,cmod.cflg_idx] == 4: failed = True
                pass

            # --- pass values from end of this step to beginning of next
            for x in [P,E,F,EF,d,sv,ED,POL,w]: x[0,:] = x[1,:]
            t += dt

            kwargs = { "time":t,
                       "stress":P[0,:],
                       "time rate of stress":dPdt,
                       "strain":E[0,:],
                       "time rate of strain":d[0,:],
                       "deformation gradient":F[0,:],
                       "extra variables":sv[0,:],
                       "electric field":EF[0,:],
                       "electric displacement":ED[0,:],
                       "polarization":POL[0,:]}
            the_model.updateMaterialData(**kwargs)

            # --- write state to file
            if (nsteps-n)%print_interval == 0:
                writeState(the_model)
                pass

            if screenout or ( verbose and (2*n - nsteps) == 0 ):
                msg = 'leg %i, step %i, time %f, dt %f'%(lnum,n,t,dt)
                reportMessage(__file__,msg)
                pass

            if failed:
                msg = 'material failed on leg %i, step %i, at time %f'%(lnum,n,t)
                reportMessage(__file__,msg)
                return 0

            # ------------------------------------------ begin{end of step SQA}
            if sqa:
                if dflg == [1] or dflg == [2] or dflg == [1,2]:
                    max_diff = np.max(np.abs(E[1,:] - Epres[2,:]))
                    dnom = max(np.max(np.abs(Epres[2,:])),0.)
                    if dnom == 0.: dnom = 1.
                    rel_diff = max_diff/dnom
                    if rel_diff > accuracyLim() and max_diff > epsilon():
                        msg = ('E differs from lcntrl excessively at end of step '
                               '%i of leg %i with a percent difference of %f'
                               %(n,lnum,rel_diff*100.))
                        reportWarning(__file__,msg)
                    pass
                elif dflg == [5]:
                    max_diff = (np.max(np.abs(F[1,:] - Fpres[2,:]))/
                                np.max(np.abs(Fpres[2,:])))
                    dnom = max(np.max(np.abs(Fpres[2,:])),0.)
                    if dnom == 0.: dnom = 1.
                    rel_diff = max_diff/dnom
                    if rel_diff > accuracyLim() and max_diff > epsilon():
                        msg = ('F differs from lcntrl excessively at end of step '
                               ' with max_diff %f' %max_diff)
                        reportWarning(__file__,msg)
                    pass
                pass
            # -------------------------------------------- end{end of step SQA}

            continue
        # ------------------------------------------------ end{processing step}

        # ----------------------------------------------- begin{end of leg SQA}
        if sqa:
            if dflg == [1] or dflg == [2] or dflg == [1,2]:
                max_diff = np.max(np.abs(E[1,:] - Epres[1,:]))
                dnom = np.max(Epres[2,:])
                if dnom <= epsilon(): dnom = 1.
                rel_diff = max_diff/dnom
                if rel_diff > accuracyLim() and max_diff > epsilon():
                    msg = ('E differs from lcntrl excessively at end of '
                           'leg %i with relative diff %f'%(lnum,rel_diff))
                    reportWarning(__file__,msg)
                pass

            elif dflg == [5]:
                max_diff = (np.max(np.abs(F[1,:] - Fpres[1,:]))/
                            np.max(np.abs(Fpres[1,:])))
                dnom = np.max(Fpres[2,:])
                if dnom <= epsilon(): dnom = 1.
                rel_diff = max_diff/dnom
                if rel_diff > accuracyLim() and max_diff > epsilon():
                    msg = ('F differs from lcntrl excessively at end of leg '
                           ' with max_diff %f' %max_diff)
                    reportWarning(__file__,msg)
                pass
            pass
        # ------------------------------------------------- end{end of leg SQA}

        # --- pass quantities from end of leg to beginning of new leg
        if write_vandd_table:
            writeVelAndDispTable(the_model.initialTime(),
                                 the_model.terminationTime(),
                                 tleg[0],tleg[1],Epres[0,:],E[1,:],k)
            pass
        tleg[0] = tleg[1]
        Pdum[0,:],Epres[0,:],Fpres[0,:],EFpres[0,:] = P[1,:],E[1,:],F[1,:],EF[1,:]
        if len(v):
            j = 0
            for i in v:
                Pdum[0,i] = Ppres[1,j]
                j += 1
                continue
            pass

        # save the state
        ileg += 1
        the_model.savePayetteState(ileg,tleg,t,dt,E,Epres,dEdt,P,Ppres,dPdt,Pdum,
                                   vdum,R,Rpres,dRdt,F,Fpres,dFdt,d,w,
                                   EF,EFpres,ED,POL,failed)

        if the_model.write_restart:
            with open(the_model.rfile,'wb') as f: pickle.dump(the_model,f,2)
            pass

        # --- print message to screen
        if verbose:
            msg = 'leg %i, step %i, time %f, dt %f'%(lnum,n+1,t,dt)
            reportMessage(__file__,msg)
            pass

        continue
    # ----------------------------------------------------- end{processing leg}

    return 0
