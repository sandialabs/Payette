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

import Source.Payette_iterative_solvers as citer
import Source.Payette_kinematics as pkin
from Source.Payette_utils import *
from Source.Payette_tensor import *

iam = "Payette_driver.runProplem(the_model,restart)"

np.set_printoptions(precision=2)

def runProblem(the_model,**kwargs):
    """
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
    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """
    if debug: pdb = __import__('pdb')
    cons_msg = "leg {0:{1}d}, step {2:{3}d}, time {4:.4E}, dt {5:.4E}"

    # -------------------- initialize and copy passed values to local variables

    try:
        restart = kwargs["restart"]
    except:
        restart = False

    # --- simulation data
    simdat = the_model.simulationData()

    # --- options
    verbose = simdat.VERBOSITY > 0

    # --- data
    ileg = int(simdat.getData("leg number"))
    legs = simdat.getData("leg data")[ileg:]
    lnl = len(str(len(legs)))
    simsteps = simdat.getData("number of steps")
    t_beg = simdat.getData("time")
    dt = simdat.getData("time step")
    vdum = np.zeros(6,dtype=int)
    sig_hld = np.zeros(6)

    # --- material data
    material = simdat.MATERIAL
    matdat = material.materialData()
    J0 = matdat.getData("jacobian")

    # -------------------------------------------------------- initialize model
    # --- start output file and Mathematica files
    msg = "starting calculations for simulation %s"%simdat.SIMNAME
    reportMessage(iam, msg)
    setupOutputFile(simdat, matdat, restart)

    # --- call the material model with zero state
    if ileg == 0:
        material.updateState(simdat,matdat)

        # advance and write data
        simdat.advanceAllData()
        matdat.advanceAllData()
        writeState(simdat,matdat)
        pass

    # ----------------------------------------------------------------------- #

    # --------------------------------------------------- begin{processing leg}
    for leg in legs:

        # test restart capability
        if simdat.TEST_RESTART and not restart:
            if ileg == int(len(legs)/2):
                print('\n\nStopping to test Payette restart capabilities.\n'
                      'Restart the simulation by executing\n\n'
                      '\t\trunPayette %s.prf\n\n'
                      %os.path.splitext(os.path.basename(
                            simdat.OUTFILE))[0])
                sys.exit(76)
                pass
            pass

        # read inputs and initialize for this leg
        lnum, t_end, nsteps, ltype, prdef = leg
        lns = len(str(nsteps))
        delt = t_end - t_beg
        if delt == 0.: continue
        nv, dflg = 0, list(set(ltype))
        nprints = simdat.NPRINTS if simdat.NPRINTS else nsteps
        if simdat.EMIT == "sparse": nprints = min(10,nsteps)
        print_interval = max(1,int(nsteps/nprints))

        # pass values from the end of the last leg to beginning of this leg
        eps_beg = simdat.getData("strain")
        sig_beg = matdat.getData("stress")
        F_beg = simdat.getData("deformation gradient")
        v = simdat.getData("prescribed stress components")
        if simdat.EFIELD_SIM:
            efld_beg = simdat.getData("electric field")
            pass

        if len(v):
            prsig_beg = simdat.getData("prescribed stress")
            j = 0
            for i in v:
                sig_beg[i] = prsig_beg[j]
                j += 1
                continue
            pass

        if verbose:
            reportMessage(iam, cons_msg.format(lnum, lnl, 1, lns, t_beg, dt))
            pass

        # --- loop through components of prdef and compute the values at the end
        #     of this leg:
        #       for ltype = 1: eps_end at t_end -> eps_beg + prdef*delt
        #       for ltype = 2: eps_end at t_end -> prdef
        #       for ltype = 3: Pf at t_end -> sig_beg + prdef*delt
        #       for ltype = 4: Pf at t_end -> prdef
        #       for ltype = 5: F_end at t_end -> prdef
        #       for ltype = 6: efld_end at t_end-> prdef
        eps_end, F_end = Z6, I9
        if simdat.EFIELD_SIM:
            efld_end = Z3
            pass

        # if stress is prescribed, we don't compute sig_end just yet, but sig_hld
        # which holds just those values of stress that are actually prescribed.


        for i in range(len(prdef)):

            if ltype[i] not in [0, 1, 2, 3, 4, 5, 6 ]:
                msg = ('\nERROR: Invalid load type (ltype) parameter\n'
                       'prescribed for leg %i\n'%lnum)
                reportError(iam,msg)
                return 1

            if ltype[i] == 0:
                continue

            # -- strain rate
            elif i < 6 and ltype[i] == 1:
                eps_end[i] = eps_beg[i] + prdef[i]*delt

            # -- strain
            elif i < 6 and ltype[i] == 2:
                eps_end[i] = prdef[i]

            # stress rate
            elif i < 6 and ltype[i] == 3:
                sig_hld[i] = sig_beg[i] + prdef[i]*delt
                vdum[nv] = i
                nv += 1

            # stress
            elif i < 6 and ltype[i] == 4:
                sig_hld[i] = prdef[i]
                vdum[nv] = i
                nv += 1

            # deformation gradient
            elif ltype[i] == 5:
                F_end[i] = prdef[i]

            # electric field
            elif simdat.EFIELD_SIM and ( i >= 9 and ltype[i] == 6 ):
                efld_end[i-9] = prdef[i]
                pass

            continue

        v = vdum[0:nv]
        simdat.advanceData("prescribed stress components",v)
        if len(v):
            prsig_beg, prsig_end = sig_beg[v], sig_hld[v]
            Js = J0[[[x] for x in v],v]
            pass

        t = t_beg
        dt = delt/nsteps

        # ---------------------------------------------- begin{processing step}
        for n in range(nsteps):

            t += dt
            simsteps += 1
            simdat.advanceData("number of steps",simsteps)

            # interpolate values of E, F, EF, and P for the current step
            a1 = float(nsteps - (n + 1))/nsteps
            a2 = float(n + 1)/nsteps

            eps_int = a1*eps_beg + a2*eps_end

            F_int = a1*F_beg + a2*F_end

            if simdat.EFIELD_SIM:
                efld_int = a1*efld_beg + a2*efld_end
                pass

            # --- initial guess for depsdt[v]
            if len(v):
                prsig_int = a1*prsig_beg + a2*prsig_end
                prsig_dif = prsig_int - matdat.getData("stress")[v]

                depsdt = (eps_int - simdat.getData("strain"))/dt
                try:
                    depsdt[v] = np.linalg.solve(Js,prsig_dif)/dt
                except:
                    depsdt[v] -= np.linalg.lstsq(Js,prsig_dif)[0]/dt
                    pass
                pass

            # advance known values to end of step
            simdat.advanceData("time",t)
            simdat.advanceData("time step",dt)
            if simdat.EFIELD_SIM:
                simdat.advanceData("electric field",efld_int)
                pass

            # --- find d (symmetric part of velocity gradient)
            if not len(v):

                if dflg[0] == 5:
                    # --- deformation gradient prescribed
                    simdat.advanceData("prescribed deformation gradient",F_int)
                    pkin.velGradCompFromF(simdat)

                else:
                    # --- strain or strain rate prescribed
                    simdat.advanceData("prescribed strain",eps_int)
                    pkin.velGradCompFromE(simdat)

            else:
                # --- One or more stresses prescribed
                simdat.advanceData("strain rate",depsdt)
                simdat.advanceData("prescribed stress",prsig_int)
                pkin.velGradCompFromP(simdat,matdat)
                simdat.advanceData("strain rate")
                depsdt = simdat.getData("strain rate")
                simdat.storeData("rate of deformation",depsdt)
                pass

            # --- update the deformation to the end of the step at this point,
            #     the rate of deformation and vorticity to the end of the step
            #     are known, advance them.
            simdat.advanceData("rate of deformation")
            simdat.advanceData("vorticity")

            # find the current {deformation gradient,strain} and advance them
            pkin.updateDeformation(simdat)
            simdat.advanceData("deformation gradient")
            simdat.advanceData("strain")
            simdat.advanceData("equivalent strain")

            if simdat.USE_TABLE:
                # use the actual table values, not the values computed above
                if dflg == [1] or dflg == [2] or dflg == [1,2]:
                    simdat.advanceData("strain",eps_int)
                elif dflg[0] == 5:
                    simdat.advanceData("deformation gradient",F_int)
                    pass
                pass

            # update material state
            material.updateState(simdat,matdat)

            # advance all data after updating state
            matdat.storeData("stress rate", (matdat.getData("stress",cur=True) -
                                             matdat.getData("stress"))/dt)
            matdat.advanceAllData()

            # multilevel failure. since there is only one element, it really
            # isn't multilevel
            if matdat.MULTI_LEVEL_FAIL:
                sv = matdat.getData("extra variables")
                if sv[cmod.cflg_idx] == 1:
                    reportMessage(iam,"multilevel failure beginning")
                    reportMessage(iam,"Failure ratio is %12.5e"
                                  %sv[cmod.fratio_idx])
                    sv[cmod.cflg_idx] = 3
                elif sv[cmod.cflg_idx] == 4:
                    matdat.advanceData("failed",True)
                    pass
                pass

            # --- write state to file
            if (nsteps-n)%print_interval == 0:
                writeState(simdat,matdat)
                pass

            if simdat.SCREENOUT or ( verbose and (2*n - nsteps) == 0 ):
                reportMessage(iam, cons_msg.format(lnum, lnl, n, lns, t, dt))
                pass

            if matdat.getData("failed"):
                msg = 'material failed on leg %i, step %i, at time %f'%(lnum,n,t)
                reportMessage(iam,msg)
                reportMessage(iam, "{0} Payette simulation ran to completion"
                              .format(the_model.name))
                return 0

            # ------------------------------------------ begin{end of step SQA}
            if simdat.SQA:
                if dflg == [1] or dflg == [2] or dflg == [1,2]:
                    eps_tmp = simdat.getData("strain")
                    max_diff = np.max(np.abs(eps_tmp - eps_int))
                    dnom = max(np.max(np.abs(eps_int)),0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff/dnom
                    if rel_diff > accuracyLim() and max_diff > epsilon():
                        msg = ('E differs from prdef excessively at end of step '
                               '%i of leg %i with a percent difference of %f'
                               %(n,lnum,rel_diff*100.))
                        reportWarning(iam,msg)
                        pass
                    pass
                elif dflg == [5]:
                    F_tmp = simdat.getData("deformation gradient")
                    max_diff = (np.max(F_tmp - F_int))/np.max(np.abs(F_int))
                    dnom = max(np.max(np.abs(F_int)),0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff/dnom
                    if rel_diff > accuracyLim() and max_diff > epsilon():
                        msg = ('F differs from prdef excessively at end of step '
                               ' with max_diff %f' %max_diff)
                        reportWarning(iam,msg)
                        pass
                    pass
                pass
            # -------------------------------------------- end{end of step SQA}

            continue # continue to next step
        # ------------------------------------------------ end{processing step}

        # --- pass quantities from end of leg to beginning of new leg
        simdat.advanceData("leg number",ileg+1)
        ileg += 1

        if simdat.WRITE_VANDD_TABLE:
            writeVelAndDispTable(simdat.INITIAL_TIME, simdat.TERMINATION_TIME,
                                 t_beg, t_end, eps_beg, eps_end, simdat.KAPPA)
            pass

        # advances time must come after writing the v & d tables above
        t_beg = t_end

        if simdat.WRITE_RESTART:
            with open(simdat.RESTART_FILE,'wb') as f: pickle.dump(the_model,f,2)
            pass

        # --- print message to screen
        if verbose and nsteps > 1:
            reportMessage(iam, cons_msg.format(lnum, lnl, n + 1, lns, t, dt))
            pass


        # ----------------------------------------------- begin{end of leg SQA}
        if simdat.SQA:
            if dflg == [1] or dflg == [2] or dflg == [1,2]:
                eps_tmp = simdat.getData("strain")
                max_diff = np.max(np.abs(eps_tmp - eps_end))
                dnom = np.max(eps_end) if np.max(eps_end) >= epsilon() else 1.
                rel_diff = max_diff/dnom
                if rel_diff > accuracyLim() and max_diff > epsilon():
                    msg = ('E differs from prdef excessively at end of '
                           'leg %i with relative diff %f'%(lnum,rel_diff))
                    reportWarning(iam,msg)
                pass

            elif dflg == [5]:
                F_tmp = simdat.getData("deformation gradient")
                max_diff = np.max(np.abs(F_tmp - F_end))/np.max(np.abs(F_end))
                dnom = np.max(F_end) if np.max(F_end) >= epsilon() else 1.
                rel_diff = max_diff/dnom
                if rel_diff > accuracyLim() and max_diff > epsilon():
                    msg = ('F differs from prdef excessively at end of leg '
                           ' with max_diff %f' %max_diff)
                    reportWarning(iam,msg)
                pass
            pass
        # ------------------------------------------------- end{end of leg SQA}

        continue # continue to next leg
    # ----------------------------------------------------- end{processing leg}

    reportMessage(iam, "{0} Payette simulation ran to completion"
                  .format(the_model.name))

    return 0
