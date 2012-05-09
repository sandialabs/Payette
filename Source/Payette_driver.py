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

iam = "Payette_driver.solid_driver(the_model,restart)"

np.set_printoptions(precision=2)

def eos_driver(the_model, **kwargs):
    """
    NAME
       eos_driver:

    PURPOSE
       Run the single element simulation for an instance of the main Payette
       class the_model

    INPUT
       the_model: Payette container class instance

    OUTPUT
       outfile: the_model.getOutputFile()

    PASSED AND LOCAL VARIABLES
    AUTHORS
       Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    """
    if debug: pdb = __import__('pdb')
    cons_msg = "leg {0:{1}d}, step {2:{3}d}, time {4:.4E}, dt {5:.4E}"

    K2eV = 8.617343e-5
    erg2joule = 1.0e-4

    simdat = the_model.simulation_data()
    material = simdat.MATERIAL
    matdat = material.material_data()

    eos_model = material.constitutiveModel()

    nprints = simdat.getOption("nprints")
    rho_temp_pairs = simdat.getData("leg data")
    base_name = os.path.splitext(simdat.OUTFILE)[0]

    txtfmt = lambda x: "{0:>25s}".format(str(x))
    fltfmt = lambda x: "{0:25.14e}".format(float(x))


################################################################################
###############             DENSITY-TEMPERATURE LEGS             ###############
################################################################################
    if len(rho_temp_pairs) != 0:
        simdat.OUTFILE = base_name + ".out"
        OUTFILE = open(simdat.OUTFILE, "w")

        headers = [x[0] for x in sorted(eos_model.outdata.iteritems())]
        msg = "{0}\n".format("".join([txtfmt(x) for x in headers]))
        OUTFILE.write(msg)

        for pair in rho_temp_pairs:
            eos_model.evaluate_eos(simdat, matdat,
                              rho = pair[0] / 1000.0, temp = pair[1] * K2eV)
            # Convert from CGSEV to MKSK
            eos_model.outdata["DENSITY"] *= 1000.0
            eos_model.outdata["ENERGY"] *= erg2joule
            eos_model.outdata["PRESSURE"] /= 10.0
            eos_model.outdata["SOUNDSPEED"] /= 100.0
            eos_model.outdata["TEMPERATURE"] /= K2eV

            values = [x[1] for x in sorted(eos_model.outdata.iteritems())]
            msg = "{0}\n".format("".join([fltfmt(x) for x in values]))
            OUTFILE.write(msg)

        OUTFILE.close()
     

################################################################################
###############                     SURFACE                      ###############
################################################################################
    if (simdat.getOption("surface increments") != None and
        simdat.getOption("density range") != None and
        simdat.getOption("temperature range") != None):

        t_range = simdat.getOption("temperature range")
        rho_range = simdat.getOption("density range")
        surf_incr = simdat.getOption("surface increments")

        simdat.OUTFILE = base_name + ".surface"
        OUTFILE = open(simdat.OUTFILE, "w")

        loginf("=" * (80 - 6))
        loginf("Begin surface")
        DEJAVU = False
        idx = 0
        for rho in np.linspace(rho_range[0], rho_range[1], surf_incr):
            for temp in np.linspace(t_range[0], t_range[1], surf_incr):
                idx += 1
                if idx%int(surf_incr**2/float(nprints)) == 0:
                    loginf("Surface step {0}/{1}".format(idx,surf_incr**2))
                # convert to CGSEV from MKSK
                tmprho = rho/1000.0
                tmptemp = K2eV*temp

                eos_model.evaluate_eos(simdat, matdat,
                                       rho = tmprho, temp = tmptemp)

                # Write the headers if this is the first time through
                if not DEJAVU:
                    DEJAVU = True
                    headers = [x[0] for x in
                               sorted(eos_model.outdata.iteritems())]
                    msg = "{0}\n".format("".join([txtfmt(x) for x in headers]))
                    OUTFILE.write(msg)

                # Convert from CGSEV to MKSK
                eos_model.outdata["DENSITY"] *= 1000.0
                eos_model.outdata["ENERGY"] *= erg2joule
                eos_model.outdata["PRESSURE"] /= 10.0
                eos_model.outdata["SOUNDSPEED"] /= 100.0
                eos_model.outdata["TEMPERATURE"] /= K2eV

                # Write the values
                values = [x[1] for x in sorted(eos_model.outdata.iteritems())]
                msg = "{0}\n".format("".join([fltfmt(x) for x in values]))
                OUTFILE.write(msg)

        OUTFILE.flush()
        OUTFILE.close()
        loginf("End surface")
        loginf("Surface file: {0}".format(simdat.OUTFILE))

################################################################################
###############                     ISOTHERM                     ###############
################################################################################
    
    if (simdat.getOption("path increments") != None and
        simdat.getOption("path isotherm") != None and
        simdat.getOption("density range") != None):

        # isotherm = [density, temperature]
        isotherm = simdat.getOption("path isotherm")
        rho_range = simdat.getOption("density range")
        path_incr = simdat.getOption("path increments")

        if not rho_range[0] <= isotherm[0] <= rho_range[1]:
            sys.exit("initial isotherm density not within range")

        simdat.OUTFILE = base_name + ".isotherm"
        OUTFILE = open(simdat.OUTFILE, "w")

        loginf("=" * (80 - 6))
        loginf("Begin isotherm")
        DEJAVU = False
        idx = 0
        for rho in np.linspace(isotherm[0], rho_range[1], path_incr):
            idx += 1
            if idx%int(path_incr/float(nprints)) == 0:
                loginf("Isotherm step {0}/{1}".format(idx,path_incr))

            tmprho = rho/1000.0
            tmptemp = K2eV*isotherm[1]

            eos_model.evaluate_eos(simdat, matdat,
                                   rho = tmprho, temp = tmptemp)

            # Write the headers if this is the first time through
            if not DEJAVU:
                DEJAVU = True
                headers = [x[0] for x in
                           sorted(eos_model.outdata.iteritems())]
                msg = "{0}\n".format("".join([txtfmt(x) for x in headers]))
                OUTFILE.write(msg)

            # Convert from CGSEV to MKSK
            eos_model.outdata["DENSITY"] *= 1000.0
            eos_model.outdata["ENERGY"] *= erg2joule
            eos_model.outdata["PRESSURE"] /= 10.0
            eos_model.outdata["SOUNDSPEED"] /= 100.0
            eos_model.outdata["TEMPERATURE"] /= K2eV

            # Write the values
            values = [x[1] for x in sorted(eos_model.outdata.iteritems())]
            msg = "{0}\n".format("".join([fltfmt(x) for x in values]))
            OUTFILE.write(msg)

        OUTFILE.flush()
        OUTFILE.close()
        loginf("End isotherm")
        loginf("Isotherm file: {0}".format(simdat.OUTFILE))

################################################################################
###############                     HUGONIOT                     ###############
################################################################################

    if (simdat.getOption("path increments") != None and
        simdat.getOption("path hugoniot") != None and
        simdat.getOption("temperature range") != None and
        simdat.getOption("density range") != None):

        # hugoniot = [density, temperature]
        hugoniot = simdat.getOption("path hugoniot")
        rho_range = simdat.getOption("density range")
        t_range = simdat.getOption("temperature range")
        path_incr = simdat.getOption("path increments")


        if not rho_range[0] <= hugoniot[0] <= rho_range[1]:
            sys.exit("initial hugoniot density not within range")
        if not t_range[0] <= hugoniot[1] <= t_range[1]:
            sys.exit("initial hugoniot temperature not within range")

        simdat.OUTFILE = base_name + ".hugoniot"
        OUTFILE = open(simdat.OUTFILE, "w")

        loginf("=" * (80 - 6))
        loginf("Begin Hugoniot")

        init_density_MKSK = hugoniot[0]
        init_temperature_MKSK = hugoniot[1]

        # Convert to CGSEV
        init_density = init_density_MKSK/1000.0
        init_temperature = K2eV*init_temperature_MKSK

        eos_model.evaluate_eos(simdat, matdat,
                               rho = init_density, temp = init_temperature)

        init_energy = eos_model.outdata["ENERGY"]
        init_pressure = eos_model.outdata["PRESSURE"]

        idx = 0
        DEJAVU = False
        e = init_energy
        for rho in np.linspace(hugoniot[0], rho_range[1], path_incr):
            idx += 1
            if idx%int(path_incr/float(nprints)) == 0:
                loginf("Hugoniot step {0}/{1}".format(idx,path_incr))
            #
            # Here we solve the Rankine-Hugoniot equation as
            # a function of energy with constant density:
            #
            # E-E0 == 0.5*[P(E,V)+P0]*(V0-V)
            #
            # Where V0 = 1/rho0 and V = 1/rho. We rewrite it as:
            #
            # 0.5*[P(E,V)+P0]*(V0-V)-E+E0 == 0.0 = f(E)
            #
            # The derivative is given by:
            #
            # df(E)/dE = 0.5*(dP/dE)*(1/rho0 - 1/rho) - 1
            #
            # The solution to the first equation is found by a simple
            # application of newton's method:
            #
            # x_n+1 = x_n - f(E)/(df(E)/dE)
            #

            r = rho/1000.0
            a = (1./init_density - 1./r)/2.

            converged_idx = 0
            CONVERGED = False
            while not CONVERGED:
                converged_idx += 1

                eos_model.evaluate_eos(simdat, matdat, rho = r, enrg = e)

                f = (eos_model.outdata["PRESSURE"] + init_pressure)*a - e + init_energy
                df = eos_model.outdata["DPDT"]/eos_model.outdata["DEDT"]*a - 1.0
                e = e - f/df*1.5
                errval = abs(f/init_energy)
                if errval < 1.0e-9 or converged_idx > 100:
                    CONVERGED = True
                    if converged_idx > 100:
                        loginf("Max iterations reached (tol = {0:14.10e}).\n".format(1.0e-9)+
                               "rel error   = {0:14.10e}\n".format(float(errval))+
                               "abs error   = {0:14.10e}\n".format(float(f))+
                               "func val    = {0:14.10e}\n".format(float(f))+
                               "init_energy = {0:14.10e}\n".format(float(init_energy)))

            # Write the headers if this is the first time through
            if not DEJAVU:
                DEJAVU = True
                headers = [x[0] for x in sorted(eos_model.outdata.iteritems())]
                msg = "{0}\n".format("".join([txtfmt(x) for x in headers]))
                OUTFILE.write(msg)

            # Convert from CGSEV to MKSK
            eos_model.outdata["DENSITY"] *= 1000.0
            eos_model.outdata["ENERGY"] *= erg2joule
            eos_model.outdata["PRESSURE"] /= 10.0
            eos_model.outdata["SOUNDSPEED"] /= 100.0
            eos_model.outdata["TEMPERATURE"] /= K2eV

            # Write the values
            values = [x[1] for x in sorted(eos_model.outdata.iteritems())]
            msg = "{0}\n".format("".join([fltfmt(x) for x in values]))
            OUTFILE.write(msg)

        OUTFILE.flush()
        OUTFILE.close()
        loginf("End Hugoniot")
        loginf("Hugoniot file: {0}".format(simdat.OUTFILE))

    return 0

 
def solid_driver(the_model, **kwargs):
    """
    NAME
       solid_driver:

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
    simdat = the_model.simulation_data()

    # --- options
    verbose = simdat.VERBOSITY > 0

    # --- data
    ileg = int(simdat.getData("leg number"))
    legs = simdat.getData("leg data")[ileg:]
    lnl = len(str(len(legs)))
    simsteps = simdat.getData("number of steps")
    t_beg = simdat.getData("time")
    dt = simdat.getData("time step")
    vdum = np.zeros(6, dtype=int)
    sig_hld = np.zeros(6)

    # --- material data
    material = simdat.MATERIAL
    matdat = material.material_data()
    J0 = matdat.getData("jacobian")

    # -------------------------------------------------------- initialize model
    # --- start output file and Mathematica files
    msg = "starting calculations for simulation %s"%simdat.SIMNAME
    reportMessage(iam, msg)
    setupOutputFile(simdat, matdat, restart)

    # --- call the material model with zero state
    if ileg == 0:
        material.updateState(simdat, matdat)

        # advance and write data
        simdat.advanceAllData()
        matdat.advanceAllData()
        writeState(simdat, matdat)

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
        eps_beg = matdat.getData("strain")
        sig_beg = matdat.getData("stress")
        F_beg = matdat.getData("deformation gradient")
        v = matdat.getData("prescribed stress components")
        if matdat.EFIELD_SIM:
            efld_beg = matdat.getData("electric field")

        if len(v):
            prsig_beg = matdat.getData("prescribed stress")
            j = 0
            for i in v:
                sig_beg[i] = prsig_beg[j]
                j += 1
                continue

        if verbose:
            reportMessage(iam, cons_msg.format(lnum, lnl, 1, lns, t_beg, dt))

        # --- loop through components of prdef and compute the values at the end
        #     of this leg:
        #       for ltype = 1: eps_end at t_end -> eps_beg + prdef*delt
        #       for ltype = 2: eps_end at t_end -> prdef
        #       for ltype = 3: Pf at t_end -> sig_beg + prdef*delt
        #       for ltype = 4: Pf at t_end -> prdef
        #       for ltype = 5: F_end at t_end -> prdef
        #       for ltype = 6: efld_end at t_end-> prdef
        eps_end, F_end = Z6, I9
        if matdat.EFIELD_SIM:
            efld_end = Z3

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
            elif matdat.EFIELD_SIM and ( i >= 9 and ltype[i] == 6 ):
                efld_end[i-9] = prdef[i]

            continue

        v = vdum[0:nv]
        matdat.advanceData("prescribed stress components", v)
        if len(v):
            prsig_beg, prsig_end = sig_beg[v], sig_hld[v]
            Js = J0[[[x] for x in v], v]

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

            if matdat.EFIELD_SIM:
                efld_int = a1*efld_beg + a2*efld_end

            # --- initial guess for depsdt[v]
            if len(v):
                prsig_int = a1 * prsig_beg + a2 * prsig_end
                prsig_dif = prsig_int - matdat.getData("stress")[v]

                depsdt = (eps_int - matdat.getData("strain"))/dt
                try:
                    depsdt[v] = np.linalg.solve(Js,prsig_dif)/dt
                except:
                    depsdt[v] -= np.linalg.lstsq(Js,prsig_dif)[0]/dt

            # advance known values to end of step
            simdat.advanceData("time",t)
            simdat.advanceData("time step",dt)
            if matdat.EFIELD_SIM:
                matdat.advanceData("electric field",efld_int)

            # --- find d (symmetric part of velocity gradient)
            if not len(v):

                if dflg[0] == 5:
                    # --- deformation gradient prescribed
                    matdat.advanceData("prescribed deformation gradient",F_int)
                    pkin.velGradCompFromF(simdat, matdat)

                else:
                    # --- strain or strain rate prescribed
                    matdat.advanceData("prescribed strain",eps_int)
                    pkin.velGradCompFromE(simdat, matdat)

            else:
                # --- One or more stresses prescribed
                matdat.advanceData("strain rate", depsdt)
                matdat.advanceData("prescribed stress", prsig_int)
                pkin.velGradCompFromP(simdat, matdat)
                matdat.advanceData("strain rate")
                depsdt = matdat.getData("strain rate")
                matdat.storeData("rate of deformation", depsdt)

            # --- update the deformation to the end of the step at this point,
            #     the rate of deformation and vorticity to the end of the step
            #     are known, advance them.
            matdat.advanceData("rate of deformation")
            matdat.advanceData("vorticity")

            # find the current {deformation gradient,strain} and advance them
            pkin.updateDeformation(simdat, matdat)
            matdat.advanceData("deformation gradient")
            matdat.advanceData("strain")
            matdat.advanceData("equivalent strain")

            if simdat.USE_TABLE:
                # use the actual table values, not the values computed above
                if dflg == [1] or dflg == [2] or dflg == [1,2]:
                    matdat.advanceData("strain",eps_int)
                elif dflg[0] == 5:
                    matdat.advanceData("deformation gradient",F_int)

            # update material state
            material.updateState(simdat, matdat)

            # advance all data after updating state
            matdat.storeData("stress rate", (matdat.getData("stress",cur=True) -
                                             matdat.getData("stress"))/dt)
            matdat.advanceAllData()

            # --- write state to file
            if (nsteps-n)%print_interval == 0:
                writeState(simdat, matdat)

            if simdat.SCREENOUT or ( verbose and (2*n - nsteps) == 0 ):
                reportMessage(iam, cons_msg.format(lnum, lnl, n, lns, t, dt))

            # ------------------------------------------ begin{end of step SQA}
            if simdat.SQA:
                if dflg == [1] or dflg == [2] or dflg == [1,2]:
                    eps_tmp = matdat.getData("strain")
                    max_diff = np.max(np.abs(eps_tmp - eps_int))
                    dnom = max(np.max(np.abs(eps_int)),0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff/dnom
                    if rel_diff > accuracyLim() and max_diff > epsilon():
                        msg = ('E differs from prdef excessively at end of step '
                               '%i of leg %i with a percent difference of %f'
                               %(n,lnum,rel_diff*100.))
                        reportWarning(iam,msg)

                elif dflg == [5]:
                    F_tmp = matdat.getData("deformation gradient")
                    max_diff = (np.max(F_tmp - F_int))/np.max(np.abs(F_int))
                    dnom = max(np.max(np.abs(F_int)),0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff/dnom
                    if rel_diff > accuracyLim() and max_diff > epsilon():
                        msg = ('F differs from prdef excessively at end of step '
                               ' with max_diff %f' %max_diff)
                        reportWarning(iam,msg)

            # -------------------------------------------- end{end of step SQA}

            continue # continue to next step
        # ------------------------------------------------ end{processing step}

        # --- pass quantities from end of leg to beginning of new leg
        simdat.advanceData("leg number",ileg+1)
        ileg += 1

        if simdat.WRITE_VANDD_TABLE:
            writeVelAndDispTable(simdat.INITIAL_TIME, simdat.TERMINATION_TIME,
                                 t_beg, t_end, eps_beg, eps_end, simdat.KAPPA)

        # advances time must come after writing the v & d tables above
        t_beg = t_end

        if simdat.WRITE_RESTART:
            with open(simdat.RESTART_FILE,'wb') as f: pickle.dump(the_model,f,2)

        # --- print message to screen
        if verbose and nsteps > 1:
            reportMessage(iam, cons_msg.format(lnum, lnl, n + 1, lns, t, dt))

        # ----------------------------------------------- begin{end of leg SQA}
        if simdat.SQA:
            if dflg == [1] or dflg == [2] or dflg == [1,2]:
                eps_tmp = matdat.getData("strain")
                max_diff = np.max(np.abs(eps_tmp - eps_end))
                dnom = np.max(eps_end) if np.max(eps_end) >= epsilon() else 1.
                rel_diff = max_diff/dnom
                if rel_diff > accuracyLim() and max_diff > epsilon():
                    msg = ('E differs from prdef excessively at end of '
                           'leg %i with relative diff %f'%(lnum,rel_diff))
                    reportWarning(iam,msg)

            elif dflg == [5]:
                F_tmp = matdat.getData("deformation gradient")
                max_diff = np.max(np.abs(F_tmp - F_end))/np.max(np.abs(F_end))
                dnom = np.max(F_end) if np.max(F_end) >= epsilon() else 1.
                rel_diff = max_diff/dnom
                if rel_diff > accuracyLim() and max_diff > epsilon():
                    msg = ('F differs from prdef excessively at end of leg '
                           ' with max_diff %f' %max_diff)
                    reportWarning(iam,msg)

        # ------------------------------------------------- end{end of leg SQA}

        continue # continue to next leg
    # ----------------------------------------------------- end{processing leg}

    return 0
