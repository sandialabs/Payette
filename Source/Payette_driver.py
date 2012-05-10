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
import Source.Payette_kinematics as pk
import Source.Payette_utils as pu
import Source.Payette_tensor as pt

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
    cons_msg = "leg {0:{1}d}, step {2:{3}d}, time {4:.4E}, dt {5:.4E}"

    K2eV = 8.617343e-5
    erg2joule = 1.0e-4

    simdat = the_model.simulation_data()
    material = simdat.MATERIAL
    matdat = material.material_data()

    eos_model = material.constitutive_model

    nprints = simdat.get_option("nprints")
    rho_temp_pairs = simdat.get_data("leg data")
    simdir = the_model.simdir
    simnam = the_model.name

    txtfmt = lambda x: "{0:>25s}".format(str(x))
    fltfmt = lambda x: "{0:25.14e}".format(float(x))


################################################################################
###############             DENSITY-TEMPERATURE LEGS             ###############
################################################################################
    if len(rho_temp_pairs) != 0:
        out_fnam = os.path.join(simdir, simnam + ".out")
        out_fobj = open(out_fnam, "w")

        headers = [x[0] for x in sorted(eos_model.outdata.iteritems())]
        msg = "{0}\n".format("".join([txtfmt(x) for x in headers]))
        out_fobj.write(msg)

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
            out_fobj.write(msg)

        out_fobj.close()


################################################################################
###############                     SURFACE                      ###############
################################################################################
    if (simdat.get_option("surface increments") != None and
        simdat.get_option("density range") != None and
        simdat.get_option("temperature range") != None):

        t_range = simdat.get_option("temperature range")
        rho_range = simdat.get_option("density range")
        surf_incr = simdat.get_option("surface increments")

        out_fnam = os.path.join(simdir, simnam + ".surface")
        out_fobj = open(out_fnam, "w")

        pu.loginf("=" * (80 - 6))
        pu.loginf("Begin surface")
        DEJAVU = False
        idx = 0
        for rho in np.linspace(rho_range[0], rho_range[1], surf_incr):
            for temp in np.linspace(t_range[0], t_range[1], surf_incr):
                idx += 1
                if idx%int(surf_incr**2/float(nprints)) == 0:
                    pu.loginf("Surface step {0}/{1}".format(idx,surf_incr**2))
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
                    out_fobj.write(msg)

                # Convert from CGSEV to MKSK
                eos_model.outdata["DENSITY"] *= 1000.0
                eos_model.outdata["ENERGY"] *= erg2joule
                eos_model.outdata["PRESSURE"] /= 10.0
                eos_model.outdata["SOUNDSPEED"] /= 100.0
                eos_model.outdata["TEMPERATURE"] /= K2eV

                # Write the values
                values = [x[1] for x in sorted(eos_model.outdata.iteritems())]
                msg = "{0}\n".format("".join([fltfmt(x) for x in values]))
                out_fobj.write(msg)

        out_fobj.flush()
        out_fobj.close()
        pu.loginf("End surface")
        pu.loginf("Surface file: {0}".format(out_fnam))

################################################################################
###############                     ISOTHERM                     ###############
################################################################################

    if (simdat.get_option("path increments") != None and
        simdat.get_option("path isotherm") != None and
        simdat.get_option("density range") != None):

        # isotherm = [density, temperature]
        isotherm = simdat.get_option("path isotherm")
        rho_range = simdat.get_option("density range")
        path_incr = simdat.get_option("path increments")

        if not rho_range[0] <= isotherm[0] <= rho_range[1]:
            sys.exit("initial isotherm density not within range")

        out_fnam = os.path.join(simdir, simnam + ".isotherm")
        out_fobj = open(out_fnam, "w")

        pu.loginf("=" * (80 - 6))
        pu.loginf("Begin isotherm")
        DEJAVU = False
        idx = 0
        for rho in np.linspace(isotherm[0], rho_range[1], path_incr):
            idx += 1
            if idx%int(path_incr/float(nprints)) == 0:
                pu.loginf("Isotherm step {0}/{1}".format(idx,path_incr))

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
                out_fobj.write(msg)

            # Convert from CGSEV to MKSK
            eos_model.outdata["DENSITY"] *= 1000.0
            eos_model.outdata["ENERGY"] *= erg2joule
            eos_model.outdata["PRESSURE"] /= 10.0
            eos_model.outdata["SOUNDSPEED"] /= 100.0
            eos_model.outdata["TEMPERATURE"] /= K2eV

            # Write the values
            values = [x[1] for x in sorted(eos_model.outdata.iteritems())]
            msg = "{0}\n".format("".join([fltfmt(x) for x in values]))
            out_fobj.write(msg)

        out_fobj.flush()
        out_fobj.close()
        pu.loginf("End isotherm")
        pu.loginf("Isotherm file: {0}".format(out_fnam))

################################################################################
###############                     HUGONIOT                     ###############
################################################################################

    if (simdat.get_option("path increments") != None and
        simdat.get_option("path hugoniot") != None and
        simdat.get_option("temperature range") != None and
        simdat.get_option("density range") != None):

        # hugoniot = [density, temperature]
        hugoniot = simdat.get_option("path hugoniot")
        rho_range = simdat.get_option("density range")
        t_range = simdat.get_option("temperature range")
        path_incr = simdat.get_option("path increments")


        if not rho_range[0] <= hugoniot[0] <= rho_range[1]:
            sys.exit("initial hugoniot density not within range")
        if not t_range[0] <= hugoniot[1] <= t_range[1]:
            sys.exit("initial hugoniot temperature not within range")

        out_fnam = os.path.join(simdir, simnam + ".hugoniot")
        out_fobj = open(out_fnam, "w")

        pu.loginf("=" * (80 - 6))
        pu.loginf("Begin Hugoniot")

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
                pu.loginf("Hugoniot step {0}/{1}".format(idx,path_incr))
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
                        pu.loginf("Max iterations reached (tol = {0:14.10e}).\n".format(1.0e-9)+
                               "rel error   = {0:14.10e}\n".format(float(errval))+
                               "abs error   = {0:14.10e}\n".format(float(f))+
                               "func val    = {0:14.10e}\n".format(float(f))+
                               "init_energy = {0:14.10e}\n".format(float(init_energy)))

            # Write the headers if this is the first time through
            if not DEJAVU:
                DEJAVU = True
                headers = [x[0] for x in sorted(eos_model.outdata.iteritems())]
                msg = "{0}\n".format("".join([txtfmt(x) for x in headers]))
                out_fobj.write(msg)

            # Convert from CGSEV to MKSK
            eos_model.outdata["DENSITY"] *= 1000.0
            eos_model.outdata["ENERGY"] *= erg2joule
            eos_model.outdata["PRESSURE"] /= 10.0
            eos_model.outdata["SOUNDSPEED"] /= 100.0
            eos_model.outdata["TEMPERATURE"] /= K2eV

            # Write the values
            values = [x[1] for x in sorted(eos_model.outdata.iteritems())]
            msg = "{0}\n".format("".join([fltfmt(x) for x in values]))
            out_fobj.write(msg)

        out_fobj.flush()
        out_fobj.close()
        pu.loginf("End Hugoniot")
        pu.loginf("Hugoniot file: {0}".format(out_fnam))

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
    cons_msg = "leg {0:{1}d}, step {2:{3}d}, time {4:.4E}, dt {5:.4E}"

    # -------------------- initialize and copy passed values to local variables
    try:
        restart = kwargs["restart"]
    except KeyError:
        restart = False

    # --- simulation data
    simdat = the_model.simulation_data()

    if simdat.DEBUG:
        pdb = __import__('pdb')

    # --- options
    verbose = simdat.VERBOSITY > 0

    # --- data
    ileg = int(simdat.get_data("leg number"))
    legs = simdat.get_data("leg data")[ileg:]
    lnl = len(str(len(legs)))
    simsteps = simdat.get_data("number of steps")
    t_beg = simdat.get_data("time")
    dt = simdat.get_data("time step")
    vdum = np.zeros(6, dtype=int)
    sig_hld = np.zeros(6)

    # --- material data
    material = simdat.MATERIAL
    matdat = material.material_data()
    J0 = matdat.get_data("jacobian")

    # -------------------------------------------------------- initialize model
    # --- start output file and Mathematica files
    msg = "starting calculations for simulation {0}".format(the_model.name)
    pu.reportMessage(iam, msg)

    # --- call the material model with zero state
    if ileg == 0:
        material.update_state(simdat, matdat)

        # advance and write data
        simdat.advance_all_data()
        matdat.advance_all_data()
        the_model.write_state()

    # ----------------------------------------------------------------------- #

    # --------------------------------------------------- begin{processing leg}
    for leg in legs:

        # test restart capability
        if the_model.test_restart and not restart:
            if ileg == int(len(legs) / 2):
                print("\n\nStopping to test Payette restart capabilities.\n"
                      "Restart the simulation by executing\n\n"
                      "\t\trunPayette {0}\n\n".format(the_model.restart_file))
                the_model.finish()
                sys.exit(76)

        # read inputs and initialize for this leg
        lnum, t_end, nsteps, ltype, prdef = leg
        lns = len(str(nsteps))
        delt = t_end - t_beg
        if delt == 0.:
            continue
        nv, dflg = 0, list(set(ltype))
        nprints = simdat.NPRINTS if simdat.NPRINTS else nsteps
        if simdat.EMIT == "sparse":
            nprints = min(10,nsteps)
        print_interval = max(1, int(nsteps / nprints))

        # pass values from the end of the last leg to beginning of this leg
        eps_beg = matdat.get_data("strain")
        sig_beg = matdat.get_data("stress")
        F_beg = matdat.get_data("deformation gradient")
        v = matdat.get_data("prescribed stress components")
        if matdat.EFIELD_SIM:
            efld_beg = matdat.get_data("electric field")

        if len(v):
            prsig_beg = matdat.get_data("prescribed stress")
            j = 0
            for i in v:
                sig_beg[i] = prsig_beg[j]
                j += 1
                continue

        if verbose:
            pu.reportMessage(iam, cons_msg.format(lnum, lnl, 1, lns, t_beg, dt))

        # --- loop through components of prdef and compute the values at the end
        #     of this leg:
        #       for ltype = 1: eps_end at t_end -> eps_beg + prdef*delt
        #       for ltype = 2: eps_end at t_end -> prdef
        #       for ltype = 3: Pf at t_end -> sig_beg + prdef*delt
        #       for ltype = 4: Pf at t_end -> prdef
        #       for ltype = 5: F_end at t_end -> prdef
        #       for ltype = 6: efld_end at t_end-> prdef
        eps_end, F_end = pt.Z6, pt.I9
        if matdat.EFIELD_SIM:
            efld_end = pt.Z3

        # if stress is prescribed, we don't compute sig_end just yet, but sig_hld
        # which holds just those values of stress that are actually prescribed.
        for i in range(len(prdef)):

            if ltype[i] not in range(7):
                msg = ("Invalid load type (ltype) parameter "
                       "{0} prescribed for leg {1}".format(ltype[i], lnum))
                pu.reportError(iam, msg)
                return 1

            if ltype[i] == 0:
                continue

            # -- strain rate
            elif i < 6 and ltype[i] == 1:
                eps_end[i] = eps_beg[i] + prdef[i] * delt

            # -- strain
            elif i < 6 and ltype[i] == 2:
                eps_end[i] = prdef[i]

            # stress rate
            elif i < 6 and ltype[i] == 3:
                sig_hld[i] = sig_beg[i] + prdef[i] * delt
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
            elif matdat.EFIELD_SIM and (i >= 9 and ltype[i] == 6):
                efld_end[i - 9] = prdef[i]

            continue

        v = vdum[0:nv]
        matdat.advance_data("prescribed stress components", v)
        if len(v):
            prsig_beg, prsig_end = sig_beg[v], sig_hld[v]
            Js = J0[[[x] for x in v], v]

        t = t_beg
        dt = delt / nsteps

        # ---------------------------------------------- begin{processing step}
        for n in range(nsteps):

            t += dt
            simsteps += 1
            simdat.advance_data("number of steps", simsteps)

            # interpolate values of E, F, EF, and P for the current step
            a1 = float(nsteps - (n + 1)) / nsteps
            a2 = float(n + 1) / nsteps

            eps_int = a1 * eps_beg + a2 * eps_end

            F_int = a1 * F_beg + a2 * F_end

            if matdat.EFIELD_SIM:
                efld_int = a1 * efld_beg + a2 * efld_end

            if len(v):
                # prescribed stress components given, get initial guess for
                # depsdt[v]
                prsig_int = a1 * prsig_beg + a2 * prsig_end
                prsig_dif = prsig_int - matdat.get_data("stress")[v]
                depsdt = (eps_int - matdat.get_data("strain")) / dt
                try:
                    depsdt[v] = np.linalg.solve(Js, prsig_dif) / dt
                except:
                    depsdt[v] -= np.linalg.lstsq(Js, prsig_dif)[0] / dt

            # advance known values to end of step
            simdat.advance_data("time", t)
            simdat.advance_data("time step", dt)
            if matdat.EFIELD_SIM:
                matdat.advance_data("electric field", efld_int)

            # --- find d (symmetric part of velocity gradient)
            if not len(v):

                if dflg[0] == 5:
                    # --- deformation gradient prescribed
                    matdat.advance_data("prescribed deformation gradient", F_int)
                    pk.velGradCompFromF(simdat, matdat)

                else:
                    # --- strain or strain rate prescribed
                    matdat.advance_data("prescribed strain", eps_int)
                    pk.velGradCompFromE(simdat, matdat)

            else:
                # --- One or more stresses prescribed
                matdat.advance_data("strain rate", depsdt)
                matdat.advance_data("prescribed stress", prsig_int)
                pk.velGradCompFromP(simdat, matdat)
                matdat.advance_data("strain rate")
                depsdt = matdat.get_data("strain rate")
                matdat.store_data("rate of deformation", depsdt)

            # --- update the deformation to the end of the step at this point,
            #     the rate of deformation and vorticity to the end of the step
            #     are known, advance them.
            matdat.advance_data("rate of deformation")
            matdat.advance_data("vorticity")

            # find the current {deformation gradient,strain} and advance them
            pk.updateDeformation(simdat, matdat)
            matdat.advance_data("deformation gradient")
            matdat.advance_data("strain")
            matdat.advance_data("equivalent strain")

            if simdat.USE_TABLE:
                # use the actual table values, not the values computed above
                if dflg == [1] or dflg == [2] or dflg == [1, 2]:
                    matdat.advance_data("strain", eps_int)
                elif dflg[0] == 5:
                    matdat.advance_data("deformation gradient", F_int)

            # update material state
            material.update_state(simdat, matdat)

            # advance all data after updating state
            matdat.store_data("stress rate",
                             (matdat.get_data("stress", cur=True) -
                              matdat.get_data("stress")) / dt)
            matdat.advance_all_data()

            # --- write state to file
            if (nsteps-n)%print_interval == 0:
                the_model.write_state()

            if simdat.SCREENOUT or ( verbose and (2 * n - nsteps) == 0 ):
                pu.reportMessage(iam, cons_msg.format(lnum, lnl, n, lns, t, dt))

            # ------------------------------------------ begin{end of step SQA}
            if simdat.SQA:
                if dflg == [1] or dflg == [2] or dflg == [1,2]:
                    eps_tmp = matdat.get_data("strain")
                    max_diff = np.max(np.abs(eps_tmp - eps_int))
                    dnom = max(np.max(np.abs(eps_int)), 0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff / dnom
                    if rel_diff > pu.ACCLIM and max_diff > pu.EPSILON:
                        msg = ("E differs from prdef excessively at end of step "
                               "{0} of leg {1} with a percent difference of {2:f}"
                               .format(n, lnum, rel_diff * 100.))
                        pu.reportWarning(iam, msg)

                elif dflg == [5]:
                    F_tmp = matdat.get_data("deformation gradient")
                    max_diff = (np.max(F_tmp - F_int)) / np.max(np.abs(F_int))
                    dnom = max(np.max(np.abs(F_int)), 0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff / dnom
                    if rel_diff > pu.ACCLIM and max_diff > pu.EPSILON:
                        msg = ("F differs from prdef excessively at end of step "
                               " with max_diff {0:f}".format(max_diff))
                        pu.reportWarning(iam, msg)

            # -------------------------------------------- end{end of step SQA}

            continue # continue to next step
        # ------------------------------------------------ end{processing step}

        # --- pass quantities from end of leg to beginning of new leg
        simdat.advance_data("leg number", ileg + 1)
        ileg += 1

        if the_model.write_vandd_table:
            the_model.write_vel_and_disp(t_beg, t_end, eps_beg, eps_end)

        # advances time must come after writing the v & d tables above
        t_beg = t_end

        if the_model.write_restart:
            with open(the_model.restart_file, 'wb') as fobj:
                pickle.dump(the_model, fobj, 2)

        # --- print message to screen
        if verbose and nsteps > 1:
            pu.reportMessage(iam, cons_msg.format(lnum, lnl, n + 1, lns, t, dt))

        # ----------------------------------------------- begin{end of leg SQA}
        if simdat.SQA:
            if dflg == [1] or dflg == [2] or dflg == [1,2]:
                eps_tmp = matdat.get_data("strain")
                max_diff = np.max(np.abs(eps_tmp - eps_end))
                dnom = np.max(eps_end) if np.max(eps_end) >= pu.EPSILON else 1.
                rel_diff = max_diff / dnom
                if rel_diff > pu.ACCLIM and max_diff > pu.EPSILON:
                    msg = ("E differs from prdef excessively at end of "
                           "leg {0} with relative diff {1:f}"
                           .format(lnum, rel_diff))
                    pu.reportWarning(iam, msg)

            elif dflg == [5]:
                F_tmp = matdat.get_data("deformation gradient")
                max_diff = np.max(np.abs(F_tmp - F_end)) / np.max(np.abs(F_end))
                dnom = np.max(F_end) if np.max(F_end) >= pu.EPSILON else 1.
                rel_diff = max_diff / dnom
                if rel_diff > pu.ACCLIM and max_diff > pu.EPSILON:
                    msg = ("F differs from prdef excessively at end of leg "
                           " with max_diff {0:f}".format(max_diff))
                    pu.reportWarning(iam, msg)

        # ------------------------------------------------- end{end of leg SQA}

        continue # continue to next leg
    # ----------------------------------------------------- end{processing leg}

    return 0
