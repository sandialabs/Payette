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

import os
import sys
import math
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle

import Source.Payette_iterative_solvers as citer
import Source.Payette_kinematics as pk
import Source.Payette_utils as pu
import Source.Payette_tensor as pt
import Source.__runopts__ as ro
from Source.Payette_unit_manager import UnitManager as UnitManager

np.set_printoptions(precision=4)

ACCLIM = .0001 / 100.
EPSILON = np.finfo(np.float).eps

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

    # jrh: Added this dict to get extra file names
    extra_files = kwargs.get("extra_files")
    if extra_files is None:
        extra_files = {}
    extra_files['path files'] = {}

    simdat = the_model.simulation_data()
    material = the_model.material
    matdat = material.material_data()

    eos_model = material.constitutive_model

    # This will make sure that everything has units set.
    eos_model.ensure_all_parameters_have_valid_units()
    simdat.ensure_valid_units()
    matdat.ensure_valid_units()

    nprints = simdat.NPRINTS

    # get boundary data
    density_range = the_model.boundary.density_range()
    temperature_range = the_model.boundary.temperature_range()
    surface_increments = the_model.boundary.surface_increments()
    path_increments = the_model.boundary.path_increments()
    path_isotherm = the_model.boundary.path_isotherm()
    path_hugoniot = the_model.boundary.path_hugoniot()
    rho_temp_pairs = the_model.boundary.rho_temp_pairs()
    simdir = the_model.simdir
    simnam = the_model.name

    input_unit_system = the_model.boundary.input_units()
    output_unit_system = the_model.boundary.output_units()

    if not UnitManager.is_valid_unit_system(input_unit_system):
        pu.report_and_raise_error(
              "Input unit system '{0}' is not a valid unit system".
                                         format(input_unit_system))
    if not UnitManager.is_valid_unit_system(output_unit_system):
        pu.report_and_raise_error(
              "Output unit system '{0}' is not a valid unit system".
                                         format(output_unit_system))

    pu.log_message("Input unit system: {0}".format(input_unit_system))
    pu.log_message("Output unit system: {0}".format(output_unit_system))

#    print(dir(the_model.boundary))
#    print(dir(matdat))
#    print(dir(simdat))
#    print(dir(eos_modeldat))

################################################################################
###############             DENSITY-TEMPERATURE LEGS             ###############
################################################################################
    if len(rho_temp_pairs) != 0:
        out_fnam = os.path.join(simdir, simnam + ".out")
        the_model._setup_out_file(out_fnam)

        for pair in rho_temp_pairs:
            eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                   rho = pair[0], temp = pair[1])
            matdat.advance_all()

            the_model.write_state(input_unit_system=input_unit_system,
                                  output_unit_system=output_unit_system)
        pu.log_message("Legs file: {0}".format(out_fnam))



################################################################################
###############                     SURFACE                      ###############
################################################################################
    if (surface_increments is not None and
        density_range is not None and
        temperature_range is not None):

        t_range = temperature_range
        rho_range = density_range
        surf_incr = surface_increments

        out_fnam = os.path.join(simdir, simnam + ".surface")
        the_model._setup_out_file(out_fnam)

        pu.log_message("=" * (80 - 6))
        pu.log_message("Begin surface")
        DEJAVU = False
        idx = 0
        for rho in np.linspace(rho_range[0], rho_range[1], surf_incr):
            for temp in np.linspace(t_range[0], t_range[1], surf_incr):
                idx += 1
                if idx%int(surf_incr**2/float(nprints)) == 0:
                    pu.log_message(
                        "Surface step {0}/{1}".format(idx, surf_incr ** 2))

                eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                       rho=rho, temp=temp)

                matdat.advance_all()
                the_model.write_state(input_unit_system=input_unit_system,
                                      output_unit_system=output_unit_system)

        pu.log_message("End surface")
        pu.log_message("Surface file: {0}".format(out_fnam))
        extra_files['surface file'] = out_fnam

################################################################################
###############                     ISOTHERM                     ###############
################################################################################

    if (path_increments is not None and
        path_isotherm is not None and
        density_range is not None):

        # isotherm = [density, temperature]
        isotherm = path_isotherm
        rho_range = density_range
        path_incr = path_increments

        if not rho_range[0] <= isotherm[0] <= rho_range[1]:
            pu.report_and_raise_error(
                "initial isotherm density not within range")

        out_fnam = os.path.join(simdir, simnam + ".isotherm")
        the_model._setup_out_file(out_fnam)

        pu.log_message("=" * (80 - 6))
        pu.log_message("Begin isotherm")
        DEJAVU = False
        idx = 0
        for rho in np.linspace(isotherm[0], rho_range[1], path_incr):
            idx += 1
            if idx%int(path_incr/float(nprints)) == 0:
                pu.log_message("Isotherm step {0}/{1}".format(idx,path_incr))

            eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                   rho = rho, temp = isotherm[1])

            matdat.advance_all()
            the_model.write_state(input_unit_system=input_unit_system,
                                  output_unit_system=output_unit_system)

        pu.log_message("End isotherm")
        pu.log_message("Isotherm file: {0}".format(out_fnam))
        extra_files['path files']['isotherm'] = out_fnam

################################################################################
###############                     HUGONIOT                     ###############
################################################################################

    if (path_increments is not None and
        path_hugoniot is not None and
        temperature_range is not None and
        density_range is not None):

        # hugoniot = [density, temperature]
        hugoniot = path_hugoniot
        rho_range = density_range
        t_range = temperature_range
        path_incr = path_increments


        if not rho_range[0] <= hugoniot[0] <= rho_range[1]:
            report_and_raise_error(
                "initial hugoniot density not within range")
        if not t_range[0] <= hugoniot[1] <= t_range[1]:
            report_and_raise_error(
                "initial hugoniot temperature not within range")

        out_fnam = os.path.join(simdir, simnam + ".hugoniot")
        the_model._setup_out_file(out_fnam)

        pu.log_message("=" * (80 - 6))
        pu.log_message("Begin Hugoniot")

        init_density_MKSK = hugoniot[0]
        init_temperature_MKSK = hugoniot[1]

        # Convert to CGSEV
        init_density = init_density_MKSK
        init_temperature = init_temperature_MKSK

        eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                               rho = init_density, temp = init_temperature)

        init_energy = matdat.get("energy")
        init_pressure = matdat.get("pressure")

        idx = 0
        DEJAVU = False
        e = init_energy
        for rho in np.linspace(hugoniot[0], rho_range[1], path_incr):
            idx += 1
            if idx%int(path_incr/float(nprints)) == 0:
                pu.log_message("Hugoniot step {0}/{1}".format(idx,path_incr))
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

            r = rho
            a = (1./init_density - 1./r)/2.

            converged_idx = 0
            CONVERGED = False
            while not CONVERGED:
                converged_idx += 1
                eos_model.evaluate_eos(simdat, matdat, input_unit_system,
                                       rho = r, enrg = e)

                f = (matdat.get("pressure") + init_pressure)*a - e + init_energy
                df = matdat.get("dpdt")/matdat.get("dedt")*a - 1.0
                e = e - f/df
                errval = abs(f/init_energy)
                if errval < 1.0e-9:
                    CONVERGED = True
                    if converged_idx > 100:
                        pu.log_message("Max iterations reached (tol = {0:14.10e}).\n".format(1.0e-9)+
                               "rel error   = {0:14.10e}\n".format(float(errval))+
                               "abs error   = {0:14.10e}\n".format(float(f))+
                               "func val    = {0:14.10e}\n".format(float(f))+
                               "init_energy = {0:14.10e}\n".format(float(init_energy)))
                        break

            matdat.advance_all()
            the_model.write_state(input_unit_system=input_unit_system,
                                  output_unit_system=output_unit_system)

        pu.log_message("End Hugoniot")
        pu.log_message("Hugoniot file: {0}".format(out_fnam))
        extra_files['path files']['hugoniot'] = out_fnam

    return 0


def solid_driver(the_model, **kwargs):
    """Run the single element simulation for an instance of the main Payette
    class the_model

    Parameters
    ----------
    the_model : object
      Payette class instance

    Returns
    -------
    None

    Updates
    -------
    the_model : object
      Payette class instance
    outfile : file object
      simulation output file

    Strategy
    --------

    """
    cons_msg = "leg {0:{1}d}, step {2:{3}d}, time {4:.4E}, dt {5:.4E}"

    # -------------------- initialize and copy passed values to local variables
    try:
        restart = kwargs["restart"]
    except KeyError:
        restart = False

    # --- simulation data
    simdat = the_model.simulation_data()

    if ro.DEBUG:
        pdb = __import__('pdb')

    # --- options
    verbose = ro.VERBOSITY > 0

    # --- data
    ileg = int(simdat.get("leg number"))
    legs = the_model.boundary.legs(ileg)
    lnl = len(str(len(legs)))
    simsteps = simdat.get("number of steps")
    t_beg = simdat.get("time")
    dt = simdat.get("time step")
    Vhld = np.zeros(6, dtype=int)
    Ph = np.zeros(6)
    K = simdat.KAPPA

    # --- material data
    material = the_model.material
    matdat = material.material_data()

    # -------------------------------------------------------- initialize model
    # --- log information to user for this run
    pu.log_message(
        "starting calculations for simulation {0}".format(the_model.name))
    if ro.VERBOSITY > 3:
        if ro.SQA:
            pu.log_message("performing additional SQA checks")
        if ro.USE_TABLE:
            pu.log_message("using table values for prescribed deformations")
        if ro.TESTRESTART and not restart:
            pu.log_message("testing restart capability")
        if ro.WRITERESTART:
            pu.log_message("restart file will be generated")
        if ro.WRITE_VANDD_TABLE:
            pu.log_message("velocity and displacement tables will be generated")

    # --- call the material model with zero state
    if ileg == 0:
        material.update_state(simdat, matdat)
        the_model.write_state()

    # ----------------------------------------------------------------------- #

    # V is an array of integers that contains the columns of prescribed stress
    V = []
    Pt = np.zeros(pt.NSYM)

    Ec = matdat.get("strain")
    Fc = matdat.get("deformation gradient")
    Pc = matdat.get("stress")
    R, dR = matdat.get("rotation"), matdat.get("rotation rate")

    # --------------------------------------------------- begin{processing leg}
    for leg in legs:

        # test restart capability
        if ro.TESTRESTART and not restart:
            if ileg == int(len(legs) / 2):
                print("\n\nStopping to test Payette restart capabilities.\n"
                      "Restart the simulation by executing\n\n"
                      "\t\tpayette {0}\n\n".format(the_model.restart_file))
                the_model.finish()
                sys.exit(76)

        # read inputs and initialize for this leg
        lnum, t_end, nsteps, ltype, prdef = leg
        ltype = [int(x) for x in ltype]
        lns = len(str(nsteps))
        delt = t_end - t_beg
        if delt == 0.:
            continue
        nv, dflg = 0, list(set(ltype))
        for inum in (0, 6):
            try:
                dflg.remove(inum)
            except ValueError:
                pass

        nprints = (simdat.NPRINTS if simdat.NPRINTS else nsteps)
        if simdat.EMIT == 0:
            nprints = min(10, nsteps)
        print_interval = max(1, int(nsteps / nprints))

        # pass values from the end of the last leg to beginning of this leg
        E0 = matdat.get("strain", copy=True)
        P0 = matdat.get("stress", copy=True)
        F0 = matdat.get("deformation gradient", copy=True)
        if ro.EFIELD_SIM: EF0 = matdat.get("electric field", copy=True)

        if len(V):
            PS0 = Pt
            for i, j in enumerate(V):
                P0[i] = PS0[j]
                continue

        if verbose:
            pu.log_message(cons_msg.format(lnum, lnl, 1, lns, t_beg, dt))

        # --- loop through components of prdef and compute the values at the end
        #     of this leg:
        #       for ltype = 1: Ef at t_end -> E0 + prdef*delt
        #       for ltype = 2: Ef at t_end -> prdef
        #       for ltype = 3: Pf at t_end -> P0 + prdef*delt
        #       for ltype = 4: Pf at t_end -> prdef
        #       for ltype = 5: Ff at t_end -> prdef
        #       for ltype = 6: EFf at t_end-> prdef
        Ef, Ff = pt.Z6, pt.I9
        if ro.EFIELD_SIM:
            EFf = pt.Z3

        # if stress is prescribed, we don't compute Pf just yet, but Ph
        # which holds just those values of stress that are actually prescribed.
        for i in range(len(prdef)):

            if ltype[i] not in range(7):
                msg = ("Invalid load type (ltype) parameter "
                       "{0} prescribed for leg {1}".format(ltype[i], lnum))
                pu.report_and_raise_error(msg)
                return 1

            if ltype[i] == 0:
                continue

            # -- strain rate
            elif i < 6 and ltype[i] == 1:
                Ef[i] = E0[i] + prdef[i] * delt

            # -- strain
            elif i < 6 and ltype[i] == 2:
                Ef[i] = prdef[i]

            # stress rate
            elif i < 6 and ltype[i] == 3:
                Ph[i] = P0[i] + prdef[i] * delt
                Vhld[nv] = i
                nv += 1

            # stress
            elif i < 6 and ltype[i] == 4:
                Ph[i] = prdef[i]
                Vhld[nv] = i
                nv += 1

            # deformation gradient
            elif ltype[i] == 5:
                Ff[i] = prdef[i]

            # electric field
            elif ro.EFIELD_SIM and (i >= 6 and ltype[i] == 6):
                EFf[i - 9] = prdef[i]

            continue

        V = Vhld[0:nv]
        if len(V):
            PS0, PSf = P0[V], Ph[V]

        t = t_beg
        dt = delt / nsteps

        # ---------------------------------------------- begin{processing step}
        for n in range(nsteps):

            t += dt
            simsteps += 1
            simdat.store("number of steps", simsteps)

            # interpolate values of E, F, EF, and P for the target values for
            # this step
            a1 = float(nsteps - (n + 1)) / nsteps
            a2 = float(n + 1) / nsteps
            Et = a1 * E0 + a2 * Ef
            Ft = a1 * F0 + a2 * Ff
            if ro.EFIELD_SIM:
                EFt = a1 * EF0 + a2 * EFf

            if len(V):
                # prescribed stress components given
                Pt = a1 * P0 + a2 * PSf # target stress

            # advance known values to end of step
            simdat.store("time", t)
            simdat.store("time step", dt)
            if ro.EFIELD_SIM:
                matdat.store("electric field", EFt)

            # --- find current value of d: sym(velocity gradient)
            if not len(V):

                if dflg[0] == 5:
                    # --- deformation gradient prescribed
                    Dc, Wc = pk.velgrad_from_defgrad(dt, Fc, Ft)

                else:
                    # --- strain or strain rate prescribed
                    Dc, Wc = pk.velgrad_from_strain(dt, K, Ec, R, dR, Et)

            else:
                # --- One or more stresses prescribed
                Dc, Wc = pk.velgrad_from_stress(
                    material, simdat, matdat, dt, Ec, Et, Pc, Pt, V)

            if not ro.USE_TABLE:
                # compute the current deformation gradient and strain from
                # previous values and the deformation rate
                Fc, Ec = pk.update_deformation(dt, K, Fc, Dc, Wc)

            else:
                # use the actual table values, not the values computed above
                if dflg == [1] or dflg == [2] or dflg == [1, 2]:
                    if ro.VERBOSITY > 3:
                        pu.log_message("retrieving updated strain from table")
                elif dflg[0] == 5:
                    if ro.VERBOSITY > 3:
                        pu.log_message("retrieving updated deformation "
                                       "gradient from table")
                Ec, Fc = Et, Ft

            # --- update the deformation to the end of the step at this point,
            #     the rate of deformation and vorticity to the end of the step
            #     are known, advance them.
            matdat.store("rate of deformation", Dc)
            matdat.store("vorticity", Wc)
            matdat.store("deformation gradient", Fc)
            matdat.store("strain", Ec)
            # compute the equivalent strain
            matdat.store(
                "equivalent strain",
                np.sqrt(2. / 3. * (sum(Ec[:3] ** 2) + 2. * sum(Ec[3:] ** 2))))


            # udpate density
            dev = pt.trace(matdat.get("rate of deformation")) * dt
            rho_old = simdat.get("payette density")
            simdat.store("payette density", rho_old * math.exp(-dev))

            # update material state
            material.update_state(simdat, matdat)

            # advance all data after updating state
            Pc = matdat.get("stress")
            dsig = (Pc - matdat.get("stress", "-")) / dt
            matdat.store("stress rate", (Pc - matdat.get("stress", "-")) / dt)
            matdat.store("pressure", -np.sum(Pc[:3]) / 3.)

            matdat.advance()
            simdat.advance()

            # --- write state to file
            if (nsteps-n)%print_interval == 0:
                the_model.write_state()

            if simdat.SCREENOUT or (verbose and (2 * n - nsteps) == 0):
                pu.log_message(cons_msg.format(lnum, lnl, n, lns, t, dt))

            # ------------------------------------------ begin{end of step SQA}
            if ro.SQA:

                if dflg == [1] or dflg == [2] or dflg == [1, 2]:
                    if ro.VERBOSITY > 3:
                        pu.log_message("checking that computed strain matches "
                                       "at end of step prescribed strain")
                    Etmp = matdat.get("strain")
                    max_diff = np.max(np.abs(Etmp - Ec))
                    dnom = max(np.max(np.abs(Ec)), 0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff / dnom
                    if rel_diff > ACCLIM:
                        msg = ("E differs from prdef excessively at end of step "
                               "{0} of leg {1} with a percent difference of {2:f}"
                               .format(n, lnum, rel_diff * 100.))
                        pu.log_warning(msg)
                    elif ro.VERBOSITY > 3:
                        pu.log_message("computed strain at end of step matches "
                                       "prescribed strain")

                elif dflg == [5]:
                    if ro.VERBOSITY > 3:
                        pu.log_message("checking that computed deformation "
                                       "gradient at end of step matches "
                                       "prescribed deformation gradient")
                    F_tmp = matdat.get("deformation gradient")
                    max_diff = (np.max(F_tmp - Fc)) / np.max(np.abs(Fc))
                    dnom = max(np.max(np.abs(Fc)), 0.)
                    dnom = dnom if dnom != 0. else 1.
                    rel_diff = max_diff / dnom
                    if rel_diff > ACCLIM:
                        msg = ("F differs from prdef excessively at end of "
                               "step {0:d} with relative diff {1:2f}%"
                               .format(n, rel_diff * 100.))
                        pu.log_warning(msg)
                    elif ro.VERBOSITY > 3:
                        pu.log_message("computed deformation gradient "
                                       "at end of step matches prescribed "
                                       "deformation gradient")

            # -------------------------------------------- end{end of step SQA}

            ro.ISTEP += 1

            continue # continue to next step
        # ------------------------------------------------ end{processing step}

        # --- pass quantities from end of leg to beginning of new leg
        ileg += 1
        simdat.store("leg number", ileg)

        if ro.WRITE_VANDD_TABLE:
            the_model.write_vel_and_disp(t_beg, t_end, E0, Ef)

        # advances time must come after writing the v & d tables above
        t_beg = t_end

        if ro.WRITERESTART:
            with open(the_model.restart_file, 'wb') as fobj:
                pickle.dump(the_model, fobj, 2)

        # --- print message to screen
        if verbose and nsteps > 1:
            pu.log_message(cons_msg.format(lnum, lnl, n + 1, lns, t, dt))

        # ----------------------------------------------- begin{end of leg SQA}
        if ro.SQA:
            if dflg == [1] or dflg == [2] or dflg == [1, 2]:
                if ro.VERBOSITY > 3:
                    pu.log_message("checking that computed strain at end of "
                                   "leg matches prescribed strain")
                Et = matdat.get("strain")
                max_diff = np.max(np.abs(Et - Ef))
                dnom = np.max(Ef) if np.max(Ef) >= EPSILON else 1.
                rel_diff = max_diff / dnom
                if rel_diff > ACCLIM:
                    msg = ("E differs from prdef excessively at end of "
                           "leg {0} with relative diff {1:2f}%"
                           .format(lnum, rel_diff * 100.))
                    pu.log_warning(msg)
                elif ro.VERBOSITY > 3:
                    pu.log_message("computed strain at end of leg matches "
                                   "prescribed strain")

            elif dflg == [5]:
                if ro.VERBOSITY > 3:
                    pu.log_message("checking that computed deformation gradient "
                                   "at end of leg matches prescribed "
                                   "deformation gradient")
                F_tmp = matdat.get("deformation gradient")
                max_diff = np.max(np.abs(F_tmp - Ff)) / np.max(np.abs(Ff))
                dnom = np.max(Ff) if np.max(Ff) >= EPSILON else 1.
                rel_diff = max_diff / dnom
                if rel_diff > ACCLIM:
                    msg = ("F differs from prdef excessively at end of "
                           "leg {0:d} with relative diff {1:2f}%"
                           .format(lnum, rel_diff * 100.))
                    pu.log_warning(msg)
                elif ro.VERBOSITY > 3:
                    pu.log_message("computed deformation gradient "
                                   "at end of leg matches prescribed "
                                   "deformation gradient")

        # ------------------------------------------------- end{end of leg SQA}

        continue # continue to next leg
    # ----------------------------------------------------- end{processing leg}

    return 0
