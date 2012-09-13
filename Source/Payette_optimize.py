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
"""Payette_optimize.py module. Contains classes and functions for optimizing
parameters using Payette.

"""

import os
import sys
import imp
import shutil
import numpy as np
import scipy
import scipy.optimize
import math
from copy import deepcopy

import Source.Payette_utils as pu
import Source.Payette_container as pc
import Source.Payette_extract as pe
import Source.Payette_input_parser as pip
import Source.runopts as ro
import Source.Payette_sim_index as psi
import Toolset.KayentaParamConv as kpc

# Module level variables
IOPT = -1
FAC = []
FNEWEXT = ".0x312.gold"
UM = None


class Optimize(object):
    r"""docstring -> needs to be completed """

    def __init__(self, input_lines):
        r""" Initialization """

        # get the optimization block
        job_inp = pip.InputParser(input_lines)
        job = job_inp.get_simulation_key()

        if "simdir" in job_inp.input_options() or ro.SIMDIR is not None:
            pu.report_and_raise_error(
                "cannot specify simdir for permutation jobs")

        optimize = job_inp.get_block("optimization")

        self.name = job

        # save Perturbate information to single "data" dictionary
        self.data = {}
        self.data["basename"] = job
        self.data["verbosity"] = ro.VERBOSITY
        self.data["return level"] = ro.DISP
        self.data["fext"] = ".opt"
        self.data["options"] = []

        # fill the data with the optimization information
        self.parse_optimization_block(optimize)

        input_lines = job_inp.get_input_lines(skip="optimization")
        self.data["baseinp"] = input_lines

        # set the loglevel to 0 for Payette simulation and save the payette
        # options to the data dictionary
        ro.set_global_option("VERBOSITY", 0, default=True)

        # check user input for required blocks
        if not job_inp.has_block("material"):
            pu.report_and_raise_error(
                "material block not found in input file")

        # check the optimization variables
        self.check_params()

        if self.data["verbosity"]:
            pu.log_message("Optimizing {0}".format(job), noisy=True)
            pu.log_message("Optimization variables: {0}"
                           .format(", ".join(self.data["optimize"])),
                           noisy=True)
            pu.log_message("Objective function file: {0}".format(UM.__file__),
                           noisy=True)
            if self.data["options"]:
                pu.log_message("Optimization options: {0}"
                               .format(", ".join(self.data["options"])),
                               noisy=True)
            pu.log_message("Optimization method: {0}"
                           .format(self.data["optimization method"]["method"]),
                           noisy=True)

    def run_job(self, *args, **kwargs):
        r"""Run the optimization job

        Set up directory to run the optimization job and call the minimizer

        Parameters
        ----------
        None

        Returns
        -------
        opt_params : array_like
            The optimized parameters

        """

        # make the directory to run the job
        cwd = os.getcwd()
        dnam = self.data["basename"] + self.data["fext"]
        base_dir = os.path.join(cwd, dnam)
        if not ro.KEEP:
            try:
                shutil.rmtree(base_dir)
            except OSError:
                pass
        elif os.path.isdir(base_dir):
            idir = 0
            dir_id = ".{0:03d}".format(idir)
            copy_dir = os.path.join(cwd, dnam + dir_id)
            while True:
                if os.path.isdir(copy_dir):
                    dir_id = ".{0:03d}".format(idir)
                    copy_dir = os.path.join(cwd, dnam + dir_id)
                else:
                    break

                idir += 1
                if idir > 100:
                    pu.report_and_raise_error("max number of dirs")
            os.rename(base_dir, copy_dir)

        if self.data["verbosity"]:
            pu.log_message("Running: {0}".format(self.data["basename"]),
                           noisy=True)

        os.mkdir(base_dir)
        os.chdir(base_dir)

        # open up the index file
        self.index = psi.SimulationIndex(base_dir)

        # copy gold file to base_dir

        # Initial guess for opt_params are those from input. opt_params must
        # be in a consistent order throughout, here we arbitrarily set that
        # order to be based on an alphabetical sort of the names of the
        # parameters to be optimized.
        opt_nams, opt_params, opt_bounds = [], [], []
        for nam in sorted(self.data["optimize"]):
            val = self.data["optimize"][nam]
            opt_nams.append(nam)
            opt_params.append(val["initial value"])
            opt_bounds.append(val["bounds"])
            continue

        # set up args and call optimzation routine
        opt_args = [opt_nams, self.data, base_dir, self.index]
        opt_method = self.data["optimization method"]["method"]
        opt_options = {"maxiter": self.data["maximum iterations"],
                       "xtol": self.data["tolerance"],
                       "ftol": self.data["tolerance"],
                       "disp": self.data["disp"],}

        opt_params = minimize(
            func, opt_params, args=opt_args, method=opt_method,
            bounds=opt_bounds, options=opt_options,
            )

        # optimum parameters found, write out final info
        msg = ", ".join(["{0} = {1:12.6E}".format(opt_nams[i], x)
                         for i, x in enumerate(opt_params)])

        pu.log_message("Optimized parameters found on iteration {0:d}"
                       .format(IOPT + 1),
                       noisy=True)
        pu.log_message("Optimized parameters: {0}".format(msg),
                       noisy=True)

        # last_job = os.path.join(base_dir,
        #                         self.data["basename"] + ".{0:03d}".format(IOPT))
        # pu.log_message("Ultimate simulation directory: {0}".format(last_job))

        # write out the optimized parameters
        opt_f = os.path.join(base_dir, self.data["basename"] + ".opt")
        with open(opt_f, "w") as fobj:
            fobj.write("Optimized parameters\n")
            for idx, nam in enumerate(opt_nams):
                opt_val = opt_params[idx]
                fobj.write("{0} = {1:12.6E}\n".format(nam, opt_val))
                continue

        os.chdir(cwd)
        retcode = 0

        if not self.data["return level"]:
            return retcode
        else:
            return {"retcode": retcode,
                    "index file": self.index.index_file(),
                    "simulation name": self.name}

    def finish(self):
        r""" finish up the optimization job """

        global IOPT, FAC, FNEWEXT

        # remove any temporary files
        for item in [x for x in os.listdir(os.getcwd()) if x.endswith(FNEWEXT)]:
            os.remove(item)
            continue

        # restore global params
        IOPT = -1
        FAC = []
        FNEWEXT = ".0x312.gold"

        self.index.dump()

        return

    def parse_optimization_block(self, opt_block):
        r"""Get the required optimization information.

        Populates the self.data dict with information parsed from the
        optimization block of the input file. Sets defaults where not
        specified.

        Parameters
        ----------
        opt_block : array_like
            optimization block from input file

        Returns
        -------
        None

        """
        global UM
        allowed_methods = {
            "simplex": {"method": "Nelder-Mead", "name": "fmin"},
            "powell": {"method": "Powell", "name": "fmin_powell"},
            "cobyla": {"method": "COBYLA", "name": "fmin_cobyla"},
            "slsqp": {"method": "SLSQP", "name":"fmin_slsqp"}}

        # default method
        opt_method = allowed_methods["simplex"]
        maxiter = 20
        tolerance = 1.e-4
        disp = False
        optimize = {}

        # get options first -> must come first because some options have a
        # default method different than the global default of simplex
        for item in opt_block:
            for pat, repl in ((",", " "), ("=", " "), ):
                item = item.replace(pat, repl)
            item = item.split()

            if "option" in item[0].lower():
                self.data["options"].append(item[1].lower())

            continue

        # get method before other options
        for item in opt_block:
            for pat, repl in ((",", " "), ("=", " "), ):
                item = item.replace(pat, repl)
            item = item.split()
            if "method" in item[0].lower():
                opt_method = allowed_methods.get(item[1].lower())
                if opt_method is None:
                    pu.report_error(
                        "invalid method {0}".format(item[1].lower()))

        if pu.error_count():
            pu.report_and_raise_error("stopping due to previous errors")

        # now get the rest
        for item in opt_block:
            for pat, repl in ((",", " "), ("=", " "), ("(", " "), (")", " "),):
                item = item.replace(pat, repl)
            item = item.split()

            if "optimize" in item[0].lower():
                # set up this parameter to optimize
                key = item[1]
                vals = item[2:]

                bounds = (None, None)
                init_val = None

                # upper bound
                if "bounds" in vals:
                    try:
                        idx = vals.index("bounds") + 1
                        bounds = [float(x) for x in vals[idx:idx+2]]
                    except ValueError:
                        pu.report_error("bounds requires 2 arguments")

                    if bounds[0] > bounds[1]:
                        pu.report_error(
                            "lower bound {0} > upper bound {1} for {2}"
                            .format(bounds[0], bounds[1], key))

                if "initial" in vals:
                    idx = vals.index("initial")
                    if vals[idx + 1] == "value":
                        idx = idx + 1
                        init_val = float(vals[idx+1])

                if "lbound" in vals:
                    pu.report_error("depricated keyword 'lbound'")

                if "ubound" in vals:
                    pu.report_error("depricated keyword 'lbound'")

                optimize[key] = {}
                optimize[key]["bounds"] = bounds
                optimize[key]["initial value"] = init_val

            elif "obj_fn" in item[0].lower():
                # user supplied objective function
                fnam = item[-1]
                if not os.path.isfile(fnam):
                    pu.report_error("obj_fn {0} not found".format(fnam))
                    continue

                if os.path.splitext(fnam)[1] not in (".py",):
                    pu.report_error("obj_fn {0} must be .py file".format(fnam))
                    continue

                pymod, pypath = pu.get_module_name_and_path(fnam)
                try:
                    fobj, path, desc = imp.find_module(pymod, pypath)
                    UM = imp.load_module(pymod, fobj, path, desc)
                    fobj.close()
                except ImportError:
                    pu.report_error("obj_fn {0} not imported".format(fnam))
                    continue

                if not hasattr(UM, "obj_fn"):
                    pu.report_error("obj_fn must define a 'obj_fn' function")

            # below are options for the scipy optimizing routines
            elif "maxiter" in item[0].lower():
                maxiter = int(item[1])

            elif "tolerance" in item[0].lower():
                tolerance = float(item[1])

            elif "disp" in item[0].lower():
                disp = item[1].lower()
                if disp == "true":
                    disp = True

                elif disp == "false":
                    disp = False

                else:
                    try:
                        disp = bool(float(item[1]))

                    except ValueError:
                        # the user specified disp in the input file, it is not
                        # one of false, true, and is not a number. assume that
                        # since it was set, the user wants disp to be true.
                        disp = True

            continue

        if not optimize:
            pu.report_error("no parameters to optimize given")

        for key, val in optimize.items():
            if val["initial value"] is None:
                pu.report_error("no initial value given for {0}".format(key))

        if pu.error_count():
            pu.report_and_raise_error("stopping due to previous errors")

        self.data["optimize"] = optimize
        self.data["maximum iterations"] = maxiter
        self.data["tolerance"] = tolerance
        self.data["optimization method"] = opt_method
        self.data["disp"] = disp

        return

    def check_params(self):
        r"""Check that the minimization parameters were specified in the input
        file and exist in the parameter table for the instantiated material.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        # Remove from the material block the optimized parameters. Current
        # values will be inserted in to the material block for each run.
        preprocessing = []
        pu.log_message("Calling model with initial parameters")
        for key, val in self.data["optimize"].items():
            preprocessing.append(
                "{0} = {1}".format(key, val["initial value"]))
        job_inp = pip.preprocess_input_deck(self.data["baseinp"],
                                            preprocessing=preprocessing)
        the_model = pc.Payette(job_inp)
        if pu.warn_count():
            pu.report_and_raise_error("exiting due to initial warnings")

        return


def minimize(fcn, x0, args=(), method="Nelder-Mead",
             bounds=None, options={}):
    r"""Wrapper to the supported minimization methods

    Parameters
    ----------
    fcn : callable
        Objective function.
    x0 : ndarray
        Initial guess.
    data : dict
        Data container for problem
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives
        (Jacobian, Hessian).
    method : str, optional
        Type of solver. Should be one of:
            {"Nelder-Mead", "Powell", "COBYLA"}
    bounds : sequence, optional
        Bounds for variables (only for COBYLA). (min, max) pairs for each
        element in x, defining the bounds on that parameter. Use None for one
        of min or max when there is no bound in that direction.
    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:
        maxiter : int
            Maximum number of iterations to perform.
        xtol : float
            Tolerance on x
        ftol : float
            Tolerance on func
        disp : bool
            Set to True to print convergence messages.

    Returns
    -------
    xopt : ndarray
        The solution.

    """

    global FAC
    meth = method.lower()

    # set up args and call optimzation routine
    maxiter = options["maxiter"]
    xtol = options["xtol"]
    ftol = options["ftol"]
    disp = options["disp"]

    # optimization methods work best with number around 1, here we
    # normalize the optimization variables and save the multiplier to be
    # used when the function gets called by the optimizer.
    for val in x0:
        mag_val = eval("1.e" + "{0:12.6E}".format(val).split("E")[1])
        FAC.append(mag_val)
        continue
    FAC = np.array(FAC)
    x0 = x0 / FAC

    has_bounds = [x for j in bounds for x in j if x is not None]
    if meth in ["nelder-mead", "powell"] and has_bounds:
        pu.log_warning(
            "Method {0} cannot handle constraints nor bounds directly."
            .format(method))

    if has_bounds:
        # user has specified bounds on the parameters to be optimized. Here,
        # we convert the bounds to inequality constraints
        lcons, ucons = [], []
        for ibnd, bound in enumerate(bounds):
            lbnd, ubnd = bound
            if lbnd is None:
                lbnd = -1.e20
            if ubnd is None:
                ubnd = 1.e20

            if lbnd > ubnd:
                pu.report_error("lbnd({0:12.6E}) > ubnd({1:12.6E})"
                                .format(lbnd, ubnd))

            lcons.append(lambda z, idx=ibnd, bnd=lbnd: z[idx] - bnd/FAC[idx])
            ucons.append(lambda z, idx=ibnd, bnd=ubnd: bnd/FAC[idx] - z[idx])

            bounds[ibnd] = (lbnd, ubnd)

            continue

        if pu.error_count():
            pu.report_and_raise_error("ERROR: Resolve previous errors")

        cons = lcons + ucons

    if meth == "nelder-mead":
        xopt = scipy.optimize.fmin(
            fcn, x0, xtol=xtol, ftol=ftol,
            args=args, maxiter=maxiter, disp=disp)

    elif meth == "powell":
        xopt = scipy.optimize.fmin_powell(
            fcn, x0, xtol=xtol, ftol=ftol,
            args=args, maxiter=maxiter, disp=disp)

    elif meth == "cobyla":
        xopt = scipy.optimize.fmin_cobyla(
            fcn, x0, cons, consargs=(),
            args=args, disp=disp)

    else:
        pu.report_and_raise_error(
            "ERROR: Unrecognized method {0}".format(method))

    return xopt * FAC


def func(xcall, xnams, data, base_dir, index):

    r"""Objective function

    Creates a directory to run the current job, runs the job through Payette
    and then gets the average normalized root mean squared error between the
    output and the gold file.

    Parameters
    ----------
    xcall : array_like
        Current best guess for optimized parameters
    data : dict
        Optimization class data container

    Returns
    -------
    error : float
        Average root mean squared error between the out file and gold file

    """

    global IOPT
    IOPT += 1

    job = data["basename"] + ".{0:03d}".format(IOPT)

    job_dir = os.path.join(base_dir, job)
    os.mkdir(job_dir)
    os.chdir(job_dir)

    # replace the optimize variables with the updated and write params to file
    msg = []
    preprocessing = []
    variables = {}
    with open(os.path.join(job_dir, job + ".opt"), "w") as fobj:
        fobj.write("Parameters for iteration {0:d}\n".format(IOPT + 1))
        for idx, item in enumerate(zip(xnams, xcall)):
            nam, opt_val = item
            # Some methods do not allow for bounds and we can get negative
            # trial values. This is a problem when optimizing, say, elastic
            # moduli that cannot be negative since if we send a negative
            # elastic modulus to the routine it will bomb and the optimization
            # will stop. This is a way of forcing the optimizer to see a very
            # large error if it tries to send in numbers above or below the
            # user specified bounds -> essentially, we are using a penalty
            # method of sorts to force the bounds we want.
            lbnd, ubnd = data["optimize"][nam]["bounds"]
            if lbnd is not None and opt_val < lbnd/FAC[idx]:
                return 1.e3

            if ubnd is not None and opt_val > ubnd/FAC[idx]:
                return 1.e3

            pstr = "{0} = {1:12.6E}".format(nam, opt_val * FAC[idx])
            preprocessing.append(
                "{0} = {1}".format(nam, opt_val * FAC[idx]))
            variables[nam] = opt_val * FAC[idx]
            fobj.write(pstr + "\n")
            msg.append(pstr)
            continue

    job_inp = pip.preprocess_input_deck(data["baseinp"],
                                        preprocessing=preprocessing)
    job_inp[0] = "begin simulation {0}".format(job)

    if data["verbosity"]:
        pu.log_message("Iteration {0:03d}, trial parameters: {1}"
                       .format(IOPT + 1, ", ".join(msg)),
                       noisy=True)

    # instantiate Payette object
    the_model = pc.Payette(job_inp)

    # run the job
    solve = the_model.run_job()

    # store the data to the index
    index.store(int(IOPT), the_model.name, the_model.simdir,
                variables, the_model.outfile)

    the_model.finish()

    if ro.DISP:
        retcode = solve["retcode"]
    else:
        retcode = solve

    if retcode != 0:
        pu.report_and_raise_error("simulation failed")

    # extract minimization variables from the simulation output
    out_f = os.path.join(job_dir, job + ".out")
    if not os.path.isfile(out_f):
        pu.report_and_raise_error("out file {0} not created".format(out_f))

    error = UM.obj_fn(out_f)

    with open(os.path.join(job_dir, job + ".opt"), "a") as fobj:
        fobj.write("error = {0:12.6E}\n".format(error))

    # go back to the base_dir
    os.chdir(base_dir)

    return error
