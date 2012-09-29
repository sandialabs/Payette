"""Payette_optimize.py module. Contains classes and functions for optimizing
parameters using Payette.

"""

import os
import sys
import imp
import re
import shutil
import numpy as np
import scipy
import scipy.optimize
import math
from copy import deepcopy

import Source.__config__ as cfg
import Source.Payette_utils as pu
import Source.Payette_container as pcntnr
import Source.Payette_extract as pe
import Source.Payette_input_parser as pip
import Source.__runopts__ as ro
import Source.Payette_sim_index as psi
import Toolset.KayentaParamConv as kpc

# Module level variables
IOPT = -1
FAC = []
FNEWEXT = ".0x312.gold"
OBJFN = None


class Optimize(object):
    r"""docstring -> needs to be completed """

    def __init__(self, ilines):
        r""" Initialization """

        # get the optimization block
        ui = pip.InputParser(ilines)
        self.name = ui.name

        regex = re.compile(r"simdir", re.I|re.M)
        if regex.search("\n".join(ui.options())) or ro.SIMDIR is not None:
            pu.report_and_raise_error(
                "cannot specify simdir for permutation jobs")
        optimize = ui.find_block("optimization")
        self.oblock = optimize

        # save optimization information to single "data" dictionary
        self.data = {}
        self.data["basename"] = self.name
        self.data["verbosity"] = ro.VERBOSITY
        self.data["return level"] = ro.DISP
        self.data["fext"] = ".opt"
        self.data["options"] = []

        # fill the data with the optimization information
        self.parse_optimization_block()

        ilines = ui.user_input(pop=("optimization",))
        ilines = re.sub(r"(?i)\btype\s.*", "type simulation", ilines)
        self.data["baseinp"] = ilines

        # set the loglevel to 0 for Payette simulation and save the payette
        # options to the data dictionary
        ro.set_global_option("VERBOSITY", 0, default=True)

        # check user input for required blocks
        if not ui.find_block("material"):
            pu.report_and_raise_error(
                "material block not found in input file")

        # check the optimization variables
        self.check_params()

        if self.data["verbosity"]:
            pu.log_message("Optimizing {0}".format(self.name), noisy=True)
            pu.log_message("Optimization variables: {0}"
                           .format(", ".join(self.data["optimize"])),
                           noisy=True)
            pu.log_message("Objective function file: {0}".format(OBJFN.__file__),
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

    def parse_optimization_block(self):
        r"""Get the required optimization information.

        Populates the self.data dict with information parsed from the
        optimization block of the input file. Sets defaults where not
        specified.

        Parameters
        ----------
        oblock : array_like
            optimization block from input file

        Returns
        -------
        None

        """
        global OBJFN
        allowed_methods = {
            "simplex": {"method": "Nelder-Mead", "name": "fmin"},
            "powell": {"method": "Powell", "name": "fmin_powell"},
            "cobyla": {"method": "COBYLA", "name": "fmin_cobyla"},
            "slsqp": {"method": "SLSQP", "name":"fmin_slsqp"}}

        # default method
        opt_method = allowed_methods["simplex"]
        optimize = {}
        min_legacy = {"abscissa": None, "vars": []}
        gold_f = None

        # get options
        while True:
            option = self.find_oblock_option("option")
            if option is None:
                break
            self.data["options"].append(option.lower())
            continue

        # get method
        method = self.find_oblock_option("method")
        if method is not None:
            method = method.lower()
            opt_method = allowed_methods.get(method)
            if opt_method is None:
                pu.report_and_raise_error("invalid method {0}".format(method))

        # tolerance, maxiter
        tolerance = float(self.find_oblock_option("tolerance", 1.e-4))
        maxiter = int(self.find_oblock_option("maxiter", 20))
        disp = self.find_oblock_option("disp", False)
        if disp:
            if re.search(r"(?i)\bfalse\b", disp) or re.search(r"\b0\b", disp):
                disp = False
            else:
                disp = True

        # objective function
        fnam = os.path.join(cfg.OPTREC, "Opt_legacy.py") # default
        fnam = self.find_oblock_option("obj_fn", fnam)
        fnam = re.sub(r"\bin\s", "", fnam).strip()
        if not os.path.isfile(fnam):
            ftry = os.path.join(cfg.OPTREC, fnam)
            if not os.path.isfile(ftry):
                pu.report_error("obj_fn {0} not found".format(fnam))
            fnam = ftry
        if os.path.splitext(fnam)[1] not in (".py",):
            pu.report_error("obj_fn {0} must be .py file".format(fnam))
        else:
            pymod, pypath = pu.get_module_name_and_path(fnam)
        try:
            fobj, path, desc = imp.find_module(pymod, pypath)
            OBJFN = imp.load_module(pymod, fobj, path, desc)
            fobj.close()
            if not hasattr(OBJFN, "obj_fn"):
                pu.report_error("obj_fn must define a 'obj_fn' function")
        except ImportError:
            pu.report_error("obj_fn {0} not imported".format(fnam))
        if pu.error_count():
            pu.report_and_raise_error("stopping due to previous errors")

        # get variables to optimize
        optmz = []
        while True:
            opt = self.find_oblock_option("optimize")
            if opt is None:
                break
            optmz.append(re.sub(r"[\(\)]", "", opt).lower().split())
            continue
        for item in optmz:
            key, vals = item[0], item[1:]
            optimize[key] = {"bounds": [None, None],
                             "initial value": None}

            if "bounds" in vals:
                try:
                    idx = vals.index("bounds") + 1
                    bounds = [float(x) for x in vals[idx:idx+2]]
                except ValueError:
                    bounds = [None, None]
                    pu.report_error("Bounds requires 2 arguments")

                if bounds[0] > bounds[1]:
                    pu.report_error(
                        "lower bound {0} > upper bound {1} for {2}"
                        .format(bounds[0], bounds[1], key))
                optimize[key]["bounds"] = bounds

            if "initial" in vals:
                idx = vals.index("initial")
                if vals[idx + 1] == "value":
                    idx = idx + 1
                    ival = float(vals[idx+1])
                else:
                    ival = None
                optimize[key]["initial value"] = ival
            continue

        # get variables to minimize
        minmz = []
        while True:
            tmp = self.find_oblock_option("minimize")
            if tmp is None:
                break
            minmz.append(re.sub(r"[\(\)]", "", tmp).lower().split())
            continue
        for min_vars in minmz:
            # get variables to minimize during the optimization
            for min_var in min_vars:
                if min_var == "versus":
                    val = min_vars[min_vars.index("versus") + 1]
                    if val[0] != "@":
                        val = "@" + val
                    min_legacy["abscissa"] = val
                    break

                if min_var[0] != "@":
                    min_var = "@" + min_var

                if min_var not in min_legacy["vars"]:
                    min_legacy["vars"].append(min_var)
                continue
            continue

        # get the gold file
        fnam = self.find_oblock_option("gold file")
        if fnam is not None:
            if not os.path.isfile(fnam):
                pu.report_error("gold file {0} not found".format(fnam))
            else:
                gold_f = fnam

        # check that minimum info was given
        if min_legacy["vars"]:
            if OBJFN is not None:
                if "Opt_legacy" not in os.path.basename(OBJFN.__file__):
                    pu.report_error(
                        "Specification of 'minimize' in optimization block "
                        "requires obj_fn to be Opt_legacy.py, instead "
                        "{0} was given".format(os.path.basename(OBJFN.__file__)))
            else:
                fnam = os.path.join(cfg.OPTREC, "Opt_legacy.py")
                pymod, pypath = pu.get_module_name_and_path(fnam)
                fobj, path, desc = imp.find_module(pymod, pypath)
                OBJFN = imp.load_module(pymod, fobj, path, desc)
                fobj.close()

            if gold_f is None:
                pu.report_error("No gold file given for optimization problem")

        if OBJFN is None:
            pu.report_error("no objective function given")

        if not optimize:
            pu.report_error("no parameters to optimize given")

        for key, val in optimize.items():
            if val["initial value"] is None:
                pu.report_error("no initial value given for {0}".format(key))

        # check for errors up to this point
        if pu.error_count():
            pu.report_and_raise_error("stopping due to previous errors")

        # initialize the module, if the user desires
        try:
            OBJFN.init(gold_f, min_legacy["abscissa"], min_legacy["vars"])
        except AttributeError:
            pass

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
        # Run the model with initial values
        preprocessor = ""
        pu.log_message("Calling model with initial parameters")
        for key, val in self.data["optimize"].items():
            preprocessor += "{0} = {1}\n".format(key, val["initial value"])
        ui = pip.preprocess(self.data["baseinp"], preprocessor=preprocessor)
        the_model = pcntnr.Payette(ui)
        if pu.warn_count():
            pu.report_and_raise_error("Stopping due to initial warnings")
        the_model.finish(wipeall=True)
        return

    def find_oblock_option(self, option, default=None):
        option = ".*".join(option.split())
        pat = r"(?i)\b{0}\s".format(option)
        fpat = pat + r".*"
        option = re.search(fpat, self.oblock)
        if option:
            s, e = option.start(), option.end()
            option = self.oblock[s:e]
            self.oblock = (self.oblock[:s] + self.oblock[e:]).strip()
            option = re.sub(pat, "", option)
            option = re.sub(r"[\,=]", " ", option).strip()
        else:
            option = default
        return option


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
            "Method {0} does not enforce bounds.  Bounds will be ignored"
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
    preprocessor = ""
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
            if lbnd is not None and opt_val < lbnd / FAC[idx]:
                return 1.e3
#            if ubnd is not None and opt_val > ubnd / FAC[idx]:
#                return 1.e3

            pstr = "{0} = {1:12.6E}".format(nam, opt_val * FAC[idx])
            preprocessor += "{0} = {1}\n".format(nam, opt_val * FAC[idx])
            variables[nam] = opt_val * FAC[idx]
            fobj.write(pstr + "\n")
            msg.append(pstr)
            continue

    ui = pip.preprocess(data["baseinp"], preprocessor=preprocessor)
    ui = re.sub(r"(?i)\bname\s.*", "name {0}".format(job), ui)
    if data["verbosity"]:
        pu.log_message("Iteration {0:03d}, trial parameters: {1}"
                       .format(IOPT + 1, ", ".join(msg)),
                       noisy=True)

    # instantiate Payette object
    the_model = pcntnr.Payette(ui)

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

    error = OBJFN.obj_fn(out_f)

    with open(os.path.join(job_dir, job + ".opt"), "a") as fobj:
        fobj.write("error = {0:12.6E}\n".format(error))

    # go back to the base_dir
    os.chdir(base_dir)

    return error
