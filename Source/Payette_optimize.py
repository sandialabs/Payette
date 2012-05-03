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
import shutil
import numpy as np
import scipy
import scipy.optimize
import math
from copy import deepcopy

import Source.Payette_utils as pu
import Source.Payette_container as pc
import Source.Payette_driver as pd
import Source.Payette_extract as pe
import Toolset.KayentaParamConv as kpc

# Module level variables
IOPT = -1
FAC = []
FNEWEXT = ".0x312.gold"


class Optimize(object):
    r"""docstring -> needs to be completed """

    def __init__(self, job, job_inp, job_opts):
        r""" Initialization """

        # get the optimization block
        optimize = job_inp["optimization"]["content"]
        del job_inp["optimization"]

        # save Perturbate information to single "data" dictionary
        self.data = {}
        self.data["basename"] = job
        self.data["verbosity"] = job_opts.verbosity
        self.data["fext"] = ".opt"
        self.data["baseinp"] = job_inp
        self.data["options"] = []

        # set verbosity to 0 for Payette simulation and save the payette
        # options to the data dictionary
        job_opts.verbosity = 0
        self.data["payette opts"] = job_opts

        # fill the data with the optimization information
        self.parse_optimization_block(optimize)

        if "shearfit" not in self.data["options"]:
            # check user input for required blocks
            if "material" not in job_inp:
                pu.logerr("material block not found in input file")
                sys.exit(123)

        # check the optimization variables
        self.check_params()

        # check minimization variables
        self.check_min_parameters()

        if "shearfit" in self.data["options"]:
            self.init_shearfit()

        if self.data["verbosity"]:
            pu.loginf("Optimizing {0}".format(job))
            pu.loginf("Optimization variables: {0}"
                      .format(", ".join(self.data["optimize"])))
            minvars = ", ".join([x[1:] for x in self.data["minimize"]["vars"]])
            pu.loginf("Minimization variables: {0}".format(minvars))
            if self.data["minimize"]["abscissa"] is not None:
                pu.loginf("using {0} as abscissa"
                          .format(self.data["minimize"]["abscissa"][1:]))
            pu.loginf("Gold file: {0}".format(self.data["gold file"]))
            if self.data["options"]:
                pu.loginf("Optimization options: {0}"
                          .format(", ".join(self.data["options"])))
            pu.loginf("Optimization method: {0}"
                      .format(self.data["optimization method"]["method"]))

    def run_job(self):
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
        idir = 0
        while True:
            if os.path.isdir(base_dir):
                dir_id = ".{0:03d}".format(idir)
                base_dir = os.path.join(cwd, dnam + dir_id)
            else:
                break

            idir += 1
            if idir > 100:
                sys.exit("ERROR: max number of dirs")

        if self.data["verbosity"]:
            pu.loginf("Running: {0}".format(self.data["basename"]))

        os.mkdir(base_dir)
        os.chdir(base_dir)

        # copy gold file to base_dir
        gold_f = os.path.join(
            base_dir, os.path.basename(self.data["gold file"]))
        shutil.copyfile(self.data["gold file"], gold_f)

        # extract only what we want from the gold file
        exargs = [gold_f, "--silent", "--xout"]
        if self.data["minimize"]["abscissa"] is not None:
            exargs.append(self.data["minimize"]["abscissa"])
        exargs.extend(self.data["minimize"]["vars"])
        pe.extract(exargs)

        # extract created a file basename(gold_f).xout, change ext to .xgold
        xgold = gold_f.replace(".gold", ".xgold")
        shutil.move(gold_f.replace(".gold", ".xout"), xgold)

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
        opt_args = [opt_nams, self.data, base_dir,
                    self.data["payette opts"], xgold]
        opt_method = self.data["optimization method"]["method"]
        opt_options = {"maxiter": self.data["maximum iterations"],
                       "xtol": self.data["tolerance"],
                       "ftol": self.data["tolerance"],
                       "disp": self.data["disp"],}

        shearfit = "shearfit" in self.data["options"]
        if shearfit:
            fcn = rtxc
        else:
            fcn = func

        opt_params = minimize(
            fcn, opt_params, args=opt_args, method=opt_method,
            bounds=opt_bounds, options=opt_options,
            )

        # optimum parameters found, write out final info
        if shearfit:
            a_params = self.data["a_params"]
            a_params[self.data["index array"]] = opt_params
            stren, peaki1, fslope, yslope = kpc.old_2_new(*a_params)
            msg = ("A1 = {0:12.6E}, ".format(a_params[0]) +
                   "A2 = {0:12.6E}, ".format(a_params[1]) +
                   "A3 = {0:12.6E}, ".format(a_params[2]) +
                   "A4 = {0:12.6E}".format(a_params[3]) +
                   "\n {0}".format(" " * 27) +
                   "STREN = {0:12.6E}, ".format(stren) +
                   "PEAKI1 = {0:12.6E}, ".format(peaki1) +
                   "FSLOPE = {0:12.6E}, ".format(fslope) +
                   "YSLOPE = {0:12.6E}".format(yslope))

        else:
            msg = ", ".join(["{0} = {1:12.6E}".format(opt_nams[i], x)
                             for i, x in enumerate(opt_params)])

        pu.loginf("Optimized parameters found on iteration {0:d}"
                  .format(IOPT + 1))
        pu.loginf("Optimized parameters: {0}".format(msg))

        # last_job = os.path.join(base_dir,
        #                         self.data["basename"] + ".{0:03d}".format(IOPT))
        # pu.loginf("Ultimate simulation directory: {0}".format(last_job))

        # write out the optimized parameters
        opt_f = os.path.join(base_dir, self.data["basename"] + ".opt")
        with open(opt_f, "w") as fobj:
            fobj.write("Optimized parameters\n")
            for idx, nam in enumerate(opt_nams):
                opt_val = opt_params[idx]
                fobj.write("{0} = {1:12.6E}\n".format(nam, opt_val))
                continue

            if shearfit:
                fobj.write("\nEquivalent new paramters:\n")
                fobj.write("STREN = {0:12.6E}\n".format(stren))
                fobj.write("PEAKI1 = {0:12.6E}\n".format(peaki1))
                fobj.write("FSLOPE = {0:12.6E}\n".format(fslope))
                fobj.write("YSLOPE = {0:12.6E}".format(yslope))

        os.chdir(cwd)
        return 0

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
        errors = 0
        gold_f = None
        minimize = {"abscissa": None, "vars": []}
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
        shearfit = False
        optimize = {}
        fix = {}

        # get options first -> must come first because some options have a
        # default method different than the global default of simplex
        for item in opt_block:
            for pat, repl in ((",", " "), ("=", " "), ):
                item = item.replace(pat, repl)
            item = item.split()

            if "option" in item[0].lower():
                self.data["options"].append(item[1].lower())

            continue

        if "shearfit" in self.data["options"]:
            shearfit = True
            opt_method = allowed_methods["cobyla"]

        # get method before other options
        for item in opt_block:
            for pat, repl in ((",", " "), ("=", " "), ):
                item = item.replace(pat, repl)
            item = item.split()
            if "method" in item[0].lower():
                opt_method = allowed_methods.get(item[1].lower())
                if opt_method is None:
                    pu.logerr("invalid method {0}".format(item[1].lower()))
                    errors += 1

        if errors:
            pu.logerr("stopping due to previous errors")
            sys.exit(123)

        # now get the rest
        for item in opt_block:
            for pat, repl in ((",", " "), ("=", " "), ("(", " "), (")", " "),):
                item = item.replace(pat, repl)
            item = item.split()

            if "gold" in item[0].lower() and "file" in item[1].lower():
                if not os.path.isfile(item[2]):
                    errors += 1
                    pu.logerr("gold file {0} not found".format(item[2]))

                else:
                    gold_f = item[2]

            elif "minimize" in item[0].lower():
                # get variables to minimize during the optimization
                min_vars = item[1:]
                for min_var in min_vars:
                    if min_var == "versus":
                        val = min_vars[min_vars.index("versus") + 1]
                        if val[0] != "@":
                            val = "@" + val
                        minimize["abscissa"] = val
                        break

                    if min_var[0] != "@":
                        min_var = "@" + min_var

                    if min_var not in minimize["vars"]:
                        minimize["vars"].append(min_var)

                    continue

            elif "optimize" in item[0].lower():
                # set up this parameter to optimize
                key = item[1]
                vals = item[2:]

                if shearfit and key.lower() not in ("a1", "a2", "a3", "a4"):
                    errors += 1
                    pu.logerr("optimize variable "+ key +
                              " not allowed with shearfit method")

                bounds = (None, None)
                init_val = None

                # uppder bound
                if "bounds" in vals:
                    try:
                        idx = vals.index("bounds") + 1
                        bounds = [float(x) for x in vals[idx:idx+2]]
                    except ValueError:
                        errors += 1
                        pu.logerr("bounds requires 2 arguments")

                    if bounds[0] > bounds[1]:
                        pu.logerr("lower bound {0} > upper bound {1} for {2}"
                                  .format(bounds[0], bounds[1], key))

                if "initial" in vals:
                    idx = vals.index("initial")
                    if vals[idx + 1] == "value":
                        idx = idx + 1
                        init_val = float(vals[idx+1])

                if "lbound" in vals:
                    errors += 1
                    pu.logerr("depricated keyword 'lbound'")

                if "ubound" in vals:
                    errors += 1
                    pu.logerr("depricated keyword 'lbound'")

                optimize[key] = {}
                optimize[key]["bounds"] = bounds
                optimize[key]["initial value"] = init_val

            elif "fix" in item[0].lower():
                # set up this parameter to fix
                key = item[1]
                vals = item[2:]

                if shearfit and key.lower() not in ("a1", "a2", "a3", "a4"):
                    errors += 1
                    pu.logerr("fixed variable "+ key +
                              " not allowed with shearfit method")

                init_val = None
                if "initial" in vals:
                    idx = vals.index("initial")
                    if vals[idx + 1] == "value":
                        idx = idx + 1
                        init_val = float(vals[idx+1])

                fix[key] = {}
                fix[key]["initial value"] = init_val

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

        if gold_f is None:
            pu.logerr("No gold file given for optimization problem")
            errors += 1

        else:
            if not os.path.isfile(gold_f):
                errors += 1
                pu.logerr("gold file {0} not found".format(gold_f))
            else:
                gold_f = os.path.realpath(gold_f)

        if not shearfit and not minimize["vars"]:
            errors += 1
            pu.logerr("no parameters to minimize given")

        if not shearfit and not optimize:
            errors += 1
            pu.logerr("no parameters to optimize given")

        for key, val in optimize.items():
            if val["initial value"] is None:
                pu.logerr("no initial value given for {0}".format(key))
                errors += 1

        for key, val in fix.items():
            if val["initial value"] is None:
                errors += 1
                pu.logerr("no initial value given for {0}".format(key))
            if key.lower() in [x.lower() for x in optimize]:
                errors += 1
                pu.logerr("cannot fix and optimize {0}".format(key))

        if errors:
            pu.logerr("stopping due to previous errors")
            sys.exit(2)

        if shearfit:
            # for shearfit, we optimize a1 - a4 by minimizing errror in rootj2
            # vs. i1 using the rtxc function. Therefore, the only minimization
            # variables needed are i1 and rootj2
            for item in minimize["vars"]:
                if (item.lower() not in ("@i1", "@rootj2") and
                    item.lower()[0:4] != "@sig"):
                    errors += 1
                    pu.logerr("minimize variable "+ min_var[1:] +
                              " not allowed with shearfit method")
                continue

            if errors:
                pu.logerr("stopping due to previous errors")
                sys.exit(3)

            head = list(set([x.lower() for x in sorted(pu.get_header(gold_f))]))
            if head != ["i1", "rootj2"]:
                sigcnt = 0
                for item in head:
                    if item.lower()[0:3] == "sig":
                        sigcnt += 1
                    continue
                if sigcnt < 3:
                    pu.logerr("{0} does not contain sufficient information to "
                              "compute I1 and ROOTJ2".format(gold_f))
                    sys.exit(3)

                # user gave file with sig11, sig22, ..., it needs to be
                # converted to i1 and rootj2
                gold_f = get_rtj2_vs_i1(gold_f)

            minimize["vars"] = ["@I1", "@ROOTJ2"]

        self.data["gold file"] = gold_f
        self.data["minimize"] = minimize
        self.data["optimize"] = optimize
        self.data["maximum iterations"] = maxiter
        self.data["tolerance"] = tolerance
        self.data["optimization method"] = opt_method
        self.data["disp"] = disp
        self.data["fix"] = fix

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

        errors = 0

        if "shearfit" in self.data["options"]:
            # for shearfit, we don't actually have to call Kayenta
            return

        # Remove from the material block the optimized parameters. Current
        # values will be inserted in to the material block for each run.
        content = []
        for line in self.data["baseinp"]["material"]["content"]:
            key = line.strip().lower().split()[0]
            if key in [x.lower() for x in self.data["optimize"]]:
                continue
            content.append(line)
            continue
        self.data["baseinp"]["material"]["content"] = content

        # copy the job input and instantiate a Payette object
        job_inp = deepcopy(self.data["baseinp"])
        for key, val in self.data["optimize"].items():
            init_val = val["initial value"]
            job_inp["material"]["content"].append(
                "{0} {1}".format(key, init_val))

        the_model = pc.Payette(self.data["basename"],
                               job_inp,
                               self.data["payette opts"])
        param_table = the_model.material.constitutive_model.parameter_table

        try:
            os.remove(self.data["basename"] + ".log")
        except OSError:
            pass

        try:
            os.remove(self.data["basename"] + ".props")
        except OSError:
            pass

        # check that the optimize variables are in this models parameters
        not_in = [x for x in self.data["optimize"] if x.lower() not in
                  [y.lower() for y in param_table.keys()]]
        if not_in:
            pu.logerr("Optimization parameter[s] {0} not in model parameters"
                      .format(", ".join(not_in)))
            errors += 1

        if errors:
            pu.logerr("exiting due to previous errors")
            sys.exit(123)

        return

    def check_min_parameters(self):

        r"""Using the extract function in Payette_extract.py, check that the
        minimization parameters exist in the gold file.

        Parameters
        ----------
        None

        Returns
        -------
        extraction : int
            0 if successfull, nonzero otherwise

        """

        exargs = [self.data["gold file"], "--silent"]
        if self.data["minimize"]["abscissa"] is not None:
            exargs.append(self.data["minimize"]["abscissa"])
        exargs.extend(self.data["minimize"]["vars"])
        extraction = pe.extract(exargs)

        if extraction:
            pu.logerr("error extracting minimization variables from {0}"
                      .format(self.data["gold file"]))
            sys.exit(123)

        return

    def init_shearfit(self):
        """ Initialize data needed by shearfit

        To run shearfit, we need values for A1 - A4, in that order. The user
        may only want to optimize some, not all, of the A parameters. Here we
        initialize all A parameters to zero, and then populate the initial
        array with input from the user.

        """

        # initial setup
        a_params = np.zeros(4, dtype=float)
        index_array = [None] * 4

        # mapping to a_params array
        idx_map = {"a1": 0, "a2": 1, "a3":2, "a4":3}

        if not self.data["optimize"]:
            # user did not specify an parameters, get the first guess in a two
            # step process:
            # 1) get A1 and A4 through a linear fit, if it is good, keep the
            # result
            # 2) if needed, do a full curve fit, using A1 from above

            dat = pu.read_data(self.data["gold file"])
            ione = dat[:, 0]
            rootj2 = dat[:, 1]

            # linear fit
            fit = np.polyfit(ione, rootj2, 1, full=True)
            if fit[1] < 1.e-16:
                # data was very linear -> drucker prager -> fix all values
                a_params[3], a_params[0] = np.abs(fit[0])
                index_array = [None] * 4

            else:
                # least squares fit
                def rtj2(i1, a1, a2, a3, a4):
                    return a1 - a3 * np.exp(a2 * i1) - a4 * i1
                p0 = np.zeros(4)
                p0[0] = fit[0][1]
                curve_fit = scipy.optimize.curve_fit(rtj2, ione, rootj2, p0=p0)
                a_params = np.abs(curve_fit[0])

            for key in idx_map:
                if key in [x.lower() for x in self.data["fix"]]:
                    continue
                self.data["optimize"][key] = {}
                self.data["optimize"][key]["bounds"] = (None, None)
                self.data["optimize"][key]["initial value"] = (
                    a_params[idx_map[key]])

        else:
            for key, val in self.data["optimize"].items():
                idx = idx_map[key.lower()]
                a_params[idx] = val["initial value"]
                index_array[idx] = idx
                continue

        # replace the values the user wanted fixed
        for key, val in self.data["fix"].items():
            idx = idx_map[key.lower()]
            a_params[idx] = val["initial value"]
            continue

        # store the initial A params, and "index array": the array of indices
        # of the A parameters to be optimized.
        self.data["a_params"] = np.array(a_params)
        self.data["index array"] = np.array(
            [x for x in index_array if x is not None], dtype=int)

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
    job_opts : sequence
        List of options to be passed to runPayette
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
    shearfit = "shearfit" in args[1]["options"]

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
        pu.logwrn("Method {0} cannot handle constraints nor bounds directly."
                  .format(method))

    if has_bounds or shearfit:
        # user has specified bounds on the parameters to be optimized. Here,
        # we convert the bounds to inequality constraints
        errors = 0
        lcons, ucons = [], []
        for ibnd, bound in enumerate(bounds):
            lbnd, ubnd = bound
            if lbnd is None:
                if shearfit:
                    lbnd = 0.
                else:
                    lbnd = -1.e20
            if ubnd is None:
                if shearfit:
                    ubnd = x0[ibnd]*FAC[ibnd] + 50 * x0[ibnd]*FAC[ibnd]
                else:
                    ubnd = 1.e20

            if lbnd > ubnd:
                errors += 1
                pu.logerr("lbnd({0:12.6E}) > ubnd({1:12.6E})"
                          .format(lbnd, ubnd))

            lcons.append(lambda z, idx=ibnd, bnd=lbnd: z[idx] - bnd/FAC[idx])
            ucons.append(lambda z, idx=ibnd, bnd=ubnd: bnd/FAC[idx] - z[idx])

            bounds[ibnd] = (lbnd, ubnd)

            continue

        if errors:
            sys.exit("ERROR: Resolve previous errors")

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
        sys.exit("ERROR: Unrecognized method {0}".format(method))

    return xopt * FAC


def func(xcall, xnams, data, base_dir, job_opts, xgold):

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
    job_opts : instance
        runPayette options
    xgold : str
        File path to gold file

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

    # instantiate the Payette object
    job_inp = deepcopy(data["baseinp"])

    # replace the optimize variables with the updated and write params to file
    msg = []
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
            job_inp["material"]["content"].append(pstr)
            fobj.write(pstr + "\n")
            msg.append(pstr)
            continue

    if data["verbosity"]:
        pu.loginf("Iteration {0:03d}, trial parameters: {1}"
                  .format(IOPT + 1, ", ".join(msg)))

    # instantiate Payette object
    the_model = pc.Payette(job, job_inp, job_opts)

    # run the job
    solve = pd.runProblem(the_model, restart=False)

    the_model.finish()

    if solve != 0:
        sys.exit("ERROR: simulation failed")

    # extract minimization variables from the simulation output
    out_f = os.path.join(job_dir, job + ".out")
    if not os.path.isfile(out_f):
        sys.exit("out file {0} not created".format(out_f))

    exargs = [out_f, "--silent", "--xout"]
    if data["minimize"]["abscissa"] is not None:
        exargs.append(data["minimize"]["abscissa"])
    exargs.extend(data["minimize"]["vars"])
    pe.extract(exargs)
    xout = out_f.replace(".out", ".xout")

    if data["minimize"]["abscissa"] is not None:
        # find the rms error between the out and gold
        errors = pu.compare_out_to_gold_rms(xgold, xout)

    else:
        errors = pu.compare_file_cols(xgold, xout)

    if errors[0]:
        sys.exit("Resolve previous errors")

    error = math.sqrt(np.sum(errors[1] ** 2) / float(len(errors[1])))
    error = np.amax(np.abs(errors[1]))

    with open(os.path.join(job_dir, job + ".opt"), "a") as fobj:
        fobj.write("error = {0:12.6E}\n".format(error))

    # go back to the base_dir
    os.chdir(base_dir)

    return error


def rtxc(xcall, xnams, data, base_dir, job_opts, xgold):
    """The Kayenta limit function

    Evaluates the Kayenta limit function

              f(ione) = a1 + a3*exp(-a2*ione) + a4*ione

    Parameters
    ----------
    ione : array_like
        values of ione
    a1 - a4 : float
        Limit surface parameters

    Returns
    -------
    rootj2 : array_like
        values of sqrt(j2) corresponding to ione

    """

    global IOPT
    IOPT += 1

    job = data["basename"] + ".{0:03d}".format(IOPT)
    job_dir = os.path.join(base_dir, job)

    os.mkdir(job_dir)
    os.chdir(job_dir)

    # write trial params to file
    msg = []
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

            if ubnd is not None and opt_val > ubnd / FAC[idx]:
                return 1.e3

            pstr = "{0} = {1:12.6E}".format(nam, opt_val * FAC[idx])
            fobj.write(pstr + "\n")
            msg.append(pstr)
            continue

    if data["verbosity"]:
        pu.loginf("Iteration {0:03d}, trial parameters: {1}"
                  .format(IOPT + 1, ", ".join(msg)))

    # compute rootj2 and error
    v = data["index array"]
    a = data["a_params"]
    a[v] = xcall * FAC
    a1, a2, a3, a4 = a

    # enforce constraints
    if a1 < 0. or a2 < 0. or a3 < 0. or a4 < 0. or a1 - a3 < 0.:
        return 1.e3

    dat = pu.read_data(xgold)
    ione = dat[:, 0]
    rootj2 = dat[:, 1]

    error = rootj2 - (a1 - a3 * np.exp(a2 * ione) - a4 * ione)
    error = math.sqrt(np.mean(error ** 2))
    dnom = abs(np.amax(rootj2))
    error = error / dnom if dnom >= 2.e-16 else error

    with open(os.path.join(job_dir, job + ".opt"), "a") as fobj:
        fobj.write("error = {0:12.6E}\n".format(error))

    # go back to the base_dir
    os.chdir(base_dir)

    return error


def get_rtj2_vs_i1(fpath):
    """Convert sigij in file fpath to ione and rootj2

    Parameters
    ----------
    fpath : str
        path to file

    Returns
    -------
    fnew : str
        path to file with the ione and rootj2 values

    """

    header = pu.get_header(fpath)
    data = pu.read_data(fpath)
    sig11, sig22, sig33, sig12, sig23, sig13 = [None] * 6
    for idx, item in enumerate(header):
        if item.lower() == "sig11":
            sig11 = data[:, idx]

        if item.lower() == "sig22":
            sig22 = data[:, idx]

        if item.lower() == "sig33":
            sig33 = data[:, idx]

        if item.lower() == "sig12":
            sig12 = data[:, idx]

        if item.lower() == "sig23":
            sig23 = data[:, idx]

        if item.lower() == "sig13":
            sig13 = data[:, idx]

        continue

    if sig11 is None or sig22 is None or sig33 is None:
        sys.exit("insufficient information in {0} to compute I1 and ROOTJ2"
                 .format(gold_f))

    if sig12 is None:
        sig12 = np.zeros(len(sig11))

    if sig23 is None:
        sig23 = np.zeros(len(sig11))

    if sig13 is None:
        sig13 = np.zeros(len(sig11))

    i1 = sig11 + sig22 + sig33
    rtj2 = np.sqrt(1. / 6. * ((sig11 - sig22) ** 2 +
                              (sig22 - sig33) ** 2 +
                              (sig33 - sig11) ** 2) +
                   sig12 ** 2 + sig23 ** 2 + sig13 ** 2)

    fnam, fext = os.path.splitext(fpath)
    fnew = fnam + FNEWEXT

    with open(fnew, "w") as fobj:
        fobj.write("{0:12s}    {1:12s}\n".format("I1", "ROOTJ2"))
        for idx in range(len(i1)):
            fobj.write("{0:12.6E}    {1:12.6E}\n".format(i1[idx], rtj2[idx]))
            continue

    return fnew


