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
import math

import Source.Payette_utils as pu
import Source.Payette_container as pc
import Source.Payette_driver as pd
import Source.Payette_extract as pe

# Module level variables
IOPT = -1


class Optimize(object):
    r"""docstring -> needs to be completed """

    def __init__(self, job, job_inp, job_opts):
        r""" Initialization """

        errors = 0

        self.basename = job
        self.job_opts = job_opts

        self.data = {}
        self.data["basename"] = job
        self.data["verbosity"] = job_opts.verbosity

        # set verbosity to 0 for Payette simulation
        self.job_opts.verbosity = 0

        # get the optimization block
        opt, optid = pu.findBlock(job_inp, "optimization")

        # save the job_inp.  findBlock above removes the optimization block
        self.data["baseinp"] = job_inp

        # fill the data with the optimization information
        self.get_opt_info(opt)

        # check the optimization variables
        errors += self.check_opt_parameters()
        if errors:
            pu.logerr("exiting due to previous errors")
            sys.exit(123)


        # check minimization variables
        errors += self.check_min_parameters()
        if errors:
            pu.logerr("error extracting minimization variables from {0}"
                      .format(self.data["gold file"]))
            sys.exit(123)

        if self.data["verbosity"]:
            pu.loginf("Optimizing {0}".format(job))
            pu.loginf("Optimization variables: {0}"
                      .format(", ".join(self.data["optimize"])))
            minvars = ", ".join([x[1:] for x in self.data["minimize"][1:]])
            pu.loginf("Minimization variables: {0}".format(minvars))
            pu.loginf("Gold file: {0}".format(self.data["gold file"]))
            pu.loginf("Minimization method: {0}"
                      .format(self.data["optimization method"].__name__))

    def optimize(self):
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
        dnam = self.basename + ".opt"
        base_dir = os.path.join(os.getcwd(), dnam)
        idir = 0
        while True:
            if os.path.isdir(base_dir):
                base_dir = os.path.join(
                    os.getcwd(), dnam + ".{0:03d}".format(idir))
            elif idir > 100:
                sys.exit("ERROR: max number of dirs")
            else:
                break
            idir += 1
        if self.data["verbosity"]:
            pu.loginf("Optimization directory: {0}".format(base_dir))

        os.mkdir(base_dir)
        os.chdir(base_dir)

        # copy gold file to base_dir
        gold_f = os.path.join(base_dir,
                              os.path.basename(self.data["gold file"]))
        shutil.copyfile(self.data["gold file"], gold_f)

        # extract only what we want from the gold file
        exargs = [gold_f, "--silent", "--xout"] + self.data["minimize"]
        pe.extract(exargs)

        # extract created a file basename(gold_f).xout, change ext to .xgold
        xgold = gold_f.replace(".gold", ".xgold")
        shutil.move(gold_f.replace(".gold", ".xout"), xgold)

        # initial guess for opt_params are those from input
        nams = [x for x in self.data["optimize"]]
        nams.sort()
        opt_params = [0.] * len(nams)
        for idx, nam in enumerate(nams):
            opt_params[idx] = self.data["optimize"][nam]["initial value"]
            continue

        # set up args and call optimzation routine
        args = [self.data, base_dir, self.job_opts, xgold]
        maxiter = self.data["maximum iterations"]
        tolerance = self.data["tolerance"]
        opt_params = self.data["optimization method"](
            func, opt_params, xtol=tolerance, ftol=tolerance,
            args=args, maxiter=maxiter, disp=False)

        # optimum parameters found, write out final info
        msg = ["{0} = {1:12.6E}".format(nams[i], x)
               for i, x in enumerate(opt_params)]
        pu.loginf("Optimized parameters found on the iteration {0:d}"
                  .format(IOPT))
        pu.loginf("Optimized parameters: {0}".format(", ".join(msg)))

        # last_job = os.path.join(base_dir,
        #                         self.basename + ".{0:03d}".format(IOPT))
        # pu.loginf("Ultimate simulation directory: {0}".format(last_job))

        # write out the optimized parameters
        with open(os.path.join(base_dir, self.basename) + ".opt", "w") as fobj:
            fobj.write("Optimized parameters\n")
            for idx, nam in enumerate(nams):
                opt_val = opt_params[idx]
                fobj.write("{0} = {1:12.6E}\n".format(nam, opt_val))
                continue

        os.chdir(cwd)
        return 0

    def finish(self):
        r""" finish up the optimization job """

        pass

    def get_opt_info(self, opt_block):
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
        minimize = ["@time"]
        allowed_methods = ["fmin", "fmin_powell"]
        opt_method = "fmin"
        maxiter = 20
        tolerance = 1.e-4
        optimize = {}

        for item in opt_block:

            item = item.replace(",", " ").replace("=", " ").split()

            if "gold" in item[0].lower() and "file" in item[1].lower():
                if not os.path.isfile(item[2]):
                    pu.logerr("gold file {0} not found".format(item[2]))
                else:
                    gold_f = item[2]

            elif "minimize" in item[0].lower():
                min_vars = item[1:]
                for min_var in min_vars:
                    if min_var[0] != "@":
                        min_var = "@" + min_var
                    if min_var not in minimize:
                        minimize.append(min_var)
                    continue

                # make sure time is in the list and it is first
                for min_var in minimize:
                    if not [x for x in minimize if "@time" in x.lower()]:
                        minimize.append("@time")

            elif "optimize" in item[0].lower():
                # set up this parameter to optimize
                key = item[1]
                vals = item[2:]

                # For now, there is no initial value. It is given in the
                # material block of the input file, we just hold its place
                # here. We will later check that this key was given in the
                # material block.
                optimize[key] = {"initial value": None}

                # uppder bound
                if "ubound" in vals:
                    try:
                        val = eval(vals[vals.index("ubound") + 1])
                    except NameError:
                        val = vals[vals.index("ubound") + 1]

                    optimize[key]["ubound"] = val

                else:
                    optimize[key]["ubound"] = None

                # lower bound
                if "lbound" in vals:
                    try:
                        val = eval(vals[vals.index("lbound") + 1])
                    except NameError:
                        val = vals[vals.index("lbound") + 1]

                    optimize[key]["lbound"] = val

                else:
                    optimize[key]["lbound"] = None

            elif "method" in item[0].lower():
                opt_method = item[1].lower()
                if opt_method not in allowed_methods:
                    pu.logerr("invalid method {0}".format(opt_method))
                    errors += 1

            elif "maxiter" in item[0].lower():
                maxiter = int(item[1])

            elif "tolerance" in item[0].lower():
                tolerance = float(item[1])

            continue

        if gold_f is None:
            pu.logerr("No gold file give for optimization problem")
            errors += 1
        else:
            if not os.path.isfile(gold_f):
                pu.logerr("gold file {0} not found".format(gold_f))
                errors += 1
            else:
                gold_f = os.path.realpath(gold_f)

        if not minimize:
            pu.logerr("No parameters to minimize given")
            errors += 1

        if not optimize:
            pu.logerr("No parameters to optimize given")
            errors += 1

        if errors:
            pu.logerr("resolve previous errors")
            sys.exit(2)

        self.data["gold file"] = gold_f
        self.data["minimize"] = minimize
        self.data["optimize"] = optimize
        self.data["maximum iterations"] = maxiter
        self.data["tolerance"] = tolerance
        self.data["optimization method"] = eval(
            "{0}.{1}".format("scipy.optimize", opt_method))

        return

    def check_opt_parameters(self):
        r"""Check that the minimization parameters were specified in the input
        file and exist in the parameter table for the instantiated material.

        Parameters
        ----------
        None

        Returns
        -------
        errors : int
            0 if successfull, nonzero otherwise

        """


        errors = 0

        # the input for the job
        job_inp = [x for x in self.data["baseinp"]]

        # check that the optimize variables were given in the input file
        mtl, idx0, idxf = pu.has_block(job_inp, "material")
        opt_params = self.data["optimize"].keys()
        opt_params.sort()
        inp_params = []
        inp_vals = {}
        for line in job_inp[idx0:idxf]:
            if "constitutive" in line:
                continue
            param = "_".join(line.split()[0:-1])
            inp_params.append(param.lower())
            inp_vals[param] = float(line.split()[-1])
            continue

        inp_params.sort()
        not_in = [x for x in opt_params if x.lower() not in inp_params]
        if not_in:
            pu.logerr("Optimization parameter[s] {0} not in input parameters"
                      .format(", ".join(not_in)))
            errors += 1

        # instantiate a Payette object
        the_model = pc.Payette(self.basename, job_inp, self.job_opts)
        param_table = the_model.material.constitutive_model.parameter_table
        params = [x.lower() for x in param_table.keys()]
        try:
            os.remove(self.basename + ".log")
        except OSError:
            pass
        try:
            os.remove(self.basename + ".props")
        except OSError:
            pass

        # check that the optimize variables are in this models parameters
        not_in = [x for x in opt_params if x.lower() not in params]
        if not_in:
            pu.logerr("Optimization parameter[s] {0} not in model parameters"
                      .format(", ".join(not_in)))
            errors += 1

        if errors:
            return errors

        # there are no errors, now we want to find the line number in the
        # input file for the optimized params
        for iline, line in enumerate([x for x in self.data["baseinp"]]):
            for opt_param in opt_params:
                if opt_param in line.split():
                    self.data["optimize"][opt_param]["input idx"] = iline
                continue
            continue

        # double check that data["optimize"] has a input idx for every
        # optimize variable, and set the initial value
        for key, val in self.data["optimize"].items():
            if "input idx" not in val:
                errors += 1
                pu.logerr("No input idx for optimize variable {0}".format(key))
                continue

            self.data["optimize"][key]["initial value"] = inp_vals[key]
            continue

        return errors

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
        exargs.extend(self.data["minimize"])
        extraction = pe.extract(exargs)

        return extraction


def func(opt_params, data, base_dir, job_opts, xgold):

    r"""Objective function

    Creates a directory to run the current job, runs the job through Payette
    and then gets the average normalized root mean squared error between the
    output and the gold file.

    Parameters
    ----------
    opt_params : array_like
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
    job_inp = [x for x in data["baseinp"]]

    # replace the optimize variables with the updated and write params to file
    nams = [x for x in data["optimize"]]
    nams.sort()
    msg = []
    with open(os.path.join(job_dir, job + ".opt"), "w") as fobj:
        fobj.write("Parameters for iteration {0:d}\n".format(IOPT))
        for idx, nam in enumerate(nams):
            opt_val = opt_params[idx]
            line = data["optimize"][nam]["input idx"]
            pstr = "{0} = {1:12.6E}".format(nam, opt_val)
            job_inp[line] = pstr
            fobj.write(pstr + "\n")
            msg.append(pstr)
            continue

    if data["verbosity"]:
        pu.loginf("Iteration {0:d}, trial parameters: {1}"
                  .format(IOPT, ", ".join(msg)))

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

    exargs = [out_f, "--silent", "--xout"] + data["minimize"]
    pe.extract(exargs)
    xout = out_f.replace(".out", ".xout")

    # find the rms error between the out and gold
    error = compare_out_to_gold_rms(xgold, xout)

    # go back to the base_dir
    os.chdir(base_dir)

    return error


def compare_out_to_gold_rms(gold_f, out_f):
    r"""Compare results from out file to those in baseline (gold) file

    Parameters
    ----------
    gold_f : str
        Path to gold file
    out_f : str
        Path to out file

    Returns
    -------
    errors : float
        average normalized root mean squared difference of columns in gold_f
        and out_f

    """

    iam = "compare_out_to_gold_rms(gold_f, out_f)"
    errors = 0

    if not os.path.isfile(gold_f):
        pu.logerr(iam + "gold file {0} not found".format(gold_f))
        errors += 1
        pass

    if not os.path.isfile(out_f):
        pu.logerr(iam + "output file {0} not found".format(out_f))
        errors += 1
        pass

    if errors:
        sys.exit("resolve previous problems")

    # read in header
    out_h = [x.lower() for x in pu.get_header(out_f)]
    gold_h = [x.lower() for x in pu.get_header(gold_f)]

    if out_h[0] != "time":
        errors += 1
        pu.logerr(iam + "time not in outfile {0}".format(out_f))

    if gold_h[0] != "time":
        errors += 1
        pu.logerr(iam + "time not in gold file {0}".format(gold_f))

    if errors:
        sys.exit("resolve previous problems")

    # read in data
    out = pu.read_data(out_f)
    gold = pu.read_data(gold_f)

    # check that time is same (lengths must be the same)
    if len(gold[:, 0]) == len(out[:, 0]):
        rmsd, nrmsd = compute_fast_rms(gold[:, 0], out[:, 0])

    else:
        rmsd, nrmsd = 1.0e99, 1.0e99

    if nrmsd > np.finfo(np.float).eps:
        errors += 1
        pu.logerr(iam + "time step error between {0} and {1}"
                  .format(out_f, gold_f))

    if errors:
        sys.exit("resolve previous errors")

    # compare results
    errors = []

    # get items to skip
    to_compare = [x for x in out_h if x in gold_h]

    for val in to_compare:
        gidx = gold_h.index(val)
        oidx = out_h.index(val)
        rmsd, nrmsd = compute_rms(gold[:, 0], gold[:, gidx],
                                  out[:, 0], out[:, oidx])
        errors.append(nrmsd)
        continue

    errors = np.array(errors)
    return math.sqrt(np.sum(errors ** 2) / float(len(errors)))


def compute_rms(set1x, set1y, set2x, set2y, step=1):
    r"""Compute the root mean square difference between data in set1{x,y} and
    set2{x,y}.

    Determines the rms of the error between two piecewise functions. In almost
    all cases, the abscissa variables (set1x and set2x) will be time.
    Currently, the abscissa variables are required to be monotonically
    increasing (for easier computing). Interpolations are used to compare data
    points where data is missing in the other set.

    "set2{x,y}" is compared against "set1{x,y}".

    len(set1x) == len(set1y)
    len(set2x) == len(set2y)

    Parameters
    ----------
    set1x : array_like
        Abscissa of set 1
    set1y : array_like
        Range of set 1
    set2x : array_like
        Abscissa of set 2
    set2y : array_like
        Range of set 2
    step : {1,}, optional
        Used for large data sets. This gives the benchmark developer more
        control over how the error is computed (choosing accuracy vs. speed).

    Returns
    -------
    rmsd : float
        root mean square difference between gold and out
    nrmsd : float
        normalized root mean square difference between gold and out


    """

    # check the lengths of the arrays passed
    lset1x, lset1y, lset2x, lset2y = [len(x)
                                      for x in [set1x, set1y, set2x, set2y]]
    if lset1x != lset1y:
        sys.exit("len(set1x) != len(set1y)")
    if lset2x != lset2y:
        sys.exit("len(set2x) != len(set2y)")

    # Use a shortcut if the lengths of the x and y data sets are the same.
    # Also, before using the shortcut, do a quick check by computing the RMS
    # on every 10th x value. The RMS check on the x-values only gets computed
    # if len(set1y) == len(set2y). This check allows for the possibility of
    # the lengths of the data to be equal but for the spacing to be not equal.
    if (lset1y == lset2y and
        np.sum((set1x[::10] - set2x[::10]) ** 2) < 1.0e-6 * np.amax(set1x)):
        return compute_fast_rms(set1y, set2y)

    else:

        # compute the running square of the difference
        err = 0.0
        for i in range(0, lset2x, step):

            for j in range(0, lset1x - 1, step):

                # mss: Very expensive look. Optimize.
                # if set2x is between set1x pair, do linear interpolation
                if set1x[j] <= set2x[i] <= set1x[j + 1]:
                    x_0 = set1x[j]
                    y_0 = set1y[j]
                    x_1 = set1x[j + 1]
                    y_1 = set1y[j + 1]
                    y_f = y_0 + (set2x[i] - x_0) * (y_1 - y_0) / (x_1 - x_0)

                    # Add the square of the difference to err
                    err += (set2y[i] - y_f) ** 2
                    break
                continue
            continue

    rmsd = math.sqrt(err / float(lset1x))
    dnom = abs(np.amax(set1y) - np.amin(set1y))
    nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
    return rmsd, nrmsd


def compute_fast_rms(gold, out):
    r"""Determines the RMS of the error between two piecewise functions of same
    length.

    This should only be called if the timestep sizes of the gold simulation
    and the simulation that produced out are the same. This function makes no
    attempt to make that determination, it is up to the caller.

    Parameters
    ----------
    gold : str
        Path to gold file
    out : str
        Path to out file

    Returns
    -------
    rmsd : float
        root mean square difference between gold and out
    nrmsd : float
        normalized root mean square difference between gold and out

    """

    rmsd = math.sqrt(np.sum((gold - out) ** 2) / float(len(gold)))
    dnom = abs(np.amax(gold) - np.amin(gold))
    nrmsd = rmsd / dnom if dnom >= np.finfo(np.float).eps else rmsd
    return rmsd, nrmsd
