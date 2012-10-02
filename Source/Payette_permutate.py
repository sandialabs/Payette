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

"""Payette_visualize.py module. Contains classes and functions for visualizing
parameters using Payette.

"""
import os
import sys
import shutil
import re
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from itertools import izip, product

import Source.Payette_utils as pu
import Source.Payette_container as pc
import Source.Payette_input_parser as pip
import Source.Payette_sim_index as psi
import Source.__runopts__ as ro

DN, MN, DP = 10, 100, 10.

class Permutate(object):
    r"""docstring -> needs to be completed """

    def __init__(self, ilines):

        # get the permutation block
        ui = pip.InputParser(ilines)
        self.name = ui.name

        regex = re.compile(r"simdir", re.I|re.M)
        if regex.search("\n".join(ui.options())) or ro.SIMDIR is not None:
            pu.report_and_raise_error(
                "cannot specify simdir for permutation jobs")
        permutate = ui.find_block("permutation")
        self.pblock = permutate

        # save permutation information to single "data" dictionary
        self.data = {}
        self.data["basename"] = self.name
        self.data["verbosity"] = ro.VERBOSITY
        self.data["return level"] = ro.DISP
        self.data["fext"] = ".perm"
        self.data["options"] = []

        # number of processors
        nproc = int(ui.get_option("nproc", ro.NPROC))
        self.data["nproc"] = min(mp.cpu_count(), nproc)

        # set verbosity to 0 for Payette simulation and save the payette
        # options to the data dictionary
        ro.set_global_option("VERBOSITY", 0, default=True)

        # place holders for param_ranges and param_names
        self.method = None
        self.param_ranges = []
        self.param_names = []
        self.initial_vals = []

        # fill the data with the permutated information
        self.parse_permutation_block()
        ilines = ui.user_input(pop=("permutation",))
        ilines = re.sub(r"(?i)\btype\s.*", "type simulation", ilines)
        self.data["baseinp"] = ilines

        # check the permutation variables
        self.check_params()

        # print info
        if self.data["verbosity"]:
            pu.log_message("Permutating job: {0}".format(self.name),
                           noisy=True)
            pu.log_message("Permutated variables: {0}"
                           .format(", ".join(self.param_names)), noisy=True)
            pu.log_message("Permutation method: {0}".format(self.method),
                           noisy=True)

    def run_job(self, *args, **kwargs):
        r"""Run the permutation job

        Set up directory to run the permutation job and call the minimizer

        Parameters
        ----------
        None

        Returns
        -------

        """

        # make the directory to run the job
        cwd = os.path.realpath(os.getcwd())
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

        # Save additional arguments to func in the global FARGS. This would be
        # handled better using something similar to scipy.optimize.py's
        # wrap_function, but that is not compatible with Pool.map.
        args = ((x, self.param_names, self.data, base_dir, self.index)
                for x in self.param_ranges)

        if self.data["nproc"] == 1:
            results = [func(arg) for arg in args]

        else:
            pool = mp.Pool(processes=self.data["nproc"])
            results = pool.map(func, args)
            pool.close()
            pool.join()
            del pool

        for result in results:
            self.index.store(*result)
        os.chdir(cwd)

        if self.data["verbosity"]:
            pu.log_message("Permutation job {0} completed"
                           .format(self.data["basename"]), pre="\n",
                           noisy=True)
            pu.log_message("Output directory: {0}".format(base_dir),
                           noisy=True)

        retcode = 0

        if not self.data["return level"]:
            return retcode
        else:
            return {"retcode": retcode,
                    "index file": self.index.index_file(),
                    "simulation name": self.name}

    def finish(self):
        r""" finish up the permutation job """

        self.index.dump()
        return

    def parse_permutation_block(self):
        r"""Get the required permutation information.

        Populates the self.data dict with information parsed from the
        permutation block of the input file. Sets defaults where not
        specified.

        Parameters
        ----------

        Returns
        -------
        None

        """
        # Allowed directives, options, methods. Defaults
        adirctvs = ("method", "options", "permutate", )
        allowed_methods = {
            "combine": {"method": "Combination"},
            "combination": {"method": "Combination"},
            "shotgun": {"method": "Shotgun"},
            "zip": {"method": "Zip"},}
        perm_method = allowed_methods["zip"]
        param_ranges = []

        # get options
        while True:
            option = self.find_pblock_option("option")
            if option is None:
                break
            self.data["options"].append(option.lower())
            continue

        # get method
        method = self.find_pblock_option("method")
        if method is not None:
            perm_method = allowed_methods.get(method.lower())
            if perm_method is None:
                pu.report_and_raise_error("invalid method {0}".format(method))
        shotgun = perm_method["method"] == "Shotgun"

        # get seed
        seed = self.find_pblock_option("seed")
        if seed is not None:
            seed = int(float(seed))

        # get variables to permutate
        permutate = []
        while True:
            perm = self.find_pblock_option("permutate")
            if perm is None:
                break
            permutate.append(perm)
            continue

        # all variables to be permutated are now in permutate and of the form
        # key [kw] <vals>
        # where key is the name of the variable being permutated, kw is one of
        # ["sequence", "range", "normal", "weibull", "uniform",]
        # and vals are the values
        ptypes = [r"+/-", "sequence", "range", "normal", "weibull", "uniform",]
        regexs = [re.compile(r"(?i){0}".format(re.escape(x))) for x in ptypes]

        for line in permutate:

            # default permutation type
            ptype, method = "sequence", "sequence"

            # can only specify one type of kw per line
            kw = [ptypes[i] for i, x in enumerate(regexs) if x.search(line)]
            if len(kw) > 1:
                pu.report_error("Can only specify one of [{0}], got '{1}'"
                                .format(", ".join(ptypes), line))
                continue

            key = line.split()[0]

            # remove the key, paranthesis, and ptype from line and split it
            if kw:
                ptype = kw[0]
                method = ptype
            pat = r"[\(\)]|{0}|{1}".format(re.escape(key), re.escape(ptype))
            try:
                info = [float(x) for x in re.sub(pat, " ", line).split()]
            except ValueError:
                pu.report_error(
                    "Bad permutation specification at:\n\t{0}".format(line))
                continue

            if not info:
                pu.report_error(
                    "Insufficient information given to permutate "
                    "[{0}] at:\n\t{1}".format(key, line))
                continue

            if shotgun:
                # Shotgun method - use uniform distribution
                # specified as uniform(low, high)
                method = "uniform"
                if kw and kw[0].lower() != "range":
                    pu.report_error("Shotgun method takes only range "
                                    "permutation type at:\n\t{0}".format(line))
                    continue
                if 2 > len(info) > 3:
                    pu.report_error("Specify low, high[,N] for shotgun at:"
                                    "\n\t{0}".format(line))
                    continue
                a, b = info[:2]
                N = DN if len(info) < 3 else int(info[2])

            # construct the range of values of this parameter
            elif ptype == "+/-":
                method = "sequence"
                # Percentage of scale
                # specified as +/-(scale, perc)
                if 1 > len(info) > 2:
                    pu.report_error("Specify scale[,%] for +/- at:"
                                    "\n\t{0}".format(line))
                    continue
                P = DP if len(info) < 2 else float(info[1])
                fac = P / 100.
                a = [info[0] * (1. - fac), info[0], info[0] * (1. + fac)]
                b, N = None, None

            elif ptype == "sequence":
                # specified sequence
                a, b, N = info[:], None, None

            elif ptype == "range":
                # user specified a range
                if 2 > len(info) > 3:
                    pu.report_error("Specify low, high[,N] for range at:"
                                    "\n\t{0}".format(line))
                    continue
                a, b = info[:2]
                N = DN if len(info) < 3 else int(info[2])

            elif ptype == "normal":
                # Normal "Gaussian" distribution
                # specified as normal(loc, scale, size)
                if 1 > len(info) > 3:
                    pu.report_error("Specify mean, scale[,N] for normal at:"
                                    "\n\t{0}".format(line))
                    continue
                a, b = info[:2]
                N = DN if len(info) < 3 else int(info[2])

            elif ptype == "uniform":
                # Uniform distribution
                # specified as uniform(low, high, size)
                if 2 > len(info) > 3:
                    pu.report_error("Specify low[,high[,N]] for uniform at:"
                                    "\n]t{0}".format(line))
                    continue
                a, b = info[:2]
                N = DN if len(info) < 3 else int(info[2])

            elif ptype == "weibull":
                # Weibull distribution
                # specified as uniform(scale, shape, size)
                if 2 > len(info) > 3:
                    pu.report_error("Specify scale, shape[,N] for weibull at:"
                                    "\n\t{0}".format(line))
                    continue
                a, b = info[:2]
                N = DN if len(info) < 3 else int(info[2])

            else:
                pu.report_error("Unrecognized permutate specification "
                                "at:\n\t{0}".format(line))
                continue

            if N > MN:
                pu.report_error("Size N[{0}] exceeds maximum[{1}]".format(N, MN))
            prange = _get_psequence(a, b, N, method, seed)

            # check that a range was given
            if prange is None or not len(prange):
                pu.report_error(
                    "Insufficient information given to permutate "
                    "[{0}] at:\n\t{1}".format(key, line))
                continue

            param_ranges.append(prange.tolist())
            self.param_names.append(key)
            self.initial_vals.append(prange[0])
            continue

        # ---- Finish up ---------------------------------------------------- #
        if pu.error_count():
            pu.report_and_raise_error("Stopping due to previous errors")

        self.method = perm_method["method"]

        if self.method == "Combination":
            param_ranges = list(product(*param_ranges))

        else:
            if len(set([len(x) for x in param_ranges])) - 1:
                pu.report_and_raise_error(
                    "Number of permutations must be the same for "
                    "all permutated parameters when using method: {0}"
                    .format(self.method))
            param_ranges = zip(*param_ranges)

        nruns = len(param_ranges)
        pad = len(str(nruns))
        self.param_ranges = izip(
            ["{0:0{1}d}".format(x, pad) for x in range(nruns)], param_ranges)
        del param_ranges
        return

    def check_params(self):
        r"""Check that the permutated parameters were specified in the input
        file and exist in the parameter table for the instantiated material.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        preprocessor = ""
        pu.log_message("Calling model with initial parameters")
        for i, name in enumerate(self.param_names):
            preprocessor += "{0} = {1}\n".format(name, self.initial_vals[i])
        ui = pip.preprocess(self.data["baseinp"], preprocessor=preprocessor)
        the_model = pc.Payette(ui)
        if pu.warn_count():
            pu.report_and_raise_error("Stopping due to initial warnings")
        the_model.finish(wipeall=True)
        return

    def find_pblock_option(self, option, default=None):
        option = ".*".join(option.split())
        pat = r"(?i)\b{0}\s".format(option)
        fpat = pat + r".*"
        option = re.search(fpat, self.pblock)
        if option:
            s, e = option.start(), option.end()
            option = self.pblock[s:e]
            self.pblock = (self.pblock[:s] + self.pblock[e:]).strip()
            option = re.sub(pat, "", option)
            option = re.sub(r"[\,=]", " ", option).strip()
        else:
            option = default
        return option


def _get_psequence(a, b=None, N=None, method=None, seed=None):
    if method == "range":
        return np.linspace(a, b, N)
    if method == "sequence":
        return np.array(a)

    # method is one of random
    rstate = np.random.RandomState(seed)
    if method == "uniform":
        return rstate.uniform(a, b, N)
    if method == "normal":
        return rstate.normal(a, b, N)
    if method == "weibull":
        return a * rstate.weibull(b, N)

    pu.report_error("Unrecognized method: {0}".format(method))
    return


def func(args):
    r"""Objective function

    Creates a directory to run the current job, runs the job through Payette
    and then gets the average normalized root mean squared error between the
    output and the gold file.

    Parameters
    ----------
    args : array_like
        Current values for permutated parameters

    Globals
    -------
    args : array_like
        Additional arguments. The use of FARGS is mandated by Pool.map's
        inability to map over more than one argument.

    Returns
    -------
    retcode : int

    """

    xcall, xnams, data, base_dir, index = args
    job_id, xcall = xcall[0], xcall[1]

    job = data["basename"] + "." + job_id

    job_dir = os.path.join(base_dir, job)
    os.mkdir(job_dir)
    os.chdir(job_dir)

    # replace the visualize variables with the updated and write params to file
    msg = []
    preprocessor = ""
    variables = {}
    with open(os.path.join(job_dir, job + data["fext"]), "w") as fobj:
        fobj.write("Parameters for job {0}\n".format(job_id))
        for nam, val in zip(xnams, xcall):
            pstr = "{0} = {1:12.6E}".format(nam, val)
            preprocessor += "{0} = {1}\n".format(nam, val)
            variables[nam] = val
            fobj.write(pstr + "\n")
            msg.append(pstr)
            continue

    ui = pip.preprocess(data["baseinp"], preprocessor=preprocessor)
    ui = re.sub(r"(?i)\bname\s.*", "name {0}".format(job), ui)
    if data["verbosity"]:
        pu.log_message("Running job {0:s}, parameters: {1}"
                       .format(job_id, ", ".join(msg)), noisy=True)

    # instantiate Payette object
    the_model = pc.Payette(ui)

    # write out the input file, not actually used, but nice to have
    the_model.write_input = True

    # run the job
    solve = the_model.run_job()

    # store the data to the index
    the_model.finish()

    if ro.DISP:
        retcode = solve["retcode"]
    else:
        retcode = solve

    if retcode != 0:
        sys.exit("ERROR: simulation failed")

    # go back to the base_dir
    os.chdir(base_dir)

    return job_id, the_model.name, the_model.simdir, variables, the_model.outfile
