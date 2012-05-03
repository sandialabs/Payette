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
"""Payette_visualize.py module. Contains classes and functions for visualizing
parameters using Payette.

"""

import os
import sys
import shutil
import numpy as np
import math
import multiprocessing as mp
from copy import deepcopy
from itertools import izip, product
import random
import string

import Source.Payette_utils as pu
import Source.Payette_container as pc
import Source.Payette_driver as pd


FARGS = []


class Permutate(object):
    r"""docstring -> needs to be completed """

    def __init__(self, job, job_inp, job_opts):

        # extract the permutation block and delete it so that it is not read
        # again
        permutate = job_inp["permutation"]["content"]
        del job_inp["permutation"]


        # save Perturbate information to single "data" dictionary
        self.data = {}
        self.data["basename"] = job
        self.data["nproc"] = job_opts.nproc
        self.data["verbosity"] = job_opts.verbosity
        self.data["fext"] = ".perm"
        self.data["baseinp"] = job_inp

        # set verbosity to 0 for Payette simulation and save the payette
        # options to the data dictionary
        job_opts.verbosity = 0
        self.data["payette opts"] = job_opts

        # allowed directives
        self.allowed_directives = ("method", "options", "permutate", )

        # allowed options
        self.allowed_options = ( )
        self.conflicting_options = (( ))

        # methods
        self.method = None
        self.allowed_methods = (("combination", "combine"), "zip")

        # place holders for param_ranges and param_nams
        self.param_ranges = []
        self.param_nams = []
        self.initial_vals = []

        # fill the data with the permutated information
        self.get_params(permutate)

        # check the permutation variables
        self.check_params()

        # print info
        if self.data["verbosity"]:
            pu.loginf("Permutating job: {0}".format(self.data["basename"]))
            pu.loginf("Permutated variables: {0}"
                      .format(", ".join(self.param_nams)))
            pu.loginf("Permutation method: {0}".format(self.method))
        pass

    def run_job(self):
        r"""Run the permutation job

        Set up directory to run the permutation job and call the minimizer

        Parameters
        ----------
        None

        Returns
        -------

        """

        global FARGS

        # make the directory to run the job
        cwd = os.path.realpath(os.getcwd())
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

        # open up the index file
        index_f = os.path.join(base_dir, "index.py")
        with open(index_f, "w") as fobj:
            fobj.write("index = {}\n")

        # Save additional arguments to func in the global FARGS. This would be
        # handled better using something similar to scipy.optimize.py's
        # wrap_function, but that is not compatible with Pool.map.
        FARGS = [self.param_nams, self.data, base_dir,
                 self.data["payette opts"], index_f]

        nproc = min(mp.cpu_count(), self.data["nproc"])
        if nproc == 1:
            map(func, self.param_ranges)

        else:
            pool = mp.Pool(processes=nproc)
            pool.map(func, self.param_ranges)
            pool.close()
            pool.join()
            del pool

        os.chdir(cwd)

        if self.data["verbosity"]:
            pu.loginf("Permutation job {0} completed"
                      .format(self.data["basename"]), pre="\n")
            pu.loginf("Output directory: {0}".format(base_dir))

        return 0

    def finish(self):
        r""" finish up the permutation job """

        pass

    def get_params(self, permutation_block):
        r"""Get the required permutation information.

        Populates the self.data dict with information parsed from the
        permutation block of the input file. Sets defaults where not
        specified.

        Parameters
        ----------
        permutation_block : array_like
            permutation block from input file

        Returns
        -------
        None

        """
        errors = 0
        param_ranges = []
        options = []

        for item in permutation_block:

            for char in ",=()[]":
                item = item.replace(char, " ")
                continue

            item = item.split()
            directive = item[0].lower()

            if directive not in self.allowed_directives:
                errors += 1
                pu.logerr("unrecognized keyword \"{0}\" in permutation block"
                          .format(directive))
                continue

            if directive == "options":

                try:
                    opt = [x.lower() for x in item[1:]]
                except IndexError:
                    errors += 1
                    pu.logerr("no options given following 'option' directive")
                    continue

                if opt[0] in pu.flatten(self.allowed_methods):
                    pu.logwrn("using deprecated 'options' keyword to specify "
                              "the {0} 'method'".format(opt[0]))
                    directive = "method"

                else:
                    bad_opt = [x for x in opt if x not in self.allowed_options]
                    if bad_opt:
                        errors += 1
                        pu.logerr("options '{0:s}' not recognized"
                                  .format(", ".join(bad_opt)))

                    else:
                        options.extend(opt)

                    continue

            if directive == "method":

                # method must be one of the allowed_methods, and must only be
                # specified once
                try:
                    meth = item[1]
                except IndexError:
                    errors += 1
                    pu.logerr("no method given following 'method' directive")
                    continue

                if self.method is None:
                    self.method = meth

                else:
                    errors += 1
                    pu.logerr("requested method '{0}' but method '{1}'"
                              .format(meth, self.method) +
                              " already requested")
                    continue

                bad_meth = self.method not in pu.flatten(self.allowed_methods)
                if bad_meth:
                    errors += 1
                    pu.logerr(
                        "method '{0:s}' not recognized, allowed methods are {1}"
                        .format(self.method, ", ".join(self.allowed_methods)))
                continue

            if directive == "permutate":

                # set up this parameter to permutate
                try:
                    key = item[1]
                except IndexError:
                    errors += 1
                    pu.logerr("No item given for permutate keyword")
                    continue

                try:
                    vals = [x.lower() for x in item[2:]]
                except IndexError:
                    errors += 1
                    pu.logerr("No values given for permutate keyword {0}"
                              .format(key))
                    continue

                # specified range
                p_range = None
                if "range" in vals:
                    p_range = ", ".join(vals[vals.index("range") + 1:])
                    if len(p_range.split(",")) < 2:
                        errors += 1
                        pu.logerr("range requires at least 2 arguments")
                        continue

                    elif len(p_range.split(",")) == 2:
                        # default to 10 steps
                        p_range += ", 10"

                    p_range = eval("{0}({1})".format("np.linspace", p_range))

                # specified sequence
                elif "sequence" in vals:
                    p_range = ", ".join(vals[vals.index("sequence") + 1:])
                    p_range = eval("{0}([{1}])".format("np.array", p_range))

                # default: same as sequence above, but without the "sequence" kw
                else:
                    p_range = ", ".join(vals)
                    p_range = eval("{0}([{1}])".format("np.array", p_range))

                # check that a range was given
                if p_range is None or not len(p_range):
                    errors += 1
                    pu.logerr("no range/sequence given for " + key)
                    p_range = np.zeros(1)

                p_range = p_range.tolist()
                param_ranges.append(p_range)
                self.param_nams.append(key)
                self.initial_vals.append(p_range[0])

                continue

            continue

        if errors:
            pu.logerr("quiting due to previous errors")
            sys.exit(2)

        # check for conflicting options
        for item in self.conflicting_options:
            conflict = [x for x in options if x in item]
            if len(conflict) - 1:
                pu.logerr(
                    "conflicting options \"{0}\" ".format(", ".join(conflict)) +
                    "given in permuation block")
                sys.exit(2)

        if self.method is None:
            self.method = "zip"

        if "combin" in self.method:
            param_ranges = list(product(*param_ranges))

        else:
            if len(set([len(x) for x in param_ranges])) - 1:
                pu.logerr("number of permutations must be the same for "
                          "all permutated parameters when using method: [zip]")
                sys.exit(3)
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

        errors = 0

        # Remove from the material block the permutated parameters. Current
        # values will be inserted in to the material block for each run.
        content = []
        for line in self.data["baseinp"]["material"]["content"]:
            key = line.strip().lower().split()[0]
            if key in [x.lower() for x in self.param_nams]:
                continue
            content.append(line)
            continue
        self.data["baseinp"]["material"]["content"] = content

        # copy the job input and instantiate a Payette object
        job_inp = deepcopy(self.data["baseinp"])
        for key, val in zip(self.param_nams, self.initial_vals):
            job_inp["material"]["content"].append("{0} {1}".format(key, val))
        the_model = pc.Payette(self.data["basename"], job_inp,
                               self.data["payette opts"])
        param_table = the_model.material.constitutive_model.parameter_table

        # remove cruft
        for ext in (".log", ".props", ".math1", ".math2", ".prf"):
            try:
                os.remove(self.data["basename"] + ext)
            except OSError:
                pass
            continue

        # check that the visualize variables are in this models parameters
        not_in = [x for x in self.param_nams if x.lower() not in
                  [y.lower() for y in param_table.keys()]]

        if not_in:
            pu.logerr("permutated parameter[s] {0} not in model parameters"
                      .format(", ".join(not_in)))
            errors += 1

        if errors:
            pu.logerr("exiting due to previous errors")
            sys.exit(123)

        the_model.finish()

        return


def func(xcall):
    r"""Objective function

    Creates a directory to run the current job, runs the job through Payette
    and then gets the average normalized root mean squared error between the
    output and the gold file.

    Parameters
    ----------
    xcall : array_like
        Current values for permutated parameters

    Globals
    -------
    FARGS : array_like
        Additional arguments. The use of FARGS is mandated by Pool.map's
        inability to map over more than one argument.

    Returns
    -------
    retcode : int

    """

    job_id = xcall[0]
    xcall = xcall[1]

    xnams, data, base_dir, job_opts, index_f = FARGS
    job = data["basename"] + "." + job_id

    job_dir = os.path.join(base_dir, job)
    os.mkdir(job_dir)
    os.chdir(job_dir)

    # instantiate the Payette object
    job_inp = deepcopy(data["baseinp"])

    # replace the visualize variables with the updated and write params to file
    msg = []
    istr = []
    with open(os.path.join(job_dir, job + data["fext"]), "w") as fobj:
        fobj.write("Parameters for job {0}\n".format(job_id))
        for nam, val in zip(xnams, xcall):
            pstr = "{0} = {1:12.6E}".format(nam, val)
            istr.append('("{0}", {1:12.6E})'.format(nam, val))
            job_inp["material"]["content"].append(pstr)
            fobj.write(pstr + "\n")
            msg.append(pstr)
            continue

    # write out the input file, not actually used, but nice to have
    pu.write_input_file(job, job_inp, os.path.join(job_dir, job + ".inp"))

    if data["verbosity"]:
        pu.loginf("Running job {0:s}, parameters: {1}"
                  .format(job_id, ", ".join(msg)))

    # write to the index file
    with open(index_f, "a") as fobj:
        fobj.write('index[{0:d}] = {{'.format(int(job_id)))
        fobj.write('"name": "{0}", '.format(job))
        fobj.write('"directory": "{0}", '.format(job_dir))
        fobj.write('"permutated variables": ({0})'.format(", ".join(istr)))
        fobj.write("}\n")

    # instantiate Payette object
    the_model = pc.Payette(job, job_inp, job_opts)

    # run the job
    solve = pd.runProblem(the_model, restart=False)

    the_model.finish()

    if solve != 0:
        sys.exit("ERROR: simulation failed")

    # go back to the base_dir
    os.chdir(base_dir)

    return 0


