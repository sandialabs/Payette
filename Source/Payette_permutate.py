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

        # allowed options
        self.allowed_options = ("combination")

        # place holders for param_ranges and param_nams
        self.param_ranges = []
        self.param_nams = []

        # fill the data with the permutated information
        self.get_params(permutate)

        # check the permutation variables
        errors = 0
        errors += self.check_params()
        if errors:
            pu.logerr("exiting due to previous errors")
            sys.exit(123)

        if self.data["verbosity"]:
            pu.loginf("Visualizing {0}".format(job))
            pu.loginf("Perturbed variables: {0}"
                      .format(", ".join(self.param_nams)))
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
        cwd = os.getcwd()
        dnam = self.data["basename"] + self.data["fext"]
        base_dir = os.path.join(os.getcwd(), dnam)
        idir = 0
        while True:
            if os.path.isdir(base_dir):
                base_dir = os.path.join(os.getcwd(), dnam + str(idir))
            elif idir > 100:
                sys.exit("ERROR: max number of dirs")
            else:
                break
            idir += 1

        if self.data["verbosity"]:
            pu.loginf("Visualization directory: {0}".format(base_dir))

        os.mkdir(base_dir)
        os.chdir(base_dir)

        # open up the index file
        index_f = open(os.path.join(base_dir, "index.py"), "w")
        index_f.write("irun = 0\nindex = {}\n")

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

        index_f.close()
        os.chdir(cwd)

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

            item = item.replace(",", " ").replace("=", " ").split()

            directive = item[0].lower()

            if directive == "options":
                opt = ", ".join(item[1:])
                opt = opt.replace("(","").replace(")","")
                opt = opt.replace("[","").replace("]","")
                opt = opt.split(",")
                bad_opt = [x for x in opt if x not in self.allowed_options]
                if bad_opt:
                    errors += 1
                    pu.logerr("options '{0:s}' not recognized"
                              .format(", ".join(bad_opt)))
                else:
                    options.extend(opt)
                continue

            elif directive in "permutate":

                # set up this parameter to permutate
                key = item[1]
                vals = item[2:]

                # For now, there is no initial value. It is given in the
                # material block of the input file, we just hold its place
                # here. We will later check that this key was given in the
                # material block.

                # specified range
                if "range" in vals:
                    p_range = ", ".join(vals[vals.index("range") + 1:])
                    p_range = p_range.replace("(","").replace(")","")
                    p_range = eval("{0}({1})".format("np.linspace", p_range))

                # specified sequence
                elif "sequence" in vals:
                    p_range = ", ".join(vals[vals.index("sequence") + 1:])
                    p_range = p_range.replace("(","").replace(")","")
                    p_range = p_range.replace("[","").replace("]","")
                    p_range = eval("{0}([{1}])".format("np.array", p_range))

                # default: same as sequence above, but without the "sequence" kw
                else:
                    p_range = ", ".join(vals)
                    p_range = p_range.replace("(","").replace(")","")
                    p_range = p_range.replace("[","").replace("]","")
                    p_range = eval("{0}([{1}])".format("np.array", p_range))

                # check that a range was given
                if not len(p_range):
                    errors += 1
                    pu.logerr("No range/sequence given for " + key)
                    p_range = np.zeros(1)

                param_ranges.append(p_range.tolist())
                self.param_nams.append(key)
            continue

        if errors:
            pu.logerr("resolve previous errors")
            sys.exit(2)

        if "combination" in options:
            self.param_ranges = product(*param_ranges)
        else:
            if len(set([len(x) for x in param_ranges])) - 1:
                pu.logerr("number of permutations must be the same for "
                          "all permutated parameters")
                sys.exit(3)
            self.param_ranges = izip(*param_ranges)

        return

    def check_params(self):
        r"""Check that the permutated parameters were specified in the input
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

        # check that the permutated variables were given in the input file
        material = self.data["baseinp"]["material"]["content"]
        inp_params = []
        inp_vals = {}
        for line in material:
            if "constitutive" in line:
                continue
            param = "_".join(line.split()[0:-1])
            inp_params.append(param.lower())
            continue

        not_in = [x for x in self.param_nams if x.lower() not in inp_params]
        if not_in:
            pu.logerr("Visualization parameter[s] {0} not in input parameters"
                      .format(", ".join(not_in)))
            errors += 1


        # instantiate a Payette object
        the_model = pc.Payette( self.data["basename"],
                                self.data["baseinp"],
                                self.data["payette opts"])
        param_table = the_model.material.constitutive_model.parameter_table
        params = [x.lower() for x in param_table.keys()]
        try:
            os.remove(self.data["basename"] + ".log")
        except OSError:
            pass
        try:
            os.remove(self.data["basename"] + ".props")
        except OSError:
            pass

        # check that the visualize variables are in this models parameters
        not_in = [x for x in self.param_nams if x.lower() not in params]
        if not_in:
            pu.logerr("Visualization parameter[s] {0} not in model parameters"
                      .format(", ".join(not_in)))
            errors += 1

        # Remove from the material block the permutated parameters. Current
        # values will be inserted in to the material block for each run.
        is_in = [x.lower() for x in self.param_nams if x.lower() in inp_params]
        material = [x for x in material if x.lower().split()[0] not in is_in]
        self.data["baseinp"]["material"]["content"] = material
        return errors


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

    char_set = string.ascii_lowercase + string.digits * 3
    job_id = "".join(random.sample(char_set, 6))

    xnams, data, base_dir, job_opts, index_f = FARGS
    job = data["basename"] + "." + job_id

    job_dir = os.path.join(base_dir, job)
    os.mkdir(job_dir)
    os.chdir(job_dir)

    # instantiate the Payette object
    job_inp = deepcopy(data["baseinp"])

    # replace the visualize variables with the updated and write params to file
    msg = []
    with open(os.path.join(job_dir, job + data["fext"]), "w") as fobj:
        fobj.write("Parameters for job {0:s}\n".format(job_id))
        for nam, val in zip(xnams, xcall):
            pstr = "{0} = {1:12.6E}".format(nam, val)
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
    index_f.write('index[irun] = {' +
                  '"name": "{0}", '.format(job) +
                  '"directory": "{0}"'.format(job_dir) +
                  '}\n')
    index_f.write("irun += 1\n")

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

