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

import Source.Payette_utils as pu
import Source.Payette_container as pc
import Source.Payette_driver as pd


class Visualize(object):
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

        # get the visualization block
        viz = job_inp["visualization"]["content"]

        # save the job_inp, minus the visualization block
        del job_inp["visualization"]
        self.data["baseinp"] = job_inp

        # fill the data with the visualization information
        self.get_viz_info(viz)

        # check the visualization variables
        errors += self.check_viz_parameters()
        if errors:
            pu.logerr("exiting due to previous errors")
            sys.exit(123)

        if self.data["verbosity"]:
            pu.loginf("Visualizing {0}".format(job))
            pu.loginf("Perturbed variables: {0}"
                      .format(", ".join(self.data["visualize"])))

    def visualize(self):
        r"""Run the visualization job

        Set up directory to run the visualization job and call the minimizer

        Parameters
        ----------
        None

        Returns
        -------
        viz_params : array_like
            The visualized parameters

        """

        # make the directory to run the job
        cwd = os.getcwd()
        dnam = self.basename + ".viz"
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
            pu.loginf("Visualization directory: {0}".format(base_dir))

        os.mkdir(base_dir)
        os.chdir(base_dir)

        # Initial values for viz_params are those from input. viz_params must
        # be in a consistent order throughout, here we arbitrarily set that
        # order to be based on an alphabetical sort of the names of the
        # parameters to be visualized.
        nams = [x for x in self.data["visualize"]]
        nams.sort()
        x0 = [0.] * len(nams)
        bounds = [None] * len(nams)
        for idx, nam in enumerate(nams):
            x0[idx] = self.data["visualize"][nam]["initial value"]
            bounds[idx] = self.data["visualize"][nam]["bounds"]
            continue

        # open up the index file
        index_f = os.path.join(base_dir, "index.py")
        with open(index_f, "w") as fobj:
            fobj.write("index = {}\n")

        # set up args and call visualization routine
        viz_args = [self.data, base_dir, self.job_opts, index_f, None]

        # call the function for each realization
        icall = 0

        nproc = self.job_opts.nproc
        nproc = min(min(mp.cpu_count(), nproc), max([len(x) for x in bounds]))
        if nproc > 1:
            self.data["verbosity"] = 0

        for iparam, param in enumerate(x0):

            # set up for multiprocessor, if applicable
            if nproc > 1:
                aargv = []
                pool = mp.Pool(processes=nproc)

            viz_params = [x for x in x0]
            for bound in bounds[iparam]:
                viz_params[iparam] = bound
                viz_args[-1] = icall
                argv = [np.array(viz_params)] + viz_args
                if nproc == 1:
                    func(argv)
                else:
                    aargv.append(argv)

                icall += 1
                continue

            if nproc > 1:
                icallf = icall
                icall0 = icall - len(bounds[iparam]) + 1
                pu.loginf("Running jobs {0:d}-{1:d}".format(icall0, icallf))
                pool.map(func, aargv)
                pool.close()
                pool.join()
                del pool

            continue

        os.chdir(cwd)

        return 0

    def finish(self):
        r""" finish up the visualization job """

        pass

    def get_viz_info(self, viz_block):
        r"""Get the required visualization information.

        Populates the self.data dict with information parsed from the
        visualization block of the input file. Sets defaults where not
        specified.

        Parameters
        ----------
        viz_block : array_like
            visualization block from input file

        Returns
        -------
        None

        """
        errors = 0
        visualize = {}

        for item in viz_block:

            item = item.replace(",", " ").replace("=", " ").split()

            if "vary" in item[0].lower():
                # set up this parameter to visualize
                key = item[1]
                vals = item[2:]

                # For now, there is no initial value. It is given in the
                # material block of the input file, we just hold its place
                # here. We will later check that this key was given in the
                # material block.
                visualize[key] = {"initial value": None}
                ubnd = None
                lbnd = None
                realizations = 10

                # uppder bound
                if "ubound" in vals:
                    try:
                        val = eval(vals[vals.index("ubound") + 1])
                    except NameError:
                        val = vals[vals.index("ubound") + 1]
                    ubnd = val

                # lower bound
                if "lbound" in vals:
                    try:
                        val = eval(vals[vals.index("lbound") + 1])
                    except NameError:
                        val = vals[vals.index("lbound") + 1]
                    lbnd = val

                if "realizations" in vals:
                    try:
                        val = eval(vals[vals.index("realizations") + 1])
                    except NameError:
                        val = vals[vals.index("realizations") + 1]
                    realizations = int(val)

                if lbnd is None:
                    pu.logerr("No lower bound to perturbed parameters given")
                    errors += 1

                if ubnd is None:
                    pu.logerr("No upper bound to perturbed parameters given")
                    errors += 1

                if lbnd > ubnd:
                    pu.logerr("lbound({0}) > ubound({1})".format(lbnd, ubnd))

                # adjust the bounds to be a range
                step = int((ubnd - lbnd) / float(realizations))
                visualize[key]["bounds"] = (
                    [lbnd + x * step for x in range(realizations)])

            continue


        if not visualize:
            pu.logerr("No parameters to visualize given")
            errors += 1

        if errors:
            pu.logerr("resolve previous errors")
            sys.exit(2)

        self.data["visualize"] = visualize

        return

    def check_viz_parameters(self):
        r"""Check that the perturbed parameters were specified in the input
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

        # check that the visualize variables were given in the input file
        material = self.data["baseinp"]["material"]["content"]
        viz_params = self.data["visualize"].keys()
        viz_params.sort()
        inp_params = []
        inp_vals = {}
        for line in material:
            if "constitutive" in line:
                continue
            param = "_".join(line.split()[0:-1])
            inp_params.append(param.lower())
            inp_vals[param] = float(line.split()[-1])
            continue

        inp_params.sort()
        not_in = [x for x in viz_params if x.lower() not in inp_params]
        if not_in:
            pu.logerr("Visualization parameter[s] {0} not in input parameters"
                      .format(", ".join(not_in)))
            errors += 1

        # instantiate a Payette object
        the_model = pc.Payette(self.basename, self.data["baseinp"], self.job_opts)
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

        # check that the visualize variables are in this models parameters
        not_in = [x for x in viz_params if x.lower() not in params]
        if not_in:
            pu.logerr("Visualization parameter[s] {0} not in model parameters"
                      .format(", ".join(not_in)))
            errors += 1

        if errors:
            return errors

        # there are no errors, now we want to find the line number in the
        # material block of the input file for the visualized params
        for iline, line in enumerate(material):
            for viz_param in viz_params:
                if viz_param in line.split():
                    self.data["visualize"][viz_param]["input idx"] = iline
                continue
            continue

        # double check that data["visualize"] has a input idx for every
        # visualize variable, and set the initial value
        for key, val in self.data["visualize"].items():
            if "input idx" not in val:
                errors += 1
                pu.logerr("No input idx for visualize variable {0}".format(key))
                continue

            self.data["visualize"][key]["initial value"] = inp_vals[key]
            continue

        return errors



def func(argv):
    r"""Objective function

    Creates a directory to run the current job, runs the job through Payette
    and then gets the average normalized root mean squared error between the
    output and the gold file.

    Parameters
    ----------
    viz_params : array_like
        Current values for perturbed parameters
    data : dict
        Visualization class data container
    job_opts : instance
        runPayette options
    icall : int
        call number

    Returns
    -------
    retcode : int

    """

    viz_params, data, base_dir, job_opts, index_f, icall = argv

    job = data["basename"] + ".{0:03d}".format(icall)

    job_dir = os.path.join(base_dir, job)
    os.mkdir(job_dir)
    os.chdir(job_dir)

    # instantiate the Payette object
    job_inp = deepcopy(data["baseinp"])

    # replace the visualize variables with the updated and write params to file
    nams = [x for x in data["visualize"]]
    nams.sort()
    msg = []
    with open(os.path.join(job_dir, job + ".viz"), "w") as fobj:
        fobj.write("Parameters for iteration {0:d}\n".format(icall))
        for idx, nam in enumerate(nams):
            viz_val = viz_params[idx]
            line = data["visualize"][nam]["input idx"]
            pstr = "{0} = {1:12.6E}".format(nam, viz_val)
            job_inp["material"]["content"][line] = pstr
            fobj.write(pstr + "\n")
            msg.append(pstr)

            continue

    # write out the input file, not actually used, but nice to have
    with open(os.path.join(job_dir, job + ".inp"), "w") as fobj:
        fobj.write("begin simulation {0}\n".format(job))
        # boundary block
        fobj.write("begin boundary\n")
        fobj.write("\n".join(job_inp["boundary"]["content"]) + "\n")
        # legs block
        fobj.write("begin legs\n")
        fobj.write("\n".join(job_inp["legs"]["content"]) + "\n")
        fobj.write("end legs\n")
        fobj.write("end boundary\n")
        # material block
        fobj.write("begin material\n")
        fobj.write("\n".join(job_inp["material"]["content"]) + "\n")
        fobj.write("end material\n")
        fobj.write("end simulation")

    if data["verbosity"]:
        pu.loginf("Iteration {0:d}, parameters: {1}"
                  .format(icall, ", ".join(msg)))

    # write to the index file
    with open(index_f, "w") as fobj:
        fobj.write('index[{0}] = {{"params": "{1}"}}\n'
                   .format(job, ", ".join(msg)))

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
