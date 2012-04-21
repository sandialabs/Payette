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
"""Payette_enumerate.py module. Contains classes and functions for performing
enumerations (and permutations) on material parameters and running simulations
for each given instance.
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


class Enumerate(object):
    r"""docstring -> needs to be completed """

    def __init__(self, job, job_inp, job_opts):
        r""" Initialization """

        errors = 0

        self.basename = job
        self.job_opts = job_opts

        self.data = {}
        self.data["basename"] = job
        self.data["verbosity"] = job_opts.verbosity
        self.data["OPTS"] = job_opts

        # The following entries in the dictionary are special
        # to the enumeration module
        self.data["enumeration list"] = []  # Enumeration data
        self.data["enumeration line index"] = {} # File line index for params
        self.data["input string list"] = [] # input file stored as list of lines
        self.data["output dir list"] = []   # List of output directories

        # set verbosity to 0 for Payette simulation
        self.job_opts.verbosity = 0

        # get the 'enumeration' block and remove it from 'job_inp'
        opt = job_inp["enumeration"]["content"]

        # save the job_inp, minus the enumeration block
        del job_inp["enumeration"]
        self.data["baseinp"] = job_inp

        # fill the data with the optimization information
        self.get_enum_info(opt)

        errors = self.check_enum_parameters()
        if errors:
            pu.logerr("exiting due to previous errors")
            sys.exit(123)
        return

    def enumerate(self):
        r"""Run the enumerated simulations

        detailed info

        Parameters
        ----------
        None

        Returns
        -------
        opt_params : array_like
            The optimized parameters

        """

        # make the directory to run the job
        dir_n = self.basename + ".enum"
        base_dir = os.path.join(os.getcwd(), dir_n)
        idir = 0
        while True:
            if os.path.isdir(base_dir):
                base_dir = os.path.join(
                    os.getcwd(), dir_n + ".{0:03d}".format(idir))
            elif idir > 100:
                sys.exit("ERROR: max number of dirs")
            else:
                break
            idir += 1

        os.mkdir(base_dir)
        if self.data["verbosity"]:
            pu.loginf("Enumeration directory: {0}".format(base_dir))

        # This computes the number of digits we need for the current set
        # of runs. No sense in having run.0000001 when there are only 3 runs.
        digits = len(str(len(self.data["enumeration list"])))

        for idx, enumeration in enumerate(self.data["enumeration list"]):
            # Create the job directory
            job_d = os.path.join(base_dir,"job.{0:0{1}d}".format(idx,digits))
            os.mkdir(job_d)

            # Create the job .inp file.
            job_f = os.path.join(job_d, self.data["basename"]+".inp")
            tmp_baseinp = deepcopy(self.data["baseinp"])
            for new_token, new_value in enumeration.items():
                new_line = self.data["enumeration line index"][new_token]
                tmp_baseinp["material"]["content"][new_line] = (
                    "{0}={1}".format(new_token, new_value))

            # write out the input file
            with open(job_f, "w") as fobj:
                fobj.write("begin simulation {0}\n".format(self.data["basename"]))
                # boundary block
                fobj.write("begin boundary\n")
                fobj.write("\n".join(tmp_baseinp["boundary"]["content"]) + "\n")
                # legs block
                fobj.write("begin legs\n")
                fobj.write("\n".join(tmp_baseinp["legs"]["content"]) + "\n")
                fobj.write("end legs\n")
                fobj.write("end boundary\n")
                # material block
                fobj.write("begin material\n")
                fobj.write("\n".join(tmp_baseinp["material"]["content"]) + "\n")
                fobj.write("end material\n")
                fobj.write("end simulation")

            self.data["input string list"].append(deepcopy(tmp_baseinp))
            self.data["output dir list"].append(job_d)

        cwd = os.getcwd()
        for idx, input_file in enumerate(self.data["input string list"]):
            # Move to the job directory
            os.chdir(self.data["output dir list"][idx])

            # Make a copy of the .inp file (because running it clobbers it)
            dum_inp = deepcopy(input_file)

            # instantiate Payette object
            the_model = pc.Payette(self.data["basename"],
                                   dum_inp,
                                   self.data["OPTS"])

            # run the job
            solve = pd.runProblem(the_model, restart=False)
            pcnt_complete = (idx+1)/float(len(self.data["input string list"])) * 100.0
            pu.loginf("Enumeration {0: 6.2f}% complete".format(pcnt_complete))


            the_model.finish()

        if solve != 0:
            sys.exit("ERROR: simulation failed")


        os.chdir(cwd)
        return 0

    def finish(self):
        r""" finish up the enumeration job """

        pass

    def get_enum_info(self, opt_block):
        r"""Get the required enumeration information.

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

        disp = False

        allowed_combination_types = ["straight","permutation"]
        combination_type = "straight"

        enumerations = {}
        for item in opt_block:

            item = item.replace(",", " ").replace("=", " ").split()

            if "enumerate" in item[0].lower():
                try:
                    tmp_key = item[1].lower()
                except IndexError:
                    pu.logerr("Invalid 'enumerate' statement.")

                # Store as strings because strings might be enumerated upon.
                enum_args = item[2:]
                if len(enum_args) == 0:
                    pu.logerr("No arguments given with 'enumerate' statement.")
                if enum_args[0].count(":") == 2:
                    lbound, ubound, N = enum_args[0].split(":")
                    lbound = float(lbound)
                    ubound = float(ubound)
                    N = int(N)
                    enumerations[tmp_key] = [lbound + x / float(N - 1) *
                                        (ubound - lbound) for x in range(0, N)]
                    enumerations[tmp_key] = [ "{0:.14e}".format(x)
                                             for x in enumerations[tmp_key]]
                else:
                    enumerations[tmp_key] = item[2:]

                continue
            elif "type" in item[0].lower():
                try:
                    combination_type = item[1].lower()
                except IndexError:
                    pu.logerr("Expected combination type.")

                if not combination_type in allowed_combination_types:
                    pu.logerr("Invalid combination type specified '{0}'\n".
                                                  format(combination_type))
                continue
            else:
                pu.logerr("Invalid statement '{0}'.".format(" ".join(item)))

            continue

        # Now, we process the permutations either as straight (each parameter
        # is given 'x' values and 'x' simulations are spawned) or permutation
        # (where 'x' of one parameter is given, and 'y' of another, and
        # 'x*y' simulations are generated).
        enumerate_db = []
        if combination_type == "straight":
            # Ensure that all the enumerations have the same number of entries.
            num_values = [len(enumerations[x]) for x in enumerations]
            if not all(num_values[0] == x for x in num_values):
                pu.logerr("For 'straight' type combination, all params must"+
                          "have the same number of values.")

            if len(num_values) == 0:
                sys.exit("No enumerations detected.")

            # This populates the enumeration list with dictionaries that
            # define each individual enumeration.
            for idx in range(0,num_values[0]):
                tmp_dict = {}
                for param in enumerations.keys():
                    tmp_dict[param] = enumerations[param][idx]
                enumerate_db.append(tmp_dict)

        elif combination_type == "permutation":
                pu.logerr("Enumeration type 'permutation' not yet implemented.")
        else:
                pu.logerr("combination type not recognized")

        self.data["enumeration list"] = enumerate_db
        return

    def check_enum_parameters(self):
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

        # If no enumerations
        if len(self.data["enumeration list"]) == 0:
            pu.logerr("No enumerations saved. Cannot proceed")
            return 1

        errors = 0

        # check that the optimize variables were given in the input file
        material = self.data["baseinp"]["material"]["content"]
        viz_params = self.data["enumeration list"][0].keys()
        viz_params.sort()

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

        # check that the optimize variables are in this models parameters
        not_in = [x for x in viz_params if x.lower() not in params]
        if not_in:
            pu.logerr("Enumeration parameter[s] {0} not in model parameters"
                      .format(", ".join(not_in)))
            errors += 1

        if errors:
            return errors

        # there are no errors, now we want to find the line number in the
        # material block of the input file for the optimized params
        self.data["enumeration line index"] = {}
        for iline, line in enumerate(material):
            for viz_param in viz_params:
                if viz_param.lower() in line.lower().split():
                    self.data["enumeration line index"][viz_param] = iline
                continue
            continue
        # double check that data["optimize"] has a input idx for every
        # optimize variable, and set the initial value
        for key in self.data["enumeration list"][0].keys():
            if key not in self.data["enumeration line index"].keys():
                errors += 1
                pu.logerr("No input idx for enumeration variable '{0}'".format(key))
                continue
            continue

        return errors
