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
        self.data["input files"] = []
        self.data["output dirs"] = []
        self.data["OPTS"] = job_opts

        # set verbosity to 0 for Payette simulation
        self.job_opts.verbosity = 0

        # get the 'enumeration' block and remove it from 'job_inp'
        opt, optid = pu.findBlock(job_inp, "enumeration")

        # save the job_inp.
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
        dnam = self.basename + ".enum"
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
            pu.loginf("Enumeration directory: {0}".format(base_dir))

        digits=len(str(  len( self.data["permutations"] ) ))

        os.mkdir(base_dir)
#        os.chdir(base_dir)

        for idx, permutation in enumerate(self.data["permutations"]):
            # Create the job directory
            job_d = os.path.join(base_dir,"job.{0:0{1}d}".format(idx,digits))
            os.mkdir(job_d)

            # Create the job .inp file.
            job_f = os.path.join(job_d, self.data["basename"]+".inp")
            tmp_baseinp = self.data["baseinp"]
            for new_token in permutation:
                new_value = permutation[new_token]
                new_line = self.data["permute keys"][new_token]
                new_string = "{0}={1}".format(new_token,new_value)
                tmp_baseinp[new_line] = new_string
            TMPFILE = open(job_f,"w")
            TMPFILE.write("begin simulation {0}\n".format(self.data["basename"])+
                          "\n".join(tmp_baseinp) +
                          "\nend simulation")
            TMPFILE.close()
            self.data["input files"].append([x for x in tmp_baseinp])
            self.data["output dirs"].append(job_d)

        cwd = os.getcwd()
        for idx, input_file in enumerate(self.data["input files"]):
            os.chdir(self.data["output dirs"][idx])

            dum_inp = [x for x in input_file]
            # instantiate Payette object
            print(dum_inp)
            the_model = pc.Payette(self.data["basename"], dum_inp, self.data["OPTS"])
    
            # run the job
            solve = pd.runProblem(the_model, restart=False)
        
            the_model.finish()

        if solve != 0:
            sys.exit("ERROR: simulation failed")


        os.chdir(cwd)
        return 0

    def finish(self):
        r""" finish up the optimization job """

        pass

    def get_enum_info(self, opt_block):
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

        disp = False

        allowed_combination_types = ["straight","permutation"]
        combination_type = "straight"

        permutations = {}
        for item in opt_block:

            item = item.replace(",", " ").replace("=", " ").split()

            if "permute" in item[0].lower():
                try:
                    tmp_key = item[1].lower()
                except IndexError:
                    pu.logerr("Invalid 'permute' statement.")

                # Store as strings because strings might be permutted upon. 
                permutations[item[1].lower()] = item[2:]
                if len(permutations) == 0:
                    pu.logerr("No values given with 'permute' statement.")

                continue
            elif "combination" in item[0].lower():
                print("Stuff going on",item)
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
        permute_db = []
        if combination_type == "straight":
            num_permutes = [len(permutations[x]) for x in permutations]
            if not all(num_permutes[0] == x for x in num_permutes):
                pu.logerr("For 'straight' type combination, all params must"+
                          "have the same number of values.")
            for idx in range(0,num_permutes[0]):
                tmp_dict = {}
                for param in permutations.keys():
                    tmp_dict[param] = permutations[param][idx]
                permute_db.append(tmp_dict)
        elif combination_type == "permutation":
                pu.logerr("'permutation' not yet implemented.")
        else:
                pu.logerr("combination type not recognized")

        print(permute_db)

        self.data["permutations"] = permute_db
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


        errors = 0

        # the input for the job
        job_inp = [x for x in self.data["baseinp"]]

        # check that the optimize variables were given in the input file
        mtl, idx0, idxf = pu.has_block(job_inp, "material")
        viz_params = self.data["permutations"][0].keys()
        viz_params.sort()

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
        not_in = [x for x in viz_params if x.lower() not in params]
        if not_in:
            pu.logerr("Enumeration parameter[s] {0} not in model parameters"
                      .format(", ".join(not_in)))
            errors += 1

        if errors:
            return errors

        # there are no errors, now we want to find the line number in the
        # input file for the optimized params
        self.data["permute keys"] = {}
        for iline, line in enumerate([x for x in self.data["baseinp"]]):
            for viz_param in viz_params:
                if viz_param.lower() in line.lower().split():
                    self.data["permute keys"][viz_param] = iline
                continue
            continue
        # double check that data["optimize"] has a input idx for every
        # optimize variable, and set the initial value
        for key in self.data["permutations"][0].keys():
            if key not in self.data["permute keys"].keys():
                errors += 1
                pu.logerr("No input idx for enumeration variable '{0}'".format(key))
                continue
            continue

        return errors


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

            line = data["optimize"][nam]["input idx"]
            pstr = "{0} = {1:12.6E}".format(nam, opt_val*FAC[idx])
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

    with open(os.path.join(job_dir, job + ".opt"), "a") as fobj:
        fobj.write("error = {0:12.6E}\n".format(error))

    # go back to the base_dir
    os.chdir(base_dir)

    return error
