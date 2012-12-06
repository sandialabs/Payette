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

"""Main Payette simulation file.
None of the functions in this file should be called directly, but only through
the executable script in $PC_ROOT/Toolset/payette

AUTHORS
Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov

"""
import sys
import os
import pickle
import optparse
import time
import multiprocessing as mp
import logging
import re
from textwrap import fill as textfill

import Source.__config__ as cfg
import Source.Payette_utils as pu
import Source.Payette_container as pcntnr
import Source.Payette_optimize as po
import Source.Payette_permutate as pp
import Source.Payette_barf as pb
import Source.Payette_parameterize as pparam
import Source.Payette_input_parser as pip
import Source.__runopts__ as ro
from Source.Payette_container import PayetteError as PayetteError


class DummyPayette:
    def __init__(self):
        self.outfile = None
        self.name = None
        self.simdir = None
    def run_job(self):
        pass
    def finish(self):
        pass


def run_payette(siminp=None, restart=False, timing=False, barf=False,
                nproc=ro.NPROC, disp=ro.DISP, verbosity=ro.VERBOSITY,
                torun=None):
    """Main function for running a Payette job.
    Read the user inputs from argv, parse options, read user input, and run
    the jobs.

    """

    user_input_sets = []
    if restart:
        with open(restart, "rb") as ftmp:
            the_model = pickle.load(ftmp)
            user_input_sets = (the_model, )
        restart = True

    elif barf:
        user_input_sets = (barf, )
        barf = True

    else:
        if isinstance(siminp, (list, tuple)):
            siminp = "\n".join(siminp)

        # look for the "control" block from the input file
        control = []
        for item in pip.find_block("control", siminp,"").split("\n"):
            item = re.sub(r"[=:\,]", " ", item).split()
            if not item:
                continue
            try:
                key, val = item
            except ValueError:
                key, val = item[0], True
            control.append((key, val))
            continue
        if control:
            ro.set_control_options(control)

        # parse the user input
        user_input_sets = pip.parse_user_input(siminp)

    # if the user requested to run only a subset of the inputs in an input
    # file, filter out the ones not requested. we have a list of user input.
    if torun:
        names = [(j, pip.get("name", y)) for j, y in enumerate(
            user_input_sets)]
        user_input_sets = [user_input_sets[i] for i, x in names if x in torun]

    if not user_input_sets:
        pu.report_and_raise_error("No user input found in input files")

    # now create a generator to send to _run_job
    job_inp = ((item, restart, barf, timing, idx == len(user_input_sets) - 1)
               for idx, item in enumerate(user_input_sets))

    # number of processors
    nproc = min(min(mp.cpu_count(), nproc), len(user_input_sets))
    verbosity = verbosity if nproc == 1 else 0
    ro.VERBOSITY = verbosity

    # set disp to match user requested value
    ro.set_global_option("DISP", disp)

    if nproc > 1:
        pu.log_warning(
            "Running with multiple processors.  Logging to the console "
            "has been turned off.  If a job hangs, [ctrl-c] at the "
            "console will not shut down Payette.  Instead, put the job "
            "in the background with [ctrl-z] and then kill it")

    # loop through simulations and run them
    if timing:
        t0 = time.time()

    if nproc > 1 and len(user_input_sets) > 1:
        pool = mp.Pool(processes=nproc)
        return_info = pool.map(_run_job, job_inp)
        pool.close()
        pool.join()

    else:
        return_info = []
        for job in job_inp:
            return_info.append(_run_job(job))
            continue

    if timing:
        write_final_timing_info(t0)

    if not disp:
        # just return retcode
        return [x["retcode"] for x in return_info]
    return return_info


def _run_job(args):

    """ run each individual job """

    # pass passed args to local arguments
    user_input, restart, barf, timing, last = args

    if pu.error_count():
        sys.exit("ERROR: Stopping due to previous errors")

    if timing:
        t0 = time.time()

    # instantiate Payette object
    error, the_model = False, None
    try:
        if restart:
            the_model = user_input
            the_model.setup_restart()

        elif barf:
            the_model = pb.PayetteBarf(user_input)

        elif re.search(r"(?i)\boptimization\b.*", user_input):
            # intantiate the Optimize object
            the_model = po.Optimize(user_input)

        elif re.search(r"(?i)\bpermutation\b.*", user_input):
            # intantiate the Optimize object
            the_model = pp.Permutate(user_input)

        elif re.search(r"(?i)\bparameterization\b.*", user_input):
            # intantiate the Optimize object
            the_model = pparam.parameterizer(user_input)

        else:
            the_model = pcntnr.Payette(user_input)

    except PayetteError as error:
        # pass on the error for now.
        the_model = DummyPayette()
        the_model.name = pip.get("name", user_input)
        pass

    # run the job
    if timing:
        t1 = time.time()

    try:
        siminfo = the_model.run_job()

    except PayetteError as error:
        pass

    except KeyboardInterrupt:
        the_model.finish()
        sys.exit(0)

    if error:
        # the simulation failed. Trap the error and print out useful info to
        # the screen
        the_model.finish()

        if ro.DEBUG:
            # raise the error to give the traceback
            raise error

        # write out the error message
        retcode = error.retcode
        l = 86
        psf = "Payette simulation {0} failed".format(the_model.name)
        char, eee = "*", " ERROR " * 12
        em = textfill(error.message, l-6)
        em = "\n".join(["{0} {1:{pad}}{0}".format(char, x, pad=l-3)
                        for x in em.split("\n")])
        pu.report_error("{0:{0}^{width}}\n{0}{2:^{pad2}}{0}\n{0}{0:>{pad}}\n"
                        "{0}{1:^{pad2}}{0}\n{0}{0:>{pad}}\n{3}\n"
                        "{0}{0:>{pad}}\n{0}{2:^{pad2}}{0}\n{0:{0}^{width}}"
                        .format(char, psf, eee, em, width=l, pad=l-1, pad2=l-2),
                        count=ro.ERROR.lower() != "ignore", anonymous=True,
                        pre="")
        # populate siminfo dict to return
        siminfo = {"retcode": retcode,
                   "output file": the_model.outfile,
                   "extra files": {},
                   "simulation name": the_model.name,
                   "simulation directory": the_model.simdir, }
    retcode = siminfo["retcode"]

    if ro.VERBOSITY and not last:
        sys.stderr.write("\n")

    if timing:
        t2 = time.time()

    # write timing info
    if timing:
        write_timing_info(t0, t1, t2, the_model.name)

    # finish up
    the_model.finish()

    del the_model

    return siminfo


def write_timing_info(t0, t1, t2, name=None):
    """ write timing info """

    ttot = time.time() - t0
    texe = t2 - t1
    pu.log_message(
        "\n-------------- {0} timing info --------------".format(name),
        pre="")
    pu.log_message("total problem execution time:\t{0:f}".format(texe),
                   pre="")
    pu.log_message("total simulation time:\t\t{0:f}\n".format(ttot), pre="")
    return


def write_final_timing_info(t0):
    """ writ timing info from end of run"""

    ttot = time.time() - t0
    pu.log_message(
        "\n-------------- simulation timing info --------------", pre="")
    pu.log_message("total simulation time:\t\t{0:f}".format(ttot), pre="")
    return
