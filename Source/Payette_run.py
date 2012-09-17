# The MIT License
#
# Copyright (c) 2011 Tim Fuller
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
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

import config as cfg
import Source.Payette_utils as pu
import Source.Payette_container as pcntnr
import Source.Payette_optimize as po
import Source.Payette_permutate as pp
import Source.Payette_parameterize as pparam
import Source.Payette_input_parser as pip
import runopts as ro

def run_payette(siminp=None, restart=False, timing=False, cchar=None,
                nproc=ro.NPROC, disp=ro.DISP, verbosity=ro.VERBOSITY):
    """Main function for running a Payette job.
    Read the user inputs from argv, parse options, read user input, and run
    the jobs.

    """

    if not isinstance(siminp, (list, tuple)):
        pu.report_and_raise_error("siminp of wrong type")

    if restart:
        with open(siminp[0], "rb") as ftmp:
            the_model = pickle.load(ftmp)
            user_input_sets = (the_model, )
    else:
        # read the user input
        user_input_sets = pip.parse_user_input(siminp, cchar)
    if not user_input_sets:
        pu.report_and_raise_error("No user input found in input files")

    # we have a list of user input.  now create a generator to send to _run_job
    nsyms = len(user_input_sets) - 1
    job_inp = ((item, disp, restart, timing, idx==nsyms)
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
        tim0 = time.time()

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
        print_final_timing_info(tim0)

    retcode = 1 if any(x["retcode"] for x in return_info) else 0

    if not disp:
        # just return retcode
        return retcode

    return return_info


def _run_job(args):

    """ run each individual job """

    # pass passed args to local arguments
    user_input, disp, restart, timing, last = args

    if timing:
        tim0 = time.time()

    # instantiate Payette object
    if restart:
        the_model = user_input
        the_model.setup_restart()

    elif any("optimization" in x for x in user_input):
        # intantiate the Optimize object
        the_model = po.Optimize(user_input)

    elif any("permutation" in x for x in user_input):
        # intantiate the Optimize object
        the_model = pp.Permutate(user_input)

    elif any("parameterization" in x for x in user_input):
        # intantiate the Optimize object
        the_model = pparam.parameterizer(user_input)

    else:
        the_model = pcntnr.Payette(user_input)

    # run the job
    if timing:
        tim1 = time.time()

    siminfo = the_model.run_job()
    if disp:
        retcode = siminfo["retcode"]
    else:
        retcode = siminfo

    if retcode != 0:
        sys.stderr.write("ERROR: simulation failed\n")

    if ro.VERBOSITY and not last:
        sys.stderr.write("\n")

    if timing:
        tim2 = time.time()

    # print timing info
    if timing:
        print_timing_info(tim0, tim1, tim2, the_model.name)

    # finish up
    the_model.finish()
    del the_model

    if not disp:
        return {"retcode": retcode}

    return siminfo


def print_timing_info(tim0, tim1, tim2, name=None):
    """ print timing info """

    ttot = time.time() - tim0
    texe = tim2 - tim1
    pu.log_message(
        "\n-------------- {0} timing info --------------".format(name),
        pre="")
    pu.log_message("total problem execution time:\t{0:f}".format(texe),
                   pre="")
    pu.log_message("total simulation time:\t\t{0:f}\n".format(ttot), pre="")
    return


def print_final_timing_info(tim0):
    """ print timing info from end of run"""

    ttot = time.time() - tim0
    pu.log_message(
        "\n-------------- simulation timing info --------------", pre="")
    pu.log_message("total simulation time:\t\t{0:f}".format(ttot), pre="")
    return

