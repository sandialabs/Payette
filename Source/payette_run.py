#!/usr/bin/env python
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

""" setup and run a Payette simulation """

from __future__ import print_function
import sys
import os
import pickle
import optparse
import time
import multiprocessing as mp

from Payette_utils import PAYETTE_MATERIALS_FILE, PAYETTE_INTRO
from Payette_utils import PAYETTE_INPUTS
from Payette_utils import logerr, logwrn, loginf, logmes
from Payette_utils import writeMessage, writeWarning, readUserInput
from Payette_utils import reportMessage

def run_payette(argv):
    """ run the user input """

    import Source.Payette_driver as pdrvr
    import Source.Payette_container as pcntnr

    def _run_job(job):
        """ run each individual job """

        iam = "run_payette"

        if opts.timing:
            tim0 = time.time()

        # instantiate Payette object
        if restart:
            the_model = user_input_dict[job]
            the_model.setupRestart()
        else:
            the_model = pcntnr.Payette(job, user_input_dict[job], opts)

        # run the problem
        if opts.timing:
            tim1 = time.time()

        solve = pdrvr.runProblem(the_model, restart=restart)

        if solve != 0:
            sys.exit("ERROR: simulation failed")

        if opts.timing:
            tim2 = time.time()

        # finish up
        the_model.finish()
        del the_model

        # print timing info
        if opts.timing:
            print_timing_info(tim0, tim1, tim2, the_model.name)

        return 0

    # *************************************************************************
    # -- command line option parsing
    usage = "usage: runPayette [options] <input file>"
    parser = optparse.OptionParser(usage=usage, version="runPayette 1.0")
    parser.add_option(
        "--clean",
        dest="clean",
        action="store_true",
        default=False,
        help="Clean Payette auxilary output and exit [default: %default]")
    parser.add_option(
        "--cleanall",
        dest="cleanall",
        action="store_true",
        default=False,
        help="Clean ALL Payette output and exit [default: %default]")
    parser.add_option(
        "--cchar",
        dest="cchar",
        action="store",
        default=None,
        help=("Additional comment characters for input file "
                "[default: %default]"))
    parser.add_option(
        "-d", "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Global debug flag [default: %default]")
    parser.add_option(
        "--input-str",
        dest="inputstr",
        action="store",
        default=None,
        help=("Input string for simulation instead of file "
                "[default: %default]"))
    parser.add_option(
        "-j", "--nproc",
        dest="nproc",
        type=int,
        default=1,
        action="store",
        help="Number of simultaneous jobs [default: %default]")
    parser.add_option(
        "-k", "--keep",
        dest="keep",
        action="store_true",
        default=False,
        help=("Do not overwrite old output files with each run "
                "[default: %default]"))
    parser.add_option(
        "-m", "--materials",
        dest="mtls",
        default=False,
        action="store_true")
    parser.add_option(
        "--no-restart",
        dest="norestart",
        action="store_true",
        default=False,
        help="Do not save restart files [default: %default]")
    parser.add_option(
        "--no-writeprops",
        dest="nowriteprops",
        action="store_true",
        default=False,
        help="Do not write checked parameters [default: %default]")
    parser.add_option(
        "-p", "--princ",
        dest="principal",
        action="store_true",
        default=False,
        help=("Diagonalize input arguments and run problem in "
                "principal coordinates [default: %default]"))
    parser.add_option(
        "--proportional",
        dest="proportional",
        action="store_true",
        default=False,
        help=("Use proportional loading for prescribed stress"
                "components. [default: %default]"))
    parser.add_option(
        "-s", "--strict",
        dest="strict",
        action="store_true",
        default=False,
        help=("Do not use approximations to update kinematic "
                "quantities (slow) [default: %default]"))
    parser.add_option(
        "-S", "--sqa",
        dest="sqa",
        action="store_true",
        default=False,
        help="Run additional verification/sqa checks [default: %default]")
    parser.add_option(
        "-t",
        dest="timing",
        action="store_true",
        default=False,
        help="time execution of Payette runs [default: %default]")
    parser.add_option(
        "-T", "--use-table",
        dest="use_table",
        action="store_true",
        default=False,
        help=("Update kinematic quantities from input when "
              "applicable [default: %default]"))
    parser.add_option(
        "--test-restart",
        dest="testrestart",
        action="store_true",
        default=False,
        help="Test restart capabilities [default: %default]")
    parser.add_option(
        "-v", "--verbosity",
        dest="verbosity",
        type="choice",
        choices=["0", "1", "2", "3", "4"],
        default="3",
        action="store",
        help="Verbosity default: %default]")
    parser.add_option(
        "-w", "--write-vandd",
        dest="write_vandd_table",
        action="store_true",
        default=False,
        help=("Write equivalent velocity and displacement table "
              "[default: %default]"))
    (opts, args) = parser.parse_args(argv)

    if not os.path.isfile(PAYETTE_MATERIALS_FILE):
        logerr("buildPayette must be run to generate "
               "Source/Materials/Payette_installed_materials.py")
        sys.exit(130)

    payette_exts = [".log", ".math1", ".math2", ".props", ".echo", ".prf"]
    if opts.cleanall:
        payette_exts.extend([".out"])
        opts.clean = True

    opts.verbosity = int(opts.verbosity)

    if opts.clean:
        cleaned = False
        # clean all the payette output and exit
        for arg in args:
            argdir = os.path.dirname(os.path.realpath(arg))
            argnam = os.path.splitext(arg)[0]
            if argnam not in [os.path.splitext(x)[0]
                              for x in os.listdir(argdir)]:
                logwrn("no Payette output for {0} found in {1}"
                       .format(arg, argdir))
                continue
            loginf("cleaning output for {0}".format(argnam))
            for ext in payette_exts:
                try:
                    os.remove(argnam + ext)
                    cleaned = True
                except OSError:
                    pass
                continue
            continue
        msg = "INFO: output cleaned" if cleaned else "WARNING: not cleaned"
        sys.exit(msg)

    # ----------------------------------------------- start: get the user input
    restart = False
    input_lines = []
    if opts.inputstr:
        # user gave input directly
        input_lines.extend(opts.inputstr.split("\n"))

    # make sure input file is given and exists
    if len(args) < 1 and not input_lines:
        parser.print_help()
        parser.error("No input given")

    # first check for barf files
    barf_files = [x for x in args if "barf" in os.path.splitext(x)[1]
                  or "barf.source" in x]
    if len(barf_files):
        if len(barf_files) > 1:
            parser.error(
                "{0:d} barf files given, but only one "
                "barf file can be processed at a time".format(len(barf_files)))
        else:
            from Source.Payette_barf import PayetteBarf
            barf_file = os.path.realpath(barf_files[0])
            if not os.path.isfile(barf_file):
                parser.error("barf file {0} not found".format(barf_file))

            if opts.verbosity:
                logmes(PAYETTE_INTRO)

            PayetteBarf(barf_file)

        return 0

    # check restart files
    rfiles = [x for x in args if os.path.splitext(x)[1] == ".prf"]
    if len(rfiles) > 1:
        parser.error(str(len(rfiles)) + " restart files given, but only one "
                     "restart file can be processed at a time")

    elif len(rfiles) and len(rfiles) != len(args):
        # choose to run a single restart file or input files, but don't mix
        parser.error(
            "Cannot process both restart and input files at same time")

    elif len(rfiles):
        # check to see if the input file exists and get it info
        rfile = rfiles[0]
        if not os.path.isfile(rfile):
            parser.error("Restart file {0} not found".format(rfile))

        restart = True
        with open(rfile, "rb") as ftmp:
            the_model = pickle.load(ftmp)
        user_input_dict = {"restart": the_model}

    else:
        # get a list of all input files, order of where to look for file f:
        # 1: f
        # 2: realpath(f)
        # 3: join(Aux/Inputs, f)
        # 4: splitext(f)[0] + .inp
        # 5: join(Aux/Inputs, splitext(f)[0] + .inp)
        foundf, badf = [], []
        for arg in args:
            ftmp = None
            fbase, fext = os.path.splitext(arg)
            if not arg:
                continue
            if fext == ".prf":
                # one last check for restart files
                parser.error("Cannot process both restart and input "
                             "files at same time")
            elif os.path.isfile(arg):
                ftmp = arg
            elif os.path.isfile(os.path.realpath(arg)):
                ftmp = os.path.realpath(arg)
            elif os.path.isfile(os.path.join(PAYETTE_INPUTS, arg)):
                ftmp = os.path.join(PAYETTE_INPUTS, arg)
                writeMessage(__file__, "Using " + ftmp + " as input")
            elif not fext or fext == ".":
                # add .inp extension to arg
                arginp = fbase + ".inp"
                if os.path.isfile(arginp):
                    ftmp = arginp
                    writeMessage(__file__, "Using " + ftmp + " as input")
                elif os.path.isfile(os.path.join(PAYETTE_INPUTS, arginp)):
                    ftmp = os.path.join(PAYETTE_INPUTS, arginp)
                    writeMessage(__file__, "Using " + ftmp + " as input")

            if not ftmp:
                writeWarning(__file__, "{0} not found in {1}, {2}, or {3}"
                             .format(arg,
                                     os.path.dirname(os.path.realpath(arg)),
                                     os.getcwd(),
                                     PAYETTE_INPUTS))
                badf.append(arg)
                continue

            if ftmp in foundf:
                parser.error("{0} given multiple times".format(arg))

            foundf.append(ftmp)
            continue

        if badf:
            writeWarning(__file__, "The following files were not found: {0}"
                         .format(", ".join(badf)))

        if not foundf and not input_lines:
            parser.print_help()
            parser.error("No input files found")

        # put contents of input files in to input_lines
        for ftmp in foundf:
            input_lines.extend(open(ftmp, "r").readlines())
            continue

        # read the user input
        user_input_dict = readUserInput(input_lines, opts.cchar)
        if not user_input_dict:
            sys.exit("ERROR: user input not found in {0:s}"
                     .format(", ".join(foundf)))
    # ----------------------------------------------------- end: get user input

    # number of processors
    nproc = min(min(mp.cpu_count(), opts.nproc), len(user_input_dict))
    opts.verbosity = opts.verbosity if nproc == 1 else 0

    if nproc > 1:
        writeWarning(__file__, """
             Running with multiple processors.  Logging to the console
             has been turned off.  If a job hangs, [ctrl-c] at the
             console will not shut down Payette.  Instead, put the job
             in the background with [ctrl-z] and then kill it""")

    if opts.verbosity:
        logmes(PAYETTE_INTRO)

    # loop through simulations and run them
    if opts.timing:
        tim0 = time.time()
    if nproc > 1 and len(user_input_dict.keys()) > 1:
        pool = mp.Pool(processes=nproc)
        pool.map(_run_job, user_input_dict.keys())
        pool.close()
        pool.join()
    else:
        for key in user_input_dict:
            _run_job(key)
            continue

    if opts.timing:
        print_final_timing_info(tim0)

    return 0


def print_timing_info(tim0, tim1, tim2, name=None):
    """ print timing info """

    ttot = time.time() - tim0
    texe = tim2 - tim1
    logmes("\n-------------- {0} timing info --------------".format(name))
    logmes("total problem execution time:\t{0:f}".format(texe))
    logmes("total simulation time:\t\t{0:f}\n".format(ttot))
    return None


def print_final_timing_info(tim0):
    """ print timing info from end of run"""

    ttot = time.time() - tim0
    logmes("\n-------------- simulation timing info --------------")
    logmes("total simulation time:\t\t{0:f}".format(ttot))
    return None


if __name__ == "__main__":

    ARGV = sys.argv[1:]

    if "--profile" in ARGV:
        PROFILE = True
        ARGV.remove("--profile")
        import cProfile
    else:
        PROFILE = False

    if PROFILE:
        CMD = "run_payette(ARGV)"
        PROF = "payette.prof"
        cProfile.runctx(CMD, globals(), locals(), PROF)
        PAYETTE = 0
    else:
        PAYETTE = run_payette(ARGV)

    sys.exit(PAYETTE)

