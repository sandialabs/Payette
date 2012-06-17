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
the executable script in $PC_ROOT/Toolset/runPayette

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

import Payette_config as pc
import Source.Payette_utils as pu
import Source.Payette_container as pcntnr
import Source.Payette_optimize as po
import Source.Payette_permutate as pp
import Source.Payette_input_parser as pip
import Source.runopts as ro

def run_payette(argv, disp=0):
    """Main function for running a Payette job.
    Read the user inputs from argv, parse options, read user input, and run
    the jobs.

    """

    # Make sure that everything is clean before we start
    pu.reset_error_and_warnings()
    ro.restore_default_options()

    # ************************************************************************
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
        "--disp",
        dest="disp",
        action="store",
        type=int,
        default=disp,
        help=("Return extra diagnostic information from run_job if disp > 0 "
              "[default: %default]"))
    parser.add_option(
        "--input-str",
        dest="inputstr",
        action="store",
        default=None,
        help=("Input string for simulation instead of file "
                "[default: %default]"))
    parser.add_option(
        "-m", "--materials",
        dest="mtls",
        default=False,
        action="store_true")
    parser.add_option(
        "-p", "--princ",
        dest="principal",
        action="store_true",
        default=False,
        help=("Diagonalize input arguments and run problem in "
                "principal coordinates [default: %default]"))
    parser.add_option(
        "-t",
        dest="timing",
        action="store_true",
        default=False,
        help="time execution of Payette runs [default: %default]")

    # the following options have defaults set in runopt.py, later, we pass the
    # user requested options back to runopt.py so they are set of the rest of
    # the modules used by runPayette
    parser.add_option(
        "-v", "--verbosity",
        dest="verbosity",
        type="choice",
        choices=["0", "1", "2", "3", "4"],
        default=str(ro.VERBOSITY),
        action="store",
        help="Verbosity default: %default]")
    parser.add_option(
        "-S", "--sqa",
        dest="sqa",
        action="store_true",
        default=ro.SQA,
        help="Run additional verification/sqa checks [default: %default]")
    parser.add_option(
        "-d", "--dbg", "--debug",
        dest="debug",
        action="store_true",
        default=ro.DEBUG,
        help="Global debug flag [default: %default]")
    parser.add_option(
        "-s", "--strict",
        dest="strict",
        action="store_true",
        default=ro.STRICT,
        help=("Do not use approximations to update kinematic "
                "quantities (slow) [default: %default]"))
    parser.add_option(
        "--write-restart",
        dest="writerestart",
        action="store_true",
        default=ro.WRITERESTART,
        help="Do not save restart files [default: %default]")
    parser.add_option(
        "--no-writeprops",
        dest="nowriteprops",
        action="store_true",
        default=ro.NOWRITEPROPS,
        help="Do not write checked parameters [default: %default]")
    parser.add_option(
        "--simdir",
        dest="simdir",
        action="store",
        default=ro.SIMDIR,
        help="Directory to run simulation [default: {0}]".format(os.getcwd()))
    parser.add_option(
        "-T", "--use-table",
        dest="use_table",
        action="store_true",
        default=ro.USE_TABLE,
        help=("Update kinematic quantities from input when "
              "applicable [default: %default]"))
    parser.add_option(
        "-k", "--keep",
        dest="keep",
        action="store_true",
        default=ro.KEEP,
        help=("Do not overwrite old output files with each run "
                "[default: %default]"))
    parser.add_option(
        "--write-vandd",
        dest="write_vandd_table",
        action="store_true",
        default=ro.WRITE_VANDD_TABLE,
        help=("Write equivalent velocity and displacement table "
              "[default: %default]"))
    parser.add_option(
        "--test-restart",
        dest="testrestart",
        action="store_true",
        default=ro.TESTRESTART,
        help="Test restart capabilities [default: %default]")
    parser.add_option(
        "--proportional",
        dest="proportional",
        action="store_true",
        default=ro.PROPORTIONAL,
        help=("Use proportional loading for prescribed stress"
                "components. [default: %default]"))
    parser.add_option(
        "-j", "--nproc",
        dest="nproc",
        type=int,
        default=ro.NPROC,
        action="store",
        help="Number of simultaneous jobs [default: %default]")
    parser.add_option(
        "--check-setup",
        dest="check_setup",
        action="store_true",
        default=ro.CHECK_SETUP,
        help=("Set up material and exit, printing set up information "
              "[default: %default]"))
    parser.add_option(
        "-w", "--write-input",
        dest="write_input",
        action="store_true",
        default=ro.WRITE_INPUT,
        help="Write input file for simulation [default: %default]")
    parser.add_option(
        "-W",
        dest="warning",
        type="choice",
        action="store",
        choices=["ignore", "warn", "error", "all"],
        default=ro.WARNING,
        help="warning level [default: %default]")
    parser.add_option(
        "--test-error",
        dest="testerror",
        action="store_true",
        default=ro.TESTERROR,
        help="test raising error [default: %default]")

    # parse the command line arguments
    (opts, args) = parser.parse_args(argv)

    payette_exts = [".log", ".math1", ".math2", ".props", ".echo", ".prf"]
    if opts.cleanall:
        payette_exts.extend([".out"])
        opts.clean = True

    if opts.debug:
        opts.verbosity = 4

    opts.verbosity = int(opts.verbosity)

    # pass command line arguments to global Payette variables
    ro.set_command_line_options(opts)

    if opts.clean:
        cleaned = False
        # clean all the payette output and exit
        for arg in args:
            argdir = os.path.dirname(os.path.realpath(arg))
            argnam = os.path.splitext(arg)[0]
            if argnam not in [os.path.splitext(x)[0]
                              for x in os.listdir(argdir)]:
                pu.log_warning("no Payette output for {0} found in {1}"
                               .format(arg, argdir))
                continue
            pu.log_message("cleaning output for {0}".format(argnam))
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
    if opts.verbosity:
        pu.log_message(pc.PC_INTRO, pre="")

    input_lines = []
    if opts.inputstr:
        # user gave input directly
        input_lines.extend(opts.inputstr.split("\n"))

    # make sure input file is given and exists
    if len(args) < 1 and not input_lines:
        parser.print_help()
        parser.error("No input given")

    # pass options to global module variables
    timing = opts.timing

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

            PayetteBarf(barf_file)

        return 0

    # check restart files
    rfiles = [x for x in args if os.path.splitext(x)[1] == ".prf"]
    restart = any(rfiles)
    if rfiles:
        nrfiles = len(rfiles)
        if nrfiles > 1:
            parser.error(
                str(len(rfiles)) + " restart files given, "
                "but only 1 restart file can be processed at a time")

        if nrfiles != len(args):
            # choose to run a single restart file or input files, but don't mix
            parser.error(
                "Cannot process both restart and input files at same time")

        # check to see if the input file exists and get it info
        rfile = rfiles[0]
        if not os.path.isfile(rfile):
            parser.error("Restart file {0} not found".format(rfile))

        with open(rfile, "rb") as ftmp:
            the_model = pickle.load(ftmp)
        user_input_sets = (the_model,)

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

            elif os.path.isfile(os.path.join(pc.PC_INPUTS, arg)):
                ftmp = os.path.join(pc.PC_INPUTS, arg)
                pu.log_message("Using " + ftmp + " as input")

            elif not fext or fext == ".":
                # add .inp extension to arg
                arginp = fbase + ".inp"
                if os.path.isfile(arginp):
                    ftmp = arginp
                    pu.log_message("Using " + ftmp + " as input")

                elif os.path.isfile(os.path.join(pc.PC_INPUTS, arginp)):
                    ftmp = os.path.join(pc.PC_INPUTS, arginp)
                    pu.log_message("Using " + ftmp + " as input")

            if not ftmp:
                pu.log_warning("{0} not found in {1}, {2}, or {3}"
                               .format(arg,
                                       os.path.dirname(os.path.realpath(arg)),
                                       os.getcwd(), pc.PC_INPUTS))
                badf.append(arg)
                continue

            if ftmp in foundf:
                parser.error("{0} given multiple times".format(arg))

            foundf.append(ftmp)
            continue

        if badf:
            pu.log_warning("The following files were not found: {0}"
                           .format(", ".join(badf)))

        if not foundf and not input_lines:
            parser.print_help()
            parser.error("No input files found")

        # put contents of input files in to input_lines
        for ftmp in foundf:
            input_lines.extend(open(ftmp, "r").readlines())
            continue

        # read the user input
        user_input_sets = pip.parse_user_input(input_lines, opts.cchar)
        if not user_input_sets:
            pu.report_and_raise_error(
                "user input not found in {0:s}".format(", ".join(foundf)))

    # ----------------------------------------------------- end: get user input

    # we have a list of user input.  now create a generator to send to _run_job
    nsyms = len(user_input_sets) - 1
    job_inp = ((item, opts, restart, timing, idx==nsyms)
               for idx, item in enumerate(user_input_sets))

    # number of processors
    nproc = min(min(mp.cpu_count(), opts.nproc), len(user_input_sets))
    opts.verbosity = opts.verbosity if nproc == 1 else 0
    ro.NPROC = nproc
    ro.VERBOSITY = opts.verbosity

    if nproc > 1:
        pu.log_warning("""
    Running with multiple processors.  Logging to the console
    has been turned off.  If a job hangs, [ctrl-c] at the
    console will not shut down Payette.  Instead, put the job
    in the background with [ctrl-z] and then kill it""")

    # loop through simulations and run them
    if timing:
        tim0 = time.time()

    if nproc > 1 and len(user_input_sets) > 1:
        pool = mp.Pool(processes=nproc)
        retcodes = pool.map(_run_job, job_inp)
        pool.close()
        pool.join()

    else:
        retcodes = []
        for job in job_inp:
            retcodes.append(_run_job(job))
            continue

    if timing:
        print_final_timing_info(tim0)

    retcode = 1 if any(retcodes) else 0
    return retcode


def _run_job(args):

    """ run each individual job """

    # pass passed args to local arguments
    user_input, opts, restart, timing, last = args

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

    else:
        the_model = pcntnr.Payette(user_input)

    # run the job
    if timing:
        tim1 = time.time()

    solve = the_model.run_job()

    if opts.disp:
        retcode = solve["retcode"]
    else:
        retcode = solve

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

    return retcode


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
    return None


def print_final_timing_info(tim0):

    """ print timing info from end of run"""

    ttot = time.time() - tim0
    pu.log_message(
        "\n-------------- simulation timing info --------------", pre="")
    pu.log_message("total simulation time:\t\t{0:f}".format(ttot), pre="")
    return None


if __name__ == "__main__":

    if not os.path.isfile(pc.PC_MTLS_FILE):
        pu.report_and_raise_error(
            "buildPayette must be executed before tests can be run")

    ARGV = sys.argv[1:]

    if "--profile" in ARGV:
        PROFILE = True
        ARGV.remove("--profile")
        import profile
    else:
        PROFILE = False

    if PROFILE:
        CMD = "run_payette(ARGV)"
        PROF = "payette.prof"
#        profile.runctx(CMD, globals(), locals(), PROF)
        profile.run(CMD)
        PAYETTE = 0
    else:
        PAYETTE = run_payette(ARGV)

    sys.exit(PAYETTE)

