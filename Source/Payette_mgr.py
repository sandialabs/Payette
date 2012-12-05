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

""" Top level interface to the Payette material model driver. """
import sys
import os
import optparse
from optparse import OptionParser, BadOptionError, AmbiguousOptionError
from textwrap import fill as textfill

FILE = os.path.realpath(__file__)
ROOTDIR = os.path.realpath(os.path.join(os.path.dirname(FILE), "../"))
SRCDIR = os.path.join(ROOTDIR, "Source")
EXENAM = "payette"

# import Payette specific files
try:
    import Source.__config__ as cfg
except ImportError:
    sys.exit("Payette must first be configured before execution")
import Source.__runopts__ as ro
import Source.Payette_utils as pu
import Source.Payette_model_index as pmi
from Source.Payette_run import run_payette

USAGE = "" if "-H" in sys.argv or "--man" in sys.argv else """\
{0}: top level interface to the Payette material model driver
usage: {0} [file_0.ext [file_1.ext [... file_n.ext]]]""".format(EXENAM)

MAN_PAGE = """\
NAME
      payette - setup and launch the Payette simulation

USAGE
      payette [OPTIONS] [input files]

DESCRIPTION
      The payette script is a wrapper around several Payette scripts that
      control the Payette GUI and physics driver.  It sets the necessary
      environment variables and builds the Payette command line consistent
      with the configure tool.

      If no options or input files are passed to payette, then the payette
      viz model selector GUI is launched and simulations can be set up and
      run in it.  Otherwise, Payette is driven from the command line.

INSTALLED MODELS
{{0}}

CONFIGURATION INFO
      Python interpreter: {0}

OPTIONS
      There are a set of options recognized by the payette script, which are
      listed here.  If an option is given which is not listed here, it is
      passed through to the different Payette modules.
""".format(cfg.PYINT)


class PassThroughOptionParser(OptionParser):
    """
    An unknown option pass-through implementation of OptionParser.

    When unknown arguments are encountered, bundle with largs and try again,
    until rargs is depleted.

    sys.exit(status) will still be called if a known argument is passed
    incorrectly (e.g. missing arguments or bad argument types, etc.)

    Copied from
    http://stackoverflow.com/questions/1885161/
          how-can-i-get-optparses-optionparser-to-ignore-invalid-arguments
    """
    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                OptionParser._process_args(self, largs, rargs, values)
            except (BadOptionError, AmbiguousOptionError), e:
                largs.append(e.opt_str)


def main(argv):
    """
    The main gateway to the Payette driver and associated tools

    Parameters
    ----------
    argv : list
        command line arguments

    Returns
    -------
    None

    """

    # Make sure that everything is clean before we start
    pu.reset_error_and_warnings()
    ro.restore_default_options()

    # --------------------------------------------- command line option parsing
    parser = PassThroughOptionParser(usage=USAGE, version="payette 1.0")
    parser.add_option(
        "-H", "--man",
        dest="MAN",
        action="store_true",
        default=False,
        help="Print man page and exit [default: %default]")
    parser.add_option(
        "--clean",
        dest="CLEAN",
        action="store_true",
        default=False,
        help="Clean Payette auxilary output and exit [default: %default]")
    parser.add_option(
        "--cleanall",
        dest="CLEANALL",
        action="store_true",
        default=False,
        help="Clean ALL Payette output and exit [default: %default]")
    parser.add_option(
        "--summary",
        dest="SUMMARY",
        action="store_true",
        default=False,
        help="write summary to screen [default: %default]")
    parser.add_option(
        "--input-str",
        dest="inputstr",
        action="store",
        default=None,
        help=("Input string for simulation instead of file "
              "[default: %default]"))
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
    parser.add_option(
        "-V",
        dest="VIZ",
        action="store_true",
        default=False,
        help="Display visualization window upon completion [default: %default]")
    parser.add_option(
        "-C",
        dest="VIZCNTRL",
        action="store_true",
        default=False,
        help="Launch the Viz controller [default: %default]")
    parser.add_option(
        "-A",
        dest="AUXMTL",
        action="store",
        default=None,
        help=("Alternate directory to find material database "
              "file [default: %default]"))
    parser.add_option(
        "-N",
        dest="NAMES",
        action="append",
        default=[],
        help=("Simulations to run from input file [default: ALL]"))

    # the following options have defaults set in runopt.py, later, we pass the
    # user requested options back to runopt.py so they are set of the rest of
    # the modules used by Payette
    parser.add_option(
        "-v", "--verbosity",
        dest="verbosity",
        type=int,
        default=ro.VERBOSITY,
        action="store",
        help="Verbosity default: %default]")
    parser.add_option(
        "-S", "--sqa",
        dest="sqa",
        action="store_true",
        default=ro.SQA,
        help="Run additional verification/sqa checks [default: %default]")
    parser.add_option(
        "--dbg", "--debug",
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
        help="Write restart files [default: %default]")
    parser.add_option(
        "--no-writeprops",
        dest="nowriteprops",
        action="store_true",
        default=ro.NOWRITEPROPS,
        help="Do not write checked parameters [default: %default]")
    parser.add_option(
        "-d", "--simdir",
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
        help="Do not overwrite old output with each run [default: %default]")
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
        "--disp",
        dest="disp",
        action="store",
        default=ro.DISP,
        help="Return extra diagnositic information if > 0 [default: %default]")
    parser.add_option(
        "--write-input",
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
        "-E",
        dest="ERROR",
        type="choice",
        action="store",
        choices=["stop", "ignore"],
        default=ro.ERROR,
        help="Error level [default: %default]")

    # parse the command line arguments
    (opts, args) = parser.parse_args(argv)
    # ----------------------------------------- end command line option parsing

    if opts.SUMMARY:
        # write the summary
        sys.exit(_write_summary_to_screen())

    if opts.debug:
        # for debug problems, increase verbosity
        opts.verbosity = 4

    if opts.CLEAN or opts.CLEANALL:
        sys.exit(_clean_file_exts(args, opts.CLEANALL))

    # ----------------------------------------------- start: get the user input
    if opts.AUXMTL is not None:
        if os.path.isfile(opts.AUXMTL):
            material_db = opts.AUXMTL
        elif os.path.isfile(os.path.join(opts.AUXMTL, cfg.AUXDB)):
            material_db = os.path.join(opts.AUXMTL, cfg.AUXDB)
        elif os.path.isfile(os.path.join(cfg.DOTPAYETTE, opts.AUXMTL)):
            material_db = os.path.join(cfg.DOTPAYETTE, opts.AUXMTL)
        elif os.path.isfile(os.path.join(cfg.DOTPAYETTE, opts.AUXMTL + ".db")):
            material_db = os.path.join(cfg.DOTPAYETTE, opts.AUXMTL + ".db")
        else:
            pu.report_and_raise_error("{0} not found".format(opts.AUXMTL))
        sys.path.insert(0, os.path.dirname(material_db))
    else:
        material_db = cfg.MTLDB

    if not os.path.isfile(material_db):
        pu.report_and_raise_error(
            "buildPayette must first be executed to create\n\t{0}"
            .format(material_db))

    if opts.MAN:
        # print the man page for this and other scripts
        installed_models = pmi.ModelIndex(material_db).constitutive_models()
        if installed_models:
            installed_models = "{0}".format(
                textfill(", ".join(installed_models), initial_indent=" " * 6,
                         subsequent_indent=" " * 6))
        else:
            installed_models = ("      buildPayette must be executed to build "
                                "and install models")
        pu.log_message(MAN_PAGE.format(installed_models), pre="")
        sys.exit(parser.print_help())

    # pass command line arguments to global Payette variables
    ro.set_command_line_options(opts)
    ro.set_global_option("MTLDB", material_db, default=True)

    if opts.VIZCNTRL:
        if not cfg.VIZ_COMPATIBLE:
            sys.exit("Visualization not supported by your Python distribution")
        import Source.Viz_Control as vc
        window = vc.ControlWindow()
        sys.exit(window.configure_traits())

    if not argv:
        if not cfg.VIZ_COMPATIBLE:
            sys.exit("Visualization not supported by your Python distribution")
        # Launch the Gui and exit
        import Source.Viz_ModelSelector as vms
        window = vms.PayetteMaterialModelSelector(model_type="any")
        sys.exit(window.configure_traits())

    # determine file type given, whether output files for viewing, input files
    # for running, or barf files for barf processing
    oexts = (".out", ".dat", ".pkl", ".gold")
    iexts = (".inp", ".i",)
    rexts = (".prf", )
    oargs, iargs, uargs, bargs, rargs = [], [], [], [], []
    for arg in list(set(args)):
        farg = os.path.realpath(os.path.expanduser(arg))

        # check for file existence
        if not os.path.isfile(farg):
            if os.path.isdir(farg):
                uargs.append(farg)
                continue
            if os.path.isfile(farg.rstrip(".") + ".inp"):
                farg = farg.rstrip(".") + ".inp"
            else:
                pu.log_warning("{0} not found".format(farg))

        fnam, fext = os.path.splitext(farg)
        if fext in oexts:
            # output file
            oargs.append(farg)

        elif fext in iexts:
            # input file
            iargs.append(farg)

        elif "barf" in fext:
            # barf file
            bargs.append(farg)

        elif fext in rexts:
            # restart file
            rargs.append(farg)

        else:
            # unkown file type
            uargs.append(farg)
        continue

    if pu.warn_count():
        pu.report_and_raise_error("Stopping due to previous errors")

    # check for incompatibilities
    if uargs:
        for uarg in uargs:
            if os.path.isdir(uarg) and "index.pkl" in os.listdir(uarg):
                oargs.append(os.path.join(uarg, "index.pkl"))
                continue
            pu.report_and_raise_error(
                "Files with unkown file extension[s] passed to Payette [{0}]"
                .format(os.path.splitext(x)[1] for x in uargs))

    if len([x for x in (oargs, iargs, bargs, rargs) if x]) > 1:
        pu.report_and_raise_error(
            "Payette can only process one file type request at a time")

    if oargs:
        # output files given, launch visualizer
        pu.log_message(cfg.INTRO, pre="")
        sys.exit(_visualize_results(outfiles=oargs))

    if opts.verbosity:
        pu.log_message(cfg.INTRO, pre="")
        pu.log_message("Using material database {0}"
                       .format(os.path.basename(material_db)))

    # We are now to the point where we will call run_payette, this could be
    # with either a restart file, or with the contents of input files.
    siminp = None
    if opts.inputstr:
        # user gave input directly
        siminp = opts.inputstr

    restart, barf = False, False
    if rargs:
        if len(rargs) > 1:
            pu.report_and_raise_error(
                "{0:d} restart files given, but only 1 restart file "
                "can be processed at a time".format(len(rargs)))
        elif siminp is not None:
            pu.report_and_raise_error(
                "Restart files cannot be run with additional input")
        restart = rargs[0]

    elif bargs:
        # user passed in a barf file
        if len(bargs) > 1:
            pu.report_and_raise_error(
                "{0:d} barf files given, but only 1 barf file "
                "can be processed at a time".format(len(bargs)))
        barf = bargs[0]

    elif iargs:
        # go through input files and load contents
        siminp = "" if siminp is None else siminp
        for iarg in iargs:
            siminp += open(iarg, "r").read()
            continue
    # ----------------------------------------------------- end: get user input

    # make sure input file is given and exists
    if not siminp and not restart and not barf:
        parser.print_help()
        parser.error("No input given")

    # call the run_payette function
    siminfo = run_payette(siminp=siminp, restart=restart, barf=barf,
                          timing=opts.timing, nproc=opts.nproc, disp=opts.disp,
                          verbosity=opts.verbosity, torun=opts.NAMES)

    # visualize the results if requested
    if opts.VIZ:
        simulation_info = [x for x in siminfo if x["retcode"] == 0]
        sys.exit(_visualize_results(simulation_info=siminfo))

    retcodes = [x["retcode"] for x in siminfo]
    if len(retcodes) == 1:
        return retcodes[0]
    return retcodes


def _visualize_results(simulation_info=None, outfiles=None):
    """visualize the results from a simulation

    Parameters
    ----------
    simulation_info : list
       list of return dictionaries

    outfiles : list
       list of output files to visualize
    """
    if not cfg.VIZ_COMPATIBLE:
        pu.log_warning(
            "Visualization not supported by your Python distribution")
        return

    from Viz_ModelPlot import create_Viz_ModelPlot
    import Source.Payette_sim_index as psi

    if len([x for x in [simulation_info, outfiles] if x is not None]) > 1:
        pu.log_warning("Cannot specify both outfiles and simulation_info")
        return

    if outfiles is not None:
        if len(outfiles) == 1 and os.path.basename(outfiles[0]) == "index.pkl":
            simulation_info = [
                {"simulation name": "Payette", "index file": outfiles[0]}]
        else:
            simulation_info = []
            warned = False
            for outfile in outfiles:
                if not os.path.isfile(outfile):
                    pu.log_warning("{0} not found".format(outfile))
                    warned = True
                    continue
                fdir, fnam = os.path.split(outfile)
                simname = os.path.splitext(fnam)[0]
                simulation_info.append({"simulation name": simname,
                                        "simulation directory": fdir,
                                        "output file": outfile, })
                continue
            if warned:
                return

    if len(simulation_info) == 1:
        # only a single simulation, get the siminfo and simname directly
        siminfo = simulation_info[0]
        simname = siminfo["simulation name"]

    else:
        # multiple simulations, create an index file
        index = psi.SimulationIndex(os.getcwd())
        simname = "Payette"
        for idx, info in enumerate(simulation_info):
            name = info["simulation name"]
            simdir = info["simulation directory"]
            outfile = info["output file"]
            variables = {}
            index.store(idx, name, simdir, variables, outfile)
            continue
        index.dump()
        siminfo = {"index file": index.index_file()}
    create_Viz_ModelPlot(simname, **siminfo)
    return


def _write_summary_to_screen():
    """ write summary of entire Payette project to the screen """

    def _num_code_lines(fpath):
        """ return the number of lines of code in fpath """
        nlines = 0
        fnam, fext = os.path.splitext(fpath)
        if fext not in code_exts:
            return 0
        cchars = {".py": "#", ".f90": "!", ".F": "!c", "C": "\\"}
        for line in open(fpath, "r").readlines():
            line = line.strip()
            if not line.split() or line[0] in cchars.get(fext, "#"):
                continue
            nlines += 1
            continue
        return nlines

    all_dirs, all_files = [], []
    code_exts = [".py", ".pyf", "", ".F", ".C", ".f", ".f90"]
    all_exts = code_exts + [".inp", ".tex", ".pdf"]
    for dirnam, dirs, files in os.walk(cfg.ROOT):
        if ".git" in dirnam:
            continue
        all_dirs.extend([os.path.join(dirnam, d) for d in dirs])
        all_files.extend([os.path.join(dirnam, ftmp) for ftmp in files
                          if not os.path.islink(os.path.join(dirnam, ftmp))
                          and os.path.splitext(ftmp)[1] in all_exts])
        continue
    num_lines = sum([_num_code_lines(ftmp) for ftmp in all_files])
    num_dirs = len(all_dirs)
    num_files = len(all_files)
    num_infiles = len([x for x in all_files if x.endswith(".inp")])
    num_pyfiles = len([x for x in all_files
                       if x.endswith(".py") or x.endswith(".pyf")])
    pu.log_message(cfg.INTRO, pre="")
    pu.log_message("Summary of Project:", pre="")
    pu.log_message("\tNumber of files in project:         {0:d}"
                   .format(num_files), pre="")
    pu.log_message("\tNumber of directories in project:   {0:d}"
                   .format(num_dirs), pre="")
    pu.log_message("\tNumber of input files in project:   {0:d}"
                   .format(num_infiles), pre="")
    pu.log_message("\tNumber of python files in project:  {0:d}"
                   .format(num_pyfiles), pre="")
    pu.log_message("\tNumber of lines of code in project: {0:d}"
                   .format(num_lines), pre="")
    return


def _clean_file_exts(files, cleanall):
    """Remove Payette generated files

    Parameters
    ----------
    files : list
        list of files to clean
    cleanall : bool
        if True, remove output files as well

    Returns
    -------
    None

    """

    pu.log_message("Cleaning Payette output for {0}".format(", ".join(files)))
    payette_exts = [".log", ".math1", ".math2", ".props", ".echo", ".prf"]
    if cleanall:
        payette_exts.extend([".out"])

    # clean all the payette output and exit
    if not files:
        pu.log_warning("No base file name given to clean")
        return

    for fpath in files:
        fpath = os.path.realpath(os.path.expanduser(fpath))
        fdir = os.path.dirname(fpath)
        fnam, fext = os.path.splitext(fpath)
        faux = [os.path.join(fdir, x) for x in os.listdir(fdir) if
                os.path.splitext(x)[1] in payette_exts and
                os.path.splitext(x)[0] == os.path.basename(fnam)]
        for fff in faux:
            try:
                os.remove(fff)
            except OSError:
                pass
            continue
        continue
    pu.log_message("Output cleaned")
    return


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# if __name__ == "__main__":

#     ARGV = sys.argv[1:]

#     if "--profile" in ARGV:
#         PROFILE = True
#         ARGV.remove("--profile")
#         import profile
#     else:
#         PROFILE = False

#     if PROFILE:
#         CMD = "run_payette(ARGV)"
#         PROF = "payette.prof"
# #        profile.runctx(CMD, globals(), locals(), PROF)
#         profile.run(CMD)
#         PAYETTE = 0
#     else:
#         PAYETTE = run_payette(ARGV)

#     sys.exit(PAYETTE)
