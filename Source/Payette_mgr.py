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
import re
import glob
from textwrap import fill as textfill

FILE = os.path.realpath(__file__)
ROOTDIR = os.path.realpath(os.path.join(os.path.dirname(FILE), "../"))
SRCDIR = os.path.join(ROOTDIR, "Source")
EXENAM = "payette"
CWD = os.getcwd()

# import Payette specific files
try:
    import Source.__config__ as cfg
except ImportError:
    sys.exit("ERROR: Payette must first be configured before execution")
import Source.__runopts__ as ro
import Source.Payette_utils as pu
import Source.Payette_model_index as pmi
from Source.Payette_run import run_payette
from Source.Payette_utils import PassThroughOptionParser

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
        "-B",
        action="store_true",
        default=False,
        help="Build Payette [default: %default]")
    parser.add_option(
        "-T",
        action="store_true",
        default=False,
        help="Run the Payette tests [default: %default]")
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
        "-D", "--simdir",
        dest="simdir",
        action="store",
        default=ro.SIMDIR,
        help="Directory to run simulation [default: {0}]".format(CWD))
    parser.add_option(
        "--use-table",
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
        # User specified alternative material db file
        if os.path.isfile(opts.AUXMTL):
            # absolute path to material db file specified
            material_db = opts.AUXMTL
        elif os.path.isfile(os.path.join(opts.AUXMTL, cfg.AUXDB)):
            # directory specified with a 'auxilary.db' material db file
            material_db = os.path.join(opts.AUXMTL, cfg.AUXDB)
        elif os.path.isfile(os.path.join(cfg.DOTPAYETTE, opts.AUXMTL)):
            # material db file name specified that resides in $DOTPAYETTE
            material_db = os.path.join(cfg.DOTPAYETTE, opts.AUXMTL)
        elif os.path.isfile(os.path.join(cfg.DOTPAYETTE, opts.AUXMTL + ".db")):
            # material db file name specified (no ext) that resides in $DOTPAYETTE
            material_db = os.path.join(cfg.DOTPAYETTE, opts.AUXMTL + ".db")
        else:
            auxmtl = opts.AUXMTL
            if os.path.isdir(auxmtl):
                auxmtl = os.path.join(auxmtl, cfg.AUXDB)
            if not auxmtl.endswith(".db"):
                auxmtl = auxmtl + ".db"
            sys.exit("ERROR: Material database {0} not found".format(auxmtl))
        sys.path.insert(0, os.path.dirname(material_db))
    else:
        material_db = cfg.MTLDB

    if not os.path.isfile(material_db):
        sys.exit("ERROR: buildPayette must first be executed to create\n\t{0}"
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
            sys.exit("ERROR: Visualization not supported by your "
                     "Python distribution")
        import Source.Viz_Control as vc
        window = vc.ControlWindow()
        sys.exit(window.configure_traits())

    if not argv:
        if not cfg.VIZ_COMPATIBLE:
            sys.exit("ERROR: Visualization not supported by your "
                     "Python distribution")
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
        sys.exit("ERROR: Stopping due to previous errors")

    # check for incompatibilities
    if uargs:
        for uarg in uargs:
            if os.path.isdir(uarg) and "index.pkl" in os.listdir(uarg):
                oargs.append(os.path.join(uarg, "index.pkl"))
                continue
            sys.exit("ERROR: Unkown file extension[s] passed to Payette [{0}]"
                     .format(os.path.splitext(x)[1] for x in uargs))

    if len([x for x in (oargs, iargs, bargs, rargs) if x]) > 1:
        sys.exit("ERROR: Payette can only process one file type at a time")

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
            sys.exit("ERROR: Only 1 restart file can be processed at a time")
        elif siminp is not None:
            sys.exit("ERROR: Restart files cannot be run with additional input")
        restart = rargs[0]

    elif bargs:
        # user passed in a barf file
        if len(bargs) > 1:
            sys.exit("ERROR: Only 1 barf file can be processed at a time")
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

    retcode = max([x["retcode"] for x in siminfo])
    return retcode


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
        index = psi.SimulationIndex(CWD)
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
    exts = [".log", ".math1", ".math2", ".props", ".echo", ".prf", ".pyc"]

    if cleanall:
        exts.extend([".out", ".diff"])

    # clean all the payette output and exit
    if not files:
        files = [x for x in os.listdir(CWD) if x.endswith(".inp")]

    for fpath in files:
        fnam, fext = os.path.splitext(os.path.realpath(os.path.expanduser(fpath)))
        pu.log_message("Cleaning {0}".format(os.path.basename(fnam)))
        for f in _glob(fnam, *exts):
            try:
                os.remove(f)
            except OSError:
                pass
            continue
        continue
    pu.log_message("Output cleaned")
    return


def _glob(path, *exts):
    path = os.path.join(path, "*") if os.path.isdir(path) else path + "*"
    return [f for files in [glob.glob(path + ext) for ext in exts] for f in files]


def build(argv):
    from Payette_build import build_payette
    built = build_payette(argv)
    warn, error = 0, 0
    if built == 0:
        sys.stderr.write("\nINFO: buildPayette succeeded\n")

    elif built < 0:
        warn += 1
        sys.stderr.write("\nWARNING: buildPayette failed to build one or "
                         "more material libraries\n")

    elif built > 0:
        error += 1
        sys.stderr.write("\nERROR: buildPayette failed\n")

    return built


def run_test(argv):
    import Source.Payette_runtest as Pr
    return Pr.main(argv)


def _repl(s, l, L):
    repl = re.sub(s, "", l).strip()
    repl = "" if repl == "-" else repl
    L = L.replace(l, repl).strip()
    return L


def _pre(argv):
    """Check if user requests to build before execution, or if the user
    requested to run tests

    """
    run = True

    # join argv for regex processing
    jargv = " ".join(argv)

    # The regex used to determine uses a negative lookbehind (?<!...) to
    # determine if -B or -T appears in argv, but not --...B... or --...T...
    # (double --). If -B, we build before execution. If -T, the tests are run
    pat = r"(?<!-)-\w*{0}\w*\s*?"
    B = re.search(pat.format("B"), jargv)
    if B:
        jargv = _repl("B", B.group(), jargv)
        if build(jargv.split()) > 0:
            sys.exit("Payette failed to build")

        # remove from argv any arguments that were meant only for the build -
        # or at least try too... The regexs were copied from 'buildPayette -h'
        # because there is no way of dynamically determining which options are
        # unique to buildPayette
        R = r"(?<![-\w])(-[mAd]\s*[\{0}a-z0-9_\-]+\s*)".format(os.sep)
        for group in re.findall(R, jargv):
            jargv = re.sub(group, "", jargv)
        R = r"--[(kmm)(dsf)(lpc)]+|-w"
        for group in re.findall(R, jargv):
            jargv = re.sub(group, "", jargv)
        jargv = jargv.strip()
        if not jargv.split():
            # assume that if the user did not send additional arguments to
            # payette that they only wanted to build
            run = False

    T = re.search(pat.format("T"), jargv)
    if T:
        jargv = _repl("T", T.group(), jargv)
        run_test(jargv.split())
        # Assume that if the user wanted to run the tests, no further work
        # should be done
        run = False

    return run, jargv.strip().split()


if __name__ == "__main__":
    run, argv = _pre(sys.argv[1:])
    if run:
        sys.exit(main(argv))
    sys.exit(0)


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
