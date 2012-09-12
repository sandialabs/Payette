#!/usr/bin/env python

"""
Top level interface to the Payette material model driver.

"""
import sys, os
import optparse
from optparse import OptionParser, BadOptionError, AmbiguousOptionError


FILE = os.path.realpath(__file__)
ROOTDIR = os.path.realpath(os.path.join(os.path.dirname(FILE), "../"))
SRCDIR = os.path.join(ROOTDIR, "Source")
EXENAM = "payette"
sys.path.insert(0, ROOTDIR)


# import Payette specific files
try:
    import Payette_config as pc
except ImportError:
    sys.exit("Payette must first be configured before execution")
import Source.runopts as ro
import Source.Payette_utils as pu
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

CONFIGURATION INFO
      Python interpreter: {0}

OPTIONS
      There are a set of options recognized by the payette script, which are
      listed here.  If an option is given which is not listed here, it is
      passed through to the different Payette modules.
""".format(pc.PC_PYINT)


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
                OptionParser._process_args(self,largs,rargs,values)
            except (BadOptionError,AmbiguousOptionError), e:
                largs.append(e.opt_str)


def warning(msg):
    sys.stdout.write("WARNING: {0}\n".format(msg))

def error(msg):
    sys.exit("ERROR: {0}\n".format(msg))


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
    payette_exts = [".log", ".math1", ".math2", ".props", ".echo", ".prf"]
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
        "--cchar",
        dest="cchar",
        action="store",
        default=None,
        help=("Additional comment characters for input file "
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
    parser.add_option(
        "-V",
        dest="VIZ",
        action="store_true",
        default=False,
        help="Display visualization window upon completion [default: %default]")
    parser.add_option(
        "-D",
        dest="AUG_DIR",
        action="store",
        default=None,
        help="Alternate directory to find index file [default: %default]")

    # ---------------------- Sandia National Labs specific material directories
    # The following option is only valid if the user configured with Lambda
    if pc.LAMBDA_MDLS:
        help_message = "Use Lambda models"
    else:
        help_message = "Payette must be configured for Lambda models to use -L"
    parser.add_option(
        "-L",
        dest="LAMBDA",
        action="store_true",
        default=False,
        help=help_message + " [default: %default]")
    # The following option is only valid if the user configured with Alegra
    if pc.ALEGRA_MDLS:
        help_message = "Run with the Alegra models"
    else:
        help_message = "Payette must be configured for Alegra models to use -A"
    parser.add_option(
        "-A",
        dest="ALEGRA",
        action="store_true",
        default=False,
        help=help_message + " [default: %default]")
    # ------------------ end Sandia National Labs specific material directories

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
        "-a", "--simdir",
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
        "--disp",
        dest="disp",
        action="store",
        default=ro.DISP,
        help="Return extra diagnositic information if > 0 [default: %default]")
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
    # ----------------------------------------- end command line option parsing

    if opts.debug:
        opts.verbosity = 4

    if opts.CLEANALL:
        payette_exts.extend([".out"])
        opts.CLEAN = True

    if opts.CLEAN:
        # ------------------------------------------------------ clean and exit
        cleaned = False
        # clean all the payette output and exit
        if not args:
            sys.exit(warning("No base file name given to clean"))
        for fpath in args:
            fpath = os.path.realpath(fpath)
            fdir = os.path.dirname(fpath)
            fnam, fext = os.path.splitext(fpath)
            if fnam not in [os.path.splitext(x)[0] for x in os.listdir(fdir)]:
                warning("no Payette output for {0} found in {1}"
                        .format(fpath, fdir))
                continue
            pu.log_message("cleaning output for {0}".format(fnam))
            for ext in payette_exts:
                try:
                    os.remove(fnam + ext)
                    cleaned = True
                except OSError:
                    pass
                continue
            continue
        if cleaned:
            pu.log_message("output cleaned")
        else:
            warning("output not cleaned")
        sys.exit(0)
        # -------------------------------------------------- end clean and exit

    if opts.MAN:
        # print the man page for this and other scripts
        pu.log_message(MAN_PAGE, pre="")
        sys.exit(parser.print_help())

    # pass command line arguments to global Payette variables
    ro.set_command_line_options(opts)

    # ----------------------------------------------- start: get the user input
    incompat = (opts.AUG_DIR, opts.LAMBDA, opts.ALEGRA, )
    if len([x for x in incompat if bool(x)]) > 1:
        parser.error("-D, -L, and -A must be specified independently")

    if opts.AUG_DIR is not None:
        if not os.path.isdir(opts.AUG_DIR):
            parser.error("{0} not found".format(opts.AUG_DIR))
        material_index = os.path.join(
            opts.AUG_DIR, os.path.basename(pc.PC_MTLS_FILE))
    elif opts.LAMBDA:
        material_index = os.path.join(
            pc.LAMBDA_MDLS[0], os.path.basename(pc.PC_MTLS_FILE))
    elif opts.ALEGRA:
        material_index = os.path.join(
            pc.ALEGRA_MDLS[0], os.path.basename(pc.PC_MTLS_FILE))
    else:
        material_index = pc.PC_MTLS_FILE

    if not os.path.isfile(material_index):
        pu.report_and_raise_error(
            "buildPayette must first be executed to create\n\t{0}"
            .format(material_index))
    ro.set_global_option("MTLDB", material_index, default=True)

    if not argv:
        if not pc.VIZ_COMPATIBLE:
            sys.exit("Visualization not supported by your Python distribution")
        # Launch the Gui and exit
        import Source.Viz_ModelSelector as vms
        pmms = vms.PayetteMaterialModelSelector(model_type="any")
        sys.exit(pmms.configure_traits())

    if opts.verbosity:
        pu.log_message(pc.PC_INTRO, pre="")

    # determine file type given, whether output files for viewing, input files
    # for running, or barf files for barf processing
    oexts = (".out", ".dat",)
    iexts = (".barf", ".inp", ".i",)
    rexts = (".prf", )
    oargs, iargs, uargs, bargs, rargs = [], [], [], [], []
    for arg in list(set(args)):
        farg = os.path.realpath(os.path.expanduser(arg))

        # check for file existence
        if not os.path.isfile(farg):
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
        pu.report_and_raise_error(
            "Files with unkown file extension[s] passed to Payette [{0}]"
            .format(os.path.splitext(x)[1] for x in uargs))
    if len([x for x in (oargs, iargs, uargs, bargs, rargs) if x]) > 1:
        pu.report_and_raise_error(
            "Payette can only process one file type request at a time")

    if oargs:
        # output files given, launch visualizer
        sys.exit(visualize_results(outfiles=oargs))

    if bargs:
        # user passed in a barf file
        if len(bargs) > 1:
            pu.report_and_raise_error(
                "{0:d} barf files given, but only 1 barf file "
                "can be processed at a time".format(len(bargs)))
        from Source.Payette_barf import PayetteBarf
        PayetteBarf(bargs[0])
        sys.exit(0)

    # We are now to the point where we will call run_payette, this could be
    # with either a restart file, or with the contents of input files.
    siminp = []
    if opts.inputstr:
        # user gave input directly
        siminp.extend(opts.inputstr.split("\n"))

    restart = False
    if rargs:
        if len(rargs) > 1:
            pu.report_and_raise_error(
                "{0:d} restart files given, but only 1 restart file "
                "can be processed at a time".format(len(rargs)))
        elif siminp:
            pu.report_and_raise_error(
                "Restart files cannot be run with additional input")
        siminp = [rargs[0]]
        restart = True

    elif iargs:
        # go through input files and load contents
        for iarg in iargs:
            siminp.extend(open(iarg, "r").readlines())
            continue
    # ----------------------------------------------------- end: get user input

    # make sure input file is given and exists
    if not siminp:
        parser.print_help()
        parser.error("No input given")

    # call the run_payette function
    siminfo = run_payette(siminp=siminp, restart=restart, timing=opts.timing,
                          cchar=opts.cchar, nproc=opts.nproc, disp=opts.disp,
                          verbosity=opts.verbosity)

    # visualize the results if requested
    if opts.VIZ:
        visualize_results(simulation_info=siminfo)

    retcode = 1 if any(x["retcode"] for x in siminfo) else 0
    return retcode

def visualize_results(simulation_info=None, outfiles=None):
    """visualize the results from a simulation

    Parameters
    ----------
    simulation_info : list
       list of return dictionaries

    outfiles : list
       list of output files to visualize
    """
    if not pc.VIZ_COMPATIBLE:
        pu.log_warning("Visualization not supported by your Python distribution")
        return

    from Viz_ModelPlot import create_Viz_ModelPlot
    import Source.Payette_sim_index as psi

    if outfiles is not None and simulation_info is not None:
        pu.log_warning("Cannot specify both outfiles and simulation_info")

    elif outfiles is not None:
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
                                        "output file": outfile,})
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

