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

from __future__ import print_function
import os
from os.path import isfile, isdir, join, realpath, dirname, expanduser, basename
import sys
import optparse
from distutils import sysconfig


__version__ = "1.0.dev"
__author__ = ("Tim Fuller, tjfulle@sandia.gov", "Scot Swan, mswan@sandia.gov")


def check_exists(itemnam, item):

    """ check if item exists on file system """

    if not isinstance(item, (list, tuple)):
        item = [item]

    errors = 0
    for tmp in item:
        if not isdir(tmp) and not isfile(tmp):
            errors += 1
            logerr("{0} not found [name: {1}]".format(tmp, itemnam))

    return errors


def begmes(msg, pre="", end="  "):

    """ begin message """

    print("{0}{1}...".format(pre, msg), end=end)
    return


def endmes(msg, pre="", end="\n"):

    """ end message """

    print("{0}{1}".format(pre, msg), end=end)
    return


def loginf(msg, pre="", end="\n"):

    """ log info """

    print("{0}INFO: {1}".format(pre, msg), end=end)
    return


def logmes(msg, pre="", end="\n"):

    """ log message """

    print("{0}{1}".format(pre, msg), end=end)
    return


def logwrn(msg, pre="", end="\n"):

    """ log warning """

    print("{0}WARNING: {1}".format(pre, msg), end=end)
    return


def logerr(msg, pre="", end="\n"):

    """ log error """

    print("{0}ERROR: {1}".format(pre, msg), end=end)
    return


def dictfrmt(key, val):

    """ format dictionary for pretty printing """

    if isinstance(val, str):
        return '{0} = "{1}"'.format(key, val)
    return '{0} = {1}'.format(key, val)


def get_exe_path(exe):

    """ return the absolute path to the executable exe """

    if isfile(exe):
        return exe

    try:
        path = os.getenv("PATH").split(os.pathsep)
    except AttributeError:
        path = []

    for dirname in path:
        if isfile(join(dirname, exe)):
            return join(dirname, exe)

    sys.exit("ERROR: executable {0} not found".format(exe))


def write_f2py(pyint, destdir):

    """
    write out f2py. we write out our own to ensure that we use the right python
    interpreter. I just copied this verbatim from my installation of f2py,
    replacing the interpreter on the shebang line with PC_PYINT

    HOWEVER, after writing this, I remembered that we never use f2py from the
    command line, but import it directly from numpy, so this is unnecessary...
    """

    f2py_file = """#!{0}
# See http://cens.ioc.ee/projects/f2py2e/
import os, sys
for mode in ["g3-numpy", "2e-numeric", "2e-numarray", "2e-numpy"]:
    try:
        i=sys.argv.index("--"+mode)
        del sys.argv[i]
        break
    except ValueError: pass
os.environ["NO_SCIPY_IMPORT"]="f2py"
if mode=="g3-numpy":
    sys.stderr.write("G3 f2py support is not implemented, yet.\\n")
    sys.exit(1)
elif mode=="2e-numeric":
    from f2py2e import main
elif mode=="2e-numarray":
    sys.argv.append("-DNUMARRAY")
    from f2py2e import main
elif mode=="2e-numpy":
    from numpy.f2py import main
else:
    sys.stderr.write("Unknown mode: " + repr(mode) + "\\n")
    sys.exit(1)
main()
""".format(pyint)
    f2py = join(destdir, "f2py")
    with open(f2py, "w") as fnew:
        for line in f2py_file:
            fnew.write(line)
    os.chmod(f2py, 0o750)
    return


ERRORS = 0


# --- compatibility checks
(MAJOR, MINOR, MICRO, RELEASELEVEL, SERIAL) = sys.version_info
if (MAJOR != 3 and MAJOR != 2) or (MAJOR == 2 and MINOR < 6):
    raise SystemExit("Payette requires Python >= 2.6\n")

# --- numpy check
try:
    import numpy
    PC_NUMPY_VER = numpy.__version__
except ImportError:
    logerr("numpy not importable")
    ERRORS += 1

# --- scipy check
try:
    import scipy
    PC_SCIPY_VER = scipy.__version__
except ImportError:
    logerr("scipy not importable")
    ERRORS += 1

if ERRORS:
    sys.exit("configure.py: ERROR: fix previously trapped errors")


# --- intro message
PC_INTRO = """
        PPPPPPPPP      A  Y     Y  EEEEE  TTTTTTTTTTT  TTTTTTTTTTT  EEEEEE
       P        P    A A   Y   Y  E            T            T      E
      P        P   A   A    Y Y  E            T            T      E
     PPPPPPPPP   A  A  A     Y  EEEE         T            T      EEEE
    P          A       A    Y  E            T            T      E
   P         A         A   Y  E            T            T      E
  P        A           A  Y  EEEEEEEE     T            T      EEEEEEEE

                              An Object Oriented Material Model Driver
{0}
""".format(" "*(62 - len(__version__)) + "version " + __version__)

# --- spacing used for logs to console
SPACE = "      "

# --- All environmental variables used by Payette are listed here
ENV_BENCHDIR = "PAYETTE_BENCHDIR"
ENV_MTLDIR = "PAYETTE_MTLDIR"

# --- base level directories
THIS_FILE = realpath(__file__)
PC_ROOT = dirname(THIS_FILE)
PC_AUX = join(PC_ROOT, "Aux")
PC_DOCS = join(PC_ROOT, "Documents")
PC_SOURCE = join(PC_ROOT, "Source")
PC_TESTS = [join(PC_ROOT, "Benchmarks")]
ENV_BENCHDIR = "PAYETTE_BENCHDIR"
USER_TESTS = os.getenv(ENV_BENCHDIR, "")
PC_TESTS.extend([x for x in USER_TESTS.split(os.pathsep) if x])
PC_TOOLS = join(PC_ROOT, "Toolset")
PC_FOUND_TESTS = join(PC_TOOLS, "__found_tests__.py")

# modify sys.path
if PC_ROOT not in sys.path:
    sys.path.insert(0, PC_ROOT)

ERRORS += check_exists("PC_ROOT", PC_ROOT)
ERRORS += check_exists("PC_AUX", PC_AUX)
check_exists("PC_DOCS", PC_DOCS)
ERRORS += check_exists("PC_SOURCE", PC_SOURCE)
ERRORS += check_exists("PC_TESTS", PC_TESTS)
ERRORS += check_exists("PC_TOOLS", PC_TOOLS)

# --- python interpreter info
PC_PYINT = realpath(sys.executable)
SAGE = True if "sage" in PC_PYINT.lower() else False
PC_PYVER = "python" if not SAGE else "sage -python"
PC_PYVER = "{0} {1}.{2}.{3}".format(PC_PYVER,MAJOR,MINOR,MICRO)

# --- Payette executable files
PC_CLEANPAYETTE = (join(PC_TOOLS, "cleanPayette"), THIS_FILE)

PC_EXTRACT = join(PC_SOURCE, "Payette_extract.py")
PC_EXTRACTPAYETTE = (join(PC_TOOLS, "extractPayette"), PC_EXTRACT)

PC_RUN = join(PC_SOURCE, "Payette_run.py")
PC_RUNPAYETTE = (join(PC_TOOLS, "runPayette"), PC_RUN)

PC_BUILD = join(PC_SOURCE, "Payette_build.py")
PC_BUILDPAYETTE = (join(PC_TOOLS, "buildPayette"), PC_BUILD)

PC_RUNTEST = join(PC_SOURCE, "Payette_runtest.py")
PC_TESTPAYETTE = (join(PC_TOOLS, "testPayette"), PC_RUNTEST)

PC_VIZ = join(PC_SOURCE, "Viz_ModelSelector.py")
PC_VIZPAYETTE = (join(PC_TOOLS, "vizPayette"), PC_VIZ)

PC_F2PY = (join(PC_TOOLS,"f2py"), None)
PC_BUILT_EXES = {"runPayette": PC_RUNPAYETTE,
                 "testPayette": PC_TESTPAYETTE,
                 "vizPayette": PC_VIZPAYETTE,
                 "buildPayette": PC_BUILDPAYETTE,
                 "cleanPayette": PC_CLEANPAYETTE,
                 "extractPayette": PC_EXTRACTPAYETTE,
                 "f2py": PC_F2PY}
PC_EXES = {}
for exe_nam, exe_info in PC_BUILT_EXES.items():
    exe_path, py_path = exe_info
    PC_EXES[exe_nam] = exe_path
    continue

# --- configuration files
PC_CONFIG_FILE = join(PC_ROOT, "Payette_config.py")

# --- subdirectories of PC_AUX
PC_INPUTS = join(PC_ROOT, "Aux/Inputs")
ERRORS += check_exists("PC_INPUTS", PC_INPUTS)

# --- subdirectories of PC_SOURCE

# PC_MTLDIRS is the directory where we search for Payette material models
PC_MTLS = join(PC_SOURCE, "Materials")
PC_MTLDIRS = [join(PC_MTLS, "Models")]
USER_MTLS = os.getenv(ENV_MTLDIR, "")
PC_MTLDIRS.extend([x for x in USER_MTLS.split(os.pathsep) if x])
for mtl_d in [x for x in PC_MTLDIRS]:
    if isdir(mtl_d):
        for dirnam, dirs, files in os.walk(mtl_d):
            if (".svn" not in dirnam and ".git" not in dirnam and
                dirnam not in PC_MTLS and
                any(y.endswith(".py") for y in files if y != "__init__.py")):
                PC_MTLDIRS.append(dirnam)
            continue
    continue
PC_FORTRAN = join(PC_SOURCE, "Fortran")
PC_MIG_UTILS = join(PC_FORTRAN, "migutils.F")
ERRORS += check_exists("PC_MTLS", PC_MTLS)
ERRORS += check_exists("PC_MIG_UTILS", PC_MIG_UTILS)

# --- Subdirectories of PC_MTLS
PC_MTLS_LIBRARY = join(PC_MTLS, "Library")
PC_MTLS_INCLUDES = join(PC_MTLS, "Includes")
PC_MTLS_FILE = join(PC_SOURCE, "installed_materials.pkl")
ERRORS += check_exists("PC_MTLS_LIBRARY", PC_MTLS_LIBRARY)
ERRORS += check_exists("PC_MTLS_INCLUDES", PC_MTLS_INCLUDES)

# --- extension module file extension
PC_EXT_MOD_FEXT = sysconfig.get_config_var("SO")

# --- get platform
PC_OSTYPE = sys.platform

# Store all of the above information for writing to the PC_CONFIG_FILE. We
# waited to write it til now so that we would only write it if everything was
# configured correctly.
PAYETTE_CONFIG = {}
PAYETTE_CONFIG["PC_PYINT"] = PC_PYINT
PAYETTE_CONFIG["PC_PYVER"] = PC_PYVER
PAYETTE_CONFIG["PC_ROOT"] = PC_ROOT
PAYETTE_CONFIG["PC_AUX"] = PC_AUX
PAYETTE_CONFIG["PC_DOCS"] = PC_DOCS
PAYETTE_CONFIG["PC_SOURCE"] = PC_SOURCE
PAYETTE_CONFIG["PC_TESTS"] = PC_TESTS
PAYETTE_CONFIG["PC_TOOLS"] = PC_TOOLS
PAYETTE_CONFIG["PC_FOUND_TESTS"] = PC_FOUND_TESTS
PAYETTE_CONFIG["PC_MTLS"] = PC_MTLS
PAYETTE_CONFIG["PC_MTLDIRS"] = PC_MTLDIRS
PAYETTE_CONFIG["PC_FORTRAN"] = PC_FORTRAN
PAYETTE_CONFIG["PC_MIG_UTILS"] = PC_MIG_UTILS
PAYETTE_CONFIG["PC_MTLS_LIBRARY"] = PC_MTLS_LIBRARY
PAYETTE_CONFIG["PC_MTLS_INCLUDES"] = PC_MTLS_INCLUDES
PAYETTE_CONFIG["PC_MTLS_FILE"] = PC_MTLS_FILE
PAYETTE_CONFIG["PC_INPUTS"] = PC_INPUTS
PAYETTE_CONFIG["PC_EXT_MOD_FEXT"] = PC_EXT_MOD_FEXT
PAYETTE_CONFIG["PC_OSTYPE"] = PC_OSTYPE
PAYETTE_CONFIG["PC_RUNTEST"] = PC_RUNTEST
PAYETTE_CONFIG["PC_RUN"] = PC_RUN
PAYETTE_CONFIG["PC_BUILD"] = PC_BUILD
PAYETTE_CONFIG["PC_EXTRACT"] = PC_EXTRACT
PAYETTE_CONFIG["PC_EXES"] = PC_EXES
PAYETTE_CONFIG["PC_CONFIG_FILE"] = PC_CONFIG_FILE
PAYETTE_CONFIG["PC_F2PY"] = PC_F2PY
PAYETTE_CONFIG["PC_RUNPAYETTE"] = PC_RUNPAYETTE
PAYETTE_CONFIG["PC_TESTPAYETTE"] = PC_TESTPAYETTE
PAYETTE_CONFIG["PC_BUILDPAYETTE"] = PC_BUILDPAYETTE
PAYETTE_CONFIG["PC_CLEANPAYETTE"] = PC_CLEANPAYETTE
PAYETTE_CONFIG["PC_EXTRACTPAYETTE"] = PC_EXTRACTPAYETTE
PAYETTE_CONFIG["PC_BUILT_EXES"] = PC_BUILT_EXES
PAYETTE_CONFIG["PC_NUMPY_VER"] = PC_NUMPY_VER
PAYETTE_CONFIG["PC_SCIPY_VER"] = PC_SCIPY_VER

# --- set up the environment
ENV = {}
ENVS = ["MPLCONFIGDIR", "PYTHONPATH", "ECLDIR", "GPDOCDIR", "RHOME",
        "GP_DATA_DIR", "PKG_CONFIG_PATH", "PYTHONHOME", "LD_LIBRARY_PATH",
        "LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH", "SINGULAR_EXECUTABLE",
        "SINGULARPATH"]

# Fortran compiler for extension libraries
FORT = "gfortran"

# --- if running with sage, configure the sage environment
if SAGE:
    # get the sage environment to save
    for skey, sval in os.environ.items():
        if "sage" in sval.lower() or "sage" in skey.lower():
            ENVS.append(skey)
            continue
        sage_local = os.environ.get("SAGE_LOCAL")
        if sage_local is not None:
            FORT = (join(sage_local, "gfortran") if
                    isfile(join(sage_local, "gfortran")) else
                    FORT)

if ERRORS:
    sys.exit("configure.py: ERROR: fix previously trapped errors")


def configure_payette(argv):

    """ create and write configuration file """

    # *************************************************************************
    # -- command line option parsing
    usage = ("usage: python %prog [options]\nmust be executed from "
             "{0}".format(PC_ROOT))
    parser = optparse.OptionParser(usage=usage, version="%prog 1.0")
    parser.add_option(
        "-o",
        dest="OPTIONS",
        action="append",
        default=[],
        help="Options to build (accumulated) [default: %default]")
    parser.add_option(
        "--fcompiler",
        dest="FCOMPILER",
        action="store",
        type="choice",
        choices=(None, "gnu95"),
        default=None,
        help="Specify Fortran compiler type by vendor [default: %default]")
    parser.add_option(
        "--f77exec",
        dest="F77EXEC",
        action="store",
        default=FORT,
        help="Specify the path F77 to compiler [default: %default]")
    parser.add_option(
        "--f2py-debug",
        dest="F2PYDBG",
        action="store_true",
        default=False,
        help="Compile (f2py) with debugging information [default: %default]")
    parser.add_option(
        "--f90exec",
        dest="F90EXEC",
        action="store",
        default=FORT,
        help="Specify the path F90 to compiler [default: %default]")
    parser.add_option(
        "--no-callback",
        dest="NOCALLBACK",
        action="store_true",
        default=False,
        help="Compile with f2py callbacks functions [default: %default]")
    parser.add_option(
        "--benchdirs",
        dest="BENCHDIRS",
        action="append",
        default=[],
        help="Additional directories to scan for benchmarks [default: %default]")
    parser.add_option(
        "--mtldirs",
        dest="MTLDIRS",
        action="append",
        default=[],
        help="Additional directories to scan for materials [default: %default]")

    opts = parser.parse_args(argv)[0]

    try:
        os.remove(PC_CONFIG_FILE)
    except OSError:
        pass

    try:
        os.remove(PC_CONFIG_FILE + "c")
    except OSError:
        pass

    errors = 0

    # clean up first
    clean_payette()

    # Report on environmental variables
    loginf("checking for Payette-related environmental variables")
    if USER_MTLS == "":
        logmes("${0} not set".format(ENV_MTLDIR), pre=SPACE)
    else:
        logmes("${0} set".format(ENV_MTLDIR), pre=SPACE)

    if USER_TESTS == "":
        logmes("${0} not set".format(ENV_BENCHDIR), pre=SPACE)
    else:
        logmes("${0} set".format(ENV_BENCHDIR), pre=SPACE)

    # configure Payette
    loginf("configuring Payette environment")

    # f2py call back
    if MAJOR == 3 or SAGE:
        opts.NOCALLBACK = True
    PAYETTE_CONFIG["PC_F2PY_CALLBACK"] = not opts.NOCALLBACK

    # f2py fortran compiler options
    if opts.FCOMPILER:
        PAYETTE_CONFIG["PC_FCOMPILER"] = opts.FCOMPILER
        PAYETTE_CONFIG["PC_F77EXEC"] = None
        PAYETTE_CONFIG["PC_F90EXEC"] = None
    else:
        PAYETTE_CONFIG["PC_FCOMPILER"] = None
        PAYETTE_CONFIG["PC_F77EXEC"] = get_exe_path(opts.F77EXEC)
        PAYETTE_CONFIG["PC_F90EXEC"] = get_exe_path(opts.F90EXEC)
    PAYETTE_CONFIG["PC_F2PYDBG"] = opts.F2PYDBG

    # add to benchmark dir
    for dirnam in opts.BENCHDIRS:
        dirnam = expanduser(dirnam)
        if isdir(dirnam):
            is_test_dir = False
            for item in os.walk(dirnam):
                if "__test_dir__.py" in item[-1]:
                    is_test_dir = True
                    PC_TESTS.append(dirnam)
                    break
                continue
            if not is_test_dir:
                errors += 1
                logerr("__test_dir__.py not found in {0}".format(dirnam))
        else:
            errors += 1
            logerr("benchmark directory {0} not found".format(dirnam))
        continue

    # add to materials dir
    for dirnam in opts.MTLDIRS:
        dirnam = expanduser(dirnam)
        if isdir(dirnam):
            PC_MTLDIRS.append(dirnam)
        else:
            errors += 1
            logerr("material directory {0} not found".format(dirnam))
        continue
    sys.path.extend(PC_MTLDIRS)

    if errors:
        sys.exit("ERROR: stopping due to previous errors")

    # get current environment
    for item in ENVS:
        if item in os.environ:
            ENV[item] = os.environ[item]
        continue

    # make sure PC_ROOT is first on PYTHONPATH
    pypath = os.pathsep.join([PC_ROOT, PC_TOOLS] + PC_MTLDIRS)
    if "PYTHONPATH" in ENV:
        pypath += (os.pathsep +
                   os.pathsep.join([x for x in ENV["PYTHONPATH"].split(os.pathsep)
                                    if x not in pypath.split(os.pathsep)]))
    ENV["PYTHONPATH"] = pypath

    # write the the configuration file
    begmes("writing Payette_config.py", pre=SPACE)
    with open(PC_CONFIG_FILE, "w") as fnew:
        fnew.write(PREAMBLE + "\n" +
                   'PC_INTRO = """{0}"""\n'.format(PC_INTRO))
        for key, val in PAYETTE_CONFIG.items():
            fnew.write(dictfrmt(key, val) + "\n")
            continue
        fnew.write("if PC_ROOT not in sys.path: "
                   "sys.path.insert(0, PC_ROOT)\n")
        fnew.write("sys.path.extend(PC_MTLDIRS)\n")
        for key, val in ENV.items():
            fnew.write('os.environ["{0}"] = "{1}"\n'.format(key, val))
            continue

    endmes("Payette_config.py written")

    # try importing the file we just wrote to test if it is importable
    try:
        import Payette_config
        if not Payette_config.PC_BUILT_EXES:
            pass
    except ImportError:
        print("ERROR: Payette_config.py not importable")
        raise

    loginf("Payette environment configured")

    return ERRORS


def create_payette_exececutables():

    """ create the Payette executables """

    loginf("writing executable scripts")

    # message for executables that require Payette be built
    exit_msg = """if [ ! -f {0} ]; then
   echo "buildPayette must be executed to create {0}"
   exit
fi
""".format(PC_MTLS_FILE)

    for name, files in PC_BUILT_EXES.items():

        exe_path, py_file = files

        # remove the executable first
        try:
            os.remove(exe_path)
        except OSError:
            pass

        begmes("writing {0}".format(name), pre=SPACE)

        if name == "f2py":
            write_f2py(PC_PYINT, PC_TOOLS)
            endmes("{0} script written".format(name))
            continue

        with open(exe_path, "w") as fnew:
            fnew.write("#!/bin/sh -f\n")
            for key, val in ENV.items():
                fnew.write("export {0}={1}\n".format(key, val))
                continue

            if name == "cleanPayette":
                fnew.write("{0} {1} {2} $* 2>&1\n"
                           .format(PC_PYINT, py_file, "clean"))

            elif name in ("buildPayette", "extractPayette",):
                fnew.write("{0} {1} $* 2>&1\n"
                           .format(PC_PYINT, py_file))

            else:
                fnew.write(exit_msg)
                fnew.write("{0} {1} $* 2>&1\n"
                           .format(PC_PYINT, py_file))


        os.chmod(exe_path, 0o750)
        endmes("{0} script written".format(name))
        continue

    path = join(PC_TOOLS, "pconfigure")
    begmes("writing pconfigure", pre=SPACE)
    with open(path, "w") as fnew:
        fnew.write("#!/bin/sh -f\n")
        fnew.write("cd {0}\n".format(PC_ROOT))
        fnew.write("{0} {1}\n".format(PC_PYINT, " ".join(sys.argv)))
    os.chmod(path, 0o750)
    endmes("pconfigure written")

    loginf("executable scripts written\n")

    return ERRORS


PREAMBLE = \
"""# *************************************************************************** #
#                                                                             #
# This file was generated automatically by the Payette. It contains important #
# global Payette parameters that are configured at build time.                #
#                                                                             #
# This file is intended to be imported by files in Payette like               #
# "from Payette_config import XXXXX"                                          #
# where XXXXXX is a specific variable.                                        #
#                                                                             #
# DO NOT EDIT THIS FILE. This entire file is regenerated automatically each   #
# time configure.py is run. Any changes you make to this file will be         #
# overwritten.                                                                #
#                                                                             #
# If changes are needed, please contact the Payette developers so that changes#
# can be made to the build scripts.                                           #
#                                                                             #
# *************************************************************************** #
import sys
import os
"""


def clean_payette():

    """ clean Payette of any automatically generated files """

    from fnmatch import fnmatch

    soext = sysconfig.get_config_var("SO")

    pats_to_remove = ["*.pyc", "*.pyo", "Payette_config.py",
                      "*{0}".format(soext),
                      "installed_materials.pkl", "__found_tests__.py",
                      "*.log", "*.echo", "*.prf", "*.diff", "*.xout", "*.out",
                      "*.math1", "*.math2", "*.props", "*.vtable", "*.dtable"]
    pats_to_remove.extend(PC_BUILT_EXES.keys())

    for item in os.walk(PC_ROOT):
        dirnam, files = item[0], item[2]

        if ".svn" in dirnam:
            continue

        for fnam in files:
            if any(fnmatch(fnam, pat) for pat in pats_to_remove):
                os.remove(join(dirnam, fnam))

            continue

        continue

    return


if __name__ == "__main__":

    if "clean" in sys.argv:
        loginf("cleaning Payette")
        clean_payette()
        loginf("Payette cleaned")
        sys.exit(0)

    elif any("vers" in x for x in sys.argv):
        sys.exit("Payette, version " + __version__)

    if sys.argv[0] != basename(__file__):
        sys.exit("configure.py must be executed from {0}".format(PC_ROOT))

    # introduce yourself
    logmes(PC_INTRO)

    # now configure
    CONFIGURE = configure_payette(sys.argv[1:])
    if CONFIGURE > 0:
        sys.exit("ERROR: configure failed\n")

    # and write the executables
    WRITE_EXE = create_payette_exececutables()
    if WRITE_EXE > 0:
        sys.exit("ERROR: failed to write executables\n")

    # all done
    loginf("configuration complete")

