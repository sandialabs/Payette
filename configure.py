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

"""
NAME
   configure.py

PURPOSE
   Configure Payette and write Payette_config.py
"""

from __future__ import print_function
import os
import sys
import optparse
from distutils import sysconfig

ERRORS = 0

# --- compatibility checks
(MAJOR, MINOR, MICRO, RELEASELEVEL, SERIAL) = sys.version_info
if (MAJOR != 3 and MAJOR != 2) or (MAJOR == 2 and MINOR < 6):
    raise SystemExit("Payette requires Python >= 2.6\n")

# --- numpy check
try:
    import numpy
    PAYETTE_NUMPY_VERSION = numpy.__version__
except ImportError:
    logerr("numpy not importable")
    ERRORS += 1

# --- scipy check
try:
    import scipy
    PAYETTE_SCIPY_VERSION = scipy.__version__
except ImportError:
    logerr("scipy not importable")
    ERRORS += 1

if ERRORS:
    sys.exit("configure.py: ERROR: fix previously trapped errors")


def check_exists(itemnam, item):

    """ check if item exists on file system """

    if not item:
        logerr("{0} not found".format(itemnam))
        return 1
    elif not os.path.isdir(item) and not os.path.isfile(item):
        logerr("{0} not found".format(item))
        return 1

    return 0


def write_f2py(pyint, destdir):

    """
    write out f2py. we write out our own to ensure that we use the right python
    interpreter. I just copied this verbatim from my installation of f2py,
    replacing the interpreter on the shebang line with PAYETTE_PYINT

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
    f2py = os.path.join(destdir, "f2py")
    with open(f2py, "w") as fnew:
        for line in f2py_file:
            fnew.write(line)
    os.chmod(f2py, 0o750)
    return

# --- intro message
PAYETTE_INTRO = """
        PPPPPPPPP      A  Y     Y  EEEEE  TTTTTTTTTTT  TTTTTTTTTTT  EEEEEE
       P        P    A A   Y   Y  E            T            T      E
      P        P   A   A    Y Y  E            T            T      E
     PPPPPPPPP   A  A  A     Y  EEEE         T            T      EEEE
    P          A       A    Y  E            T            T      E
   P         A         A   Y  E            T            T      E
  P        A           A  Y  EEEEEEEE     T            T      EEEEEEEE

                              An Object Oriented Material Model Driver
"""

# --- spacing used for logs to console
SPACE = "      "

# --- base level directories
THIS_FILE = os.path.realpath(__file__)
PAYETTE_ROOT = os.path.dirname(THIS_FILE)
PAYETTE_AUX = os.path.join(PAYETTE_ROOT, "Aux")
PAYETTE_DOCUMENTS = os.path.join(PAYETTE_ROOT, "Documents")
PAYETTE_SOURCE = os.path.join(PAYETTE_ROOT, "Source")
PAYETTE_TESTS = os.path.join(PAYETTE_ROOT, "Tests")
PAYETTE_TOOLSET = os.path.join(PAYETTE_ROOT, "Toolset")
# modify sys.path
if PAYETTE_ROOT not in sys.path:
    sys.path.insert(0, PAYETTE_ROOT)
ERRORS += check_exists("PAYETTE_ROOT", PAYETTE_ROOT)
ERRORS += check_exists("PAYETTE_AUX", PAYETTE_AUX)
ERRORS += check_exists("PAYETTE_DOCUMENTS", PAYETTE_DOCUMENTS)
ERRORS += check_exists("PAYETTE_SOURCE", PAYETTE_SOURCE)
ERRORS += check_exists("PAYETTE_TESTS", PAYETTE_TESTS)
ERRORS += check_exists("PAYETTE_TOOLSET", PAYETTE_TOOLSET)

# --- python interpreter info
PAYETTE_PYINT = os.path.realpath(sys.executable)

# --- Payette executable files
PAYETTE_RUNTEST = os.path.join(PAYETTE_SOURCE, "Payette_runtest.py")
PAYETTE_RUN = os.path.join(PAYETTE_SOURCE, "payette_run.py")
PAYETTE_BUILD = os.path.join(PAYETTE_SOURCE, "payette_build.py")
RUNPAYETTE = os.path.join(PAYETTE_TOOLSET, "runPayette")
BUILDPAYETTE = os.path.join(PAYETTE_TOOLSET, "buildPayette")
CLEANPAYETTE = os.path.join(PAYETTE_TOOLSET, "cleanPayette")
EXTRACTPAYETTE = os.path.join(PAYETTE_TOOLSET, "extractPayette.py")
TESTPAYETTE = os.path.join(PAYETTE_TOOLSET, "testPayette")
PAYETTE_F2PY = os.path.join(PAYETTE_TOOLSET,"f2py")
PAYETTE_BUILT_EXECUTABLES = {"runPayette": RUNPAYETTE,
                             "testPayette": TESTPAYETTE,
                             "buildPayette": BUILDPAYETTE,
                             "cleanPayette": CLEANPAYETTE,
                             "f2py": PAYETTE_F2PY}
PAYETTE_EXECUTABLES = {"extractPayette.py": EXTRACTPAYETTE}
ERRORS += check_exists("extractPayette", EXTRACTPAYETTE)

for exe_nam, exe_path in PAYETTE_BUILT_EXECUTABLES.items():
    PAYETTE_EXECUTABLES[exe_nam] = exe_path
    continue

# --- auxilary Payette environment variables
PAYETTE_KAYENTA = os.getenv("PAYETTE_KAYENTA")
PAYETTE_ALEGRANEVADA = os.getenv("PAYETTE_ALEGRANEVADA")
PAYETTE_NLOPT = os.getenv("NLOPTLOC")

# --- configuration files
PAYETTE_CONFIG_FILE = os.path.join(PAYETTE_ROOT, "Payette_config.py")
try:
    os.remove(PAYETTE_CONFIG_FILE)
except OSError:
    pass
try:
    os.remove(PAYETTE_CONFIG_FILE + "c")
except OSError:
    pass

# --- subdirectories of PAYETTE_AUX
PAYETTE_INPUTS = os.path.join(PAYETTE_ROOT, "Aux/Inputs")
ERRORS += check_exists("PAYETTE_INPUTS", PAYETTE_INPUTS)

# --- subdirectories of PAYETTE_SOURCE
PAYETTE_MATERIALS = os.path.join(PAYETTE_SOURCE, "Materials")
PAYETTE_MIG_UTILS = os.path.join(PAYETTE_SOURCE, "Fortran/migutils.F")
ERRORS += check_exists("PAYETTE_MATERIALS", PAYETTE_MATERIALS)
ERRORS += check_exists("PAYETTE_MIG_UTILS", PAYETTE_MIG_UTILS)

# --- Subdirectories of PAYETTE_MATERIALS
PAYETTE_MATERIALS_LIBRARY = os.path.join(PAYETTE_MATERIALS, "Library")
PAYETTE_MATERIALS_FORTRAN = os.path.join(PAYETTE_MATERIALS, "Fortran")
PAYETTE_MATERIALS_FORTRAN_INCLUDES = os.path.join(PAYETTE_MATERIALS,
                                                  "Fortran/Includes")
PAYETTE_MATERIALS_FILE = os.path.join(PAYETTE_MATERIALS,
                                      "Payette_installed_materials.py")
ERRORS += check_exists("PAYETTE_MATERIALS_LIBRARY", PAYETTE_MATERIALS_LIBRARY)
ERRORS += check_exists("PAYETTE_MATERIALS_FORTRAN", PAYETTE_MATERIALS_FORTRAN)
ERRORS += check_exists("PAYETTE_MATERIALS_FORTRAN_INCLUDES",
                       PAYETTE_MATERIALS_FORTRAN_INCLUDES)

# --- extension module file extension
PAYETTE_EXTENSION_MODULE_FEXT = sysconfig.get_config_var("SO")

# --- if OSTYPE is not defined, just set it to linux
if not os.getenv("OSTYPE"):
    logwrn("environment variable OSTYPE not set, "
           "setting PAYETTE_OSTYPE to linux")
    PAYETTE_OSTYPE = "linux"
else:
    PAYETTE_OSTYPE = os.getenv("OSTYPE").lower()

# Store all of the above information for writing to the PAYETTE_CONFIG_FILE. We
# waited to write it til now so that we would only write it if everything was
# configured correctly.
PAYETTE_CONFIG = {}
PAYETTE_CONFIG["PAYETTE_PYINT"] = PAYETTE_PYINT
PAYETTE_CONFIG["PAYETTE_ROOT"] = PAYETTE_ROOT
PAYETTE_CONFIG["PAYETTE_AUX"] = PAYETTE_AUX
PAYETTE_CONFIG["PAYETTE_DOCUMENTS"] = PAYETTE_DOCUMENTS
PAYETTE_CONFIG["PAYETTE_SOURCE"] = PAYETTE_SOURCE
PAYETTE_CONFIG["PAYETTE_TESTS"] = PAYETTE_TESTS
PAYETTE_CONFIG["PAYETTE_TOOLSET"] = PAYETTE_TOOLSET
PAYETTE_CONFIG["PAYETTE_MATERIALS"] = PAYETTE_MATERIALS
PAYETTE_CONFIG["PAYETTE_MIG_UTILS"] = PAYETTE_MIG_UTILS
PAYETTE_CONFIG["PAYETTE_MATERIALS_LIBRARY"] = PAYETTE_MATERIALS_LIBRARY
PAYETTE_CONFIG["PAYETTE_MATERIALS_FORTRAN"] = PAYETTE_MATERIALS_FORTRAN
PAYETTE_CONFIG["PAYETTE_MATERIALS_FORTRAN_INCLUDES"] = (
    PAYETTE_MATERIALS_FORTRAN_INCLUDES)
PAYETTE_CONFIG["PAYETTE_MATERIALS_FILE"] = PAYETTE_MATERIALS_FILE
PAYETTE_CONFIG["PAYETTE_INPUTS"] = PAYETTE_INPUTS
PAYETTE_CONFIG["PAYETTE_EXTENSION_MODULE_FEXT"] = (
    PAYETTE_EXTENSION_MODULE_FEXT)
PAYETTE_CONFIG["PAYETTE_OSTYPE"] = PAYETTE_OSTYPE
PAYETTE_CONFIG["PAYETTE_RUNTEST"] = PAYETTE_RUNTEST
PAYETTE_CONFIG["PAYETTE_RUN"] = PAYETTE_RUN
PAYETTE_CONFIG["PAYETTE_BUILD"] = PAYETTE_BUILD
PAYETTE_CONFIG["PAYETTE_EXECUTABLES"] = PAYETTE_EXECUTABLES
PAYETTE_CONFIG["PAYETTE_CONFIG_FILE"] = PAYETTE_CONFIG_FILE
PAYETTE_CONFIG["PAYETTE_KAYENTA"] = PAYETTE_KAYENTA
PAYETTE_CONFIG["PAYETTE_ALEGRANEVADA"] = PAYETTE_ALEGRANEVADA
PAYETTE_CONFIG["PAYETTE_NLOPT"] = PAYETTE_NLOPT
PAYETTE_CONFIG["PAYETTE_F2PY"] = PAYETTE_F2PY
PAYETTE_CONFIG["RUNPAYETTE"] = RUNPAYETTE
PAYETTE_CONFIG["TESTPAYETTE"] = TESTPAYETTE
PAYETTE_CONFIG["BUILDPAYETTE"] = BUILDPAYETTE
PAYETTE_CONFIG["CLEANPAYETTE"] = CLEANPAYETTE
PAYETTE_CONFIG["PAYETTE_BUILT_EXECUTABLES"] = PAYETTE_BUILT_EXECUTABLES
PAYETTE_CONFIG["PAYETTE_NUMPY_VERSION"] = PAYETTE_NUMPY_VERSION
PAYETTE_CONFIG["PAYETTE_SCIPY_VERSION"] = PAYETTE_SCIPY_VERSION

# --- if running with sage, configure the sage environment
SAGE = True if "sage" in PAYETTE_PYINT.lower() else False

# --- set up the environment
ENV = {}
ENVS = ["MPLCONFIGDIR", "PYTHONPATH", "ECLDIR", "GPDOCDIR", "RHOME",
        "GP_DATA_DIR", "PKG_CONFIG_PATH", "PYTHONHOME", "LD_LIBRARY_PATH",
        "LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH", "SINGULAR_EXECUTABLE",
        "SINGULARPATH"]
if SAGE:
    # get the sage environment to save
    for skey, sval in os.environ.items():
        if "sage" in sval.lower() or "sage" in skey.lower():
            ENVS.append(skey)
            continue

if ERRORS:
    sys.exit("configure.py: ERROR: fix previously trapped errors")


def configure_payette(argv):

    """ create and write configuration file """

    # *************************************************************************
    # -- command line option parsing
    usage = ("usage: python %prog [options]\nmust be executed from "
             "{0}".format(os.path.dirname(PAYETTE_ROOT)))
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
        default="gfortran",
        help="Specify the path F77 to compiler [default: %default]")
    parser.add_option(
        "--f90exec",
        dest="F90EXEC",
        action="store",
        default="gfortran",
        help="Specify the path F90 to compiler [default: %default]")
    parser.add_option(
        "--no-callback",
        dest="NOCALLBACK",
        action="store_true",
        default=False,
        help="Compile with f2py callbacks functions [default: %default]")

    opts = parser.parse_args(argv)[0]

    # configure Payette
    loginf("configuring Payette environment")

    # f2py call back
    if MAJOR == 3 or SAGE:
        opts.NOCALLBACK = True
    PAYETTE_CONFIG["PAYETTE_F2PY_CALLBACK"] = not opts.NOCALLBACK

    # f2py fortran compiler options
    if opts.FCOMPILER:
        PAYETTE_CONFIG["PAYETTE_FCOMPILER"] = opts.FCOMPILER
        PAYETTE_CONFIG["PAYETTE_F77EXEC"] = None
        PAYETTE_CONFIG["PAYETTE_F90EXEC"] = None
    else:
        PAYETTE_CONFIG["PAYETTE_FCOMPILER"] = None
        PAYETTE_CONFIG["PAYETTE_F77EXEC"] = get_exe_path(opts.F77EXEC)
        PAYETTE_CONFIG["PAYETTE_F90EXEC"] = get_exe_path(opts.F90EXEC)

    for item in ENVS:
        if item in os.environ:
            ENV[item] = os.environ[item]
        continue

    # make sure PAYETTE_ROOT is first on PYTHONPATH
    pypath = os.pathsep.join([PAYETTE_ROOT, PAYETTE_TOOLSET,
                              PAYETTE_MATERIALS])
    if "PYTHONPATH" in ENV:
        pypath += (os.pathsep +
                   os.pathsep.join([x for x in ENV["PYTHONPATH"].split(os.pathsep)
                                    if x not in pypath.split(os.pathsep)]))
    ENV["PYTHONPATH"] = pypath

    # write the the configuration file
    begmes("writing Payette_config.py", pre=SPACE)
    with open(PAYETTE_CONFIG_FILE, "w") as fnew:
        fnew.write(PREAMBLE + "\n" +
                   'PAYETTE_INTRO = """{0}"""\n'.format(PAYETTE_INTRO))
        for key, val in PAYETTE_CONFIG.items():
            fnew.write(dictfrmt(key, val) + "\n")
            continue
        fnew.write("if PAYETTE_ROOT not in sys.path: "
                   "sys.path.insert(0, PAYETTE_ROOT)\n")
        for key, val in ENV.items():
            fnew.write('os.environ["{0}"] = "{1}"\n'.format(key, val))
            continue

    endmes("Payette_config.py written")

    # try importing the file we just wrote to test if it is importable
    try:
        import Payette_config
        if not Payette_config.PAYETTE_BUILT_EXECUTABLES:
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
    exit_msg = """
if [ ! -f {0} ]; then
   echo "buildPayette must be executed to create {0}"
   exit
fi

""".format(PAYETTE_MATERIALS_FILE)

    for nam, path in PAYETTE_BUILT_EXECUTABLES.items():

        # remove the executable first
        try:
            os.remove(path)
        except OSError:
            pass

        begmes("writing {0}".format(nam), pre=SPACE)

        if path == PAYETTE_F2PY:
            write_f2py(PAYETTE_PYINT, PAYETTE_TOOLSET)
            endmes("{0} script written".format(nam))
            continue

        with open(path, "w") as fnew:
            fnew.write("#!/bin/sh -f\n")
            for key, val in ENV.items():
                fnew.write("export {0}={1}\n".format(key, val))
                continue

            if path == CLEANPAYETTE:
                fnew.write("{0} {1} {2} $* 2>&1\n"
                           .format(PAYETTE_PYINT, THIS_FILE, "clean"))

            elif path == RUNPAYETTE:
                fnew.write(exit_msg)
                fnew.write("{0} {1} $* 2>&1\n"
                           .format(PAYETTE_PYINT, PAYETTE_RUN))

            elif path == TESTPAYETTE:
                fnew.write(exit_msg)
                fnew.write("{0} {1} $* 2>&1\n"
                           .format(PAYETTE_PYINT, PAYETTE_RUNTEST))

            elif path == BUILDPAYETTE:
                fnew.write("{0} {1} $* 2>&1\n"
                           .format(PAYETTE_PYINT, PAYETTE_BUILD))

        os.chmod(path, 0o750)
        endmes("{0} script written".format(nam))
        continue

    loginf("executable scripts written\n")

    return ERRORS


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

    if os.path.isfile(exe):
        return exe

    try:
        path = os.getenv("PATH").split(os.pathsep)
    except AttributeError:
        path = []

    for dirname in path:
        if os.path.isfile(os.path.join(dirname, exe)):
            return os.path.join(dirname, exe)

    sys.exit("ERROR: executable {0} not found".format(exe))


PREAMBLE = """
# *************************************************************************** #
#                                                                             #
# This file was generated automatically by the Payette. It contains important #
# global Payette parameters that are configured at build time.                #
#                                                                             #
# This file is intended to be imported by Payette using                       #
# "from Payette_config import *"                                              #
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
                      "Payette_installed_materials.py", "*{0}".format(soext),
                      "*.log", "*.echo", "*.prf", "*.diff", "*.xout", "*.out",
                      "*.math1", "*.math2", "*.props", "*.vtable", "*.dtable"]
    pats_to_remove.extend(PAYETTE_BUILT_EXECUTABLES.keys())

    for item in os.walk(PAYETTE_ROOT):
        dirnam, files = item[0], item[2]

        if ".svn" in dirnam:
            continue

        for fnam in files:
            if any(fnmatch(fnam, pat) for pat in pats_to_remove):
                os.remove(os.path.join(dirnam, fnam))

            continue

        continue

    return

if __name__ == "__main__":

    if "clean" in sys.argv:
        loginf("cleaning Payette")
        clean_payette()
        loginf("Payette cleaned")
        sys.exit(0)

    if sys.argv[0] != os.path.basename(__file__):
        sys.exit("configure.py must be executed from {0}".format(PAYETTE_ROOT))

    # introduce yourself
    logmes(PAYETTE_INTRO)

    # clean up first
    clean_payette()

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
