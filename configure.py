#!/usr/bin/env python
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



import os
import sys
from distutils import sysconfig
import subprocess
import optparse
import shutil
import re

# git hash
GIT = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".git")
HEAD = os.path.join(
    GIT, open(os.path.join(GIT, "HEAD")).readlines()[0].split()[1])
__hash__ = open(HEAD).read().strip()

__version_info__ = (1, 2, 0, __hash__[:7])
__version__ = ".".join(str(x) for x in __version_info__)
__author__ = ("Tim Fuller, tjfulle@sandia.gov", "Scot Swan, mswan@sandia.gov")
__intro__ = """
Copyright (2011) Sandia Corporation. Under the terms of Contract
DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
rights in this software.


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

(major, minor, micro, releaselevel, serial) = sys.version_info

# --- Environmental variables used by Payette
USER_ENV = {"DOTPAYETTE": os.getenv("DOTPAYETTE"),
            "BENCHDIR": os.getenv("PAYETTE_BENCHDIR"),
            "MTLDIR": os.getenv("PAYETTE_MTLDIR"),
            }

# supported fortran vendors
FORTRAN_VENDORS = {"gnu": {"exe": "gfortran", "pre": "-E"}}

def configure(argv):
    """Parse user arguments and create the Payette configuration"""

    # *************************************************************************
    # -- command line option parsing
    usage = "usage: python %prog [options]"
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
        choices=("gnu",),
        default="gnu",
        help="Specify Fortran compiler type by vendor [default: %default]")
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
        "-M", "--mtldirs",
        dest="MTLDIRS",
        action="append",
        default=[],
        help="Additional directories to scan for materials [default: %default]")
    parser.add_option(
        "-B",
        dest="DONTWRITEBYTECODE",
        action="store_true",
        default=sys.dont_write_bytecode,
        help="Don't write bytecode files [default: %default]")
    parser.add_option(
        "-E",
        dest="SKIPENVIRON",
        action="store_true",
        default=False,
        help="Do not use user environment [default: %default]")
    parser.add_option(
        "--no-clean",
        dest="NOCLEAN",
        action="store_true",
        default=False,
        help="Don't clean when configuring [default: %default]")
    parser.add_option(
        "--deployed",
        dest="DEPLOYED",
        action="store_true",
        default=False,
        help="Configure to be deployed to many users [default: %default]")

    opts, args = parser.parse_args(argv)

    logmes(__intro__, skip=False, pre="")
    if opts.DEPLOYED:
        opts.DONTWRITEBYTECODE = True

    sys.dont_write_bytecode = opts.DONTWRITEBYTECODE

    use_env = not opts.SKIPENVIRON and not opts.DEPLOYED
    cfg = PayetteConfig(use_env=use_env, fcompiler=opts.FCOMPILER,
                        deployed=opts.DEPLOYED, callback=not opts.NOCALLBACK)

    # clean up first
    if not opts.NOCLEAN:
        clean_payette()

    # configure Payette
    logmes("Configuring Payette environment")

    # add to tests
    _user_tests = [os.path.expanduser(x) for x in opts.BENCHDIRS]
    cfg.add_user_tests(_user_tests)

    # add to materials
    _user_mtls = [os.path.expanduser(x) for x in opts.MTLDIRS]
    cfg.add_user_mtls(_user_mtls)

    # ------ Report on environmental variables --------------------------------
    if use_env:
        logmes("Checking for Payette-related environmental variables")
        for key, val in USER_ENV.items():
            if val is not None:
                msg = "set to {0}".format(val)
            else:
                msg = "not set"
            logmes("{0} {1}".format(key, msg), pre="indent")

    # ------ Write out the configuration file and executable files ------------
    cfg.write_cfg()
    cfg.write_exes()
    return


class PayetteConfig:
    errors = 0

    # --- base directories
    intro = __intro__
    root = os.path.dirname(os.path.realpath(__file__))
    aux = os.path.join(root, "Aux")
    inputs = os.path.join(root, "Aux/Inputs")
    docs = os.path.join(root, "Documents")
    source = os.path.join(root, "Source")
    optrec = os.path.join(source, "OptRecipes")
    tests = [os.path.join(root, "Benchmarks")]
    toolset = os.path.join(root, "Toolset")
    fortran = os.path.join(source, "Fortran")
    version = __version__
    version_info = __version_info__

    # materials is the directory where we search for Payette material models
    materials = os.path.join(source, "Materials")
    models = os.path.join(materials, "Models")
    library = os.path.join(materials, "Library")
    includes = os.path.join(materials, "Includes")
    mtldirs = [models]

    # --- Payette configuration files
    hdir = os.path.join(root, ".payette")
    config_file = os.path.join(source, "__config__.py")

    # fortran files
    mig_utils = os.path.join(fortran, "migutils.F")
    tensor_tools = os.path.join(fortran, "tensor_toolkit.f90")

    # --- built executables
    executables = [
        {"name": "extractPayette",
         "pyfile": os.path.join(source, "Payette_extract.py"),
         "path": os.path.join(toolset, "extractPayette")},
        {"name": "payette",
         "pyfile": os.path.join(source, "Payette_mgr.py"),
         "path": os.path.join(toolset, "payette")},
        {"name": "buildPayette",
         "pyfile": os.path.join(source, "Payette_build.py"),
         "path": os.path.join(toolset, "buildPayette")},
        {"name": "testPayette",
         "pyfile": os.path.join(source, "Payette_runtest.py"),
         "path": os.path.join(toolset, "testPayette")},]

    # --- python interpreter info
    pyint = os.path.realpath(sys.executable)
    sage = True if "sage" in pyint.lower() else False
    pyver = "python" if not sage else "sage -python"
    pyver = "{0} {1}.{2}.{3}".format(pyver, major, minor, micro)
    ext_mod_fext = sysconfig.get_config_var("SO")

    # --- custom environment
    # --- set up the environment
    env = {}
    envs = ["MPLCONFIGDIR", "PYTHONPATH", "ECLDIR", "GPDOCDIR", "RHOME",
            "GP_DATA_DIR", "PKG_CONFIG_PATH", "PYTHONHOME", "LD_LIBRARY_PATH",
            "LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH", "SINGULAR_EXECUTABLE",
            "SINGULARPATH"]

    # --- platform info
    ostype = sys.platform.lower()

    # --- f2py setup
    f2py = {"fexe": "gfortran", "f2py": "f2py",
            "callback": major != 3 and not sage,}

    # --- visualization
    viz_compatible = False

    def __init__(self, use_env=True, fcompiler="gnu", deployed=False,
                 callback=True):
        """Check prerequisites and initialize the PayetteConfig object

        Parameters
        ----------
        use_env : bool
            Use the user environment when configuring

        """
        # -- check prerequisites
        self.check_prereqs()
        if self.root not in sys.path:
            sys.path.append(self.root)

        # --- dotpayette directory
        self.deployed = deployed
        if self.deployed:
            self.use_env = False
            self.dotpayette = os.path.join(self.root, ".payette")
        else:
            if USER_ENV["DOTPAYETTE"] is not None:
                self.dotpayette = os.path.realpath(USER_ENV["DOTPAYETTE"])
            elif "darwin" in self.ostype:
                self.dotpayette = os.path.expanduser("~/Library/Preferences/Payette")
            else:
                self.dotpayette = os.path.expanduser("~/.payette")

        try: os.makedirs(self.dotpayette)
        except OSError: pass

        # location of material database file and library directory
        self.libdir = self.library
        self.mtldb = os.path.join(self.libdir, "materials.db")
        self.auxdb = "auxiliary_materials.db"

        # files that go in the dotpayette directory
        self.user_config_file = os.path.join(
            self.dotpayette, "__user_config__.py")
        self.prev_tests = os.path.join(self.dotpayette, "__prev_tests__.py")

        # --- setup user environment defined defaults
        self.use_env = use_env
        if self.use_env and not self.deployed:
            self.setup_from_user_env()

        # --- if running with sage, configure the sage environment
        if self.sage:
            self.config_sage()
            self.set_callback_mode(mode=False)
        else:
            self.set_callback_mode(mode=callback)

        # --- f2py setup
        self.setup_f2py(fcompiler)

        # --- set up the rest of the environment
        self.setup_env()

        # --- check if enthought is installed
        self.check_viz()

        if self.errors:
            raise SystemExit(
                "configure.py: ERROR: fix previously trapped errors")

        pass

    def check_prereqs(self):
        """Check that the python interpreter satisifies the prerequisites

        """
        # check where we are being executed
        fpath = os.path.basename(__file__)
        if sys.argv[0] != fpath:
            raise SystemExit(
                "configure.py must be executed from {0}".format(self.root))
        if not os.path.isdir(os.path.join(self.hdir)):
            raise SystemExit(
                "configure.py must be executed in the Payette root directory")

        if (major != 3 and major != 2) or (major == 2 and minor < 6):
            raise SystemExit("Payette requires Python >= 2.6\n")

        # --- numpy check
        try: import numpy
        except ImportError: raise SystemExit("numpy not importable")

        # --- scipy check
        try: import scipy
        except ImportError: raise SystemExit("scipy not importable")

    def expand_mtldirs(self):
        """Expand mtldirs to only include those with control files

        """
        self.mtldirs = self.walk_mtldirs()
        return

    def walk_mtldirs(self, path=None):
        """Walk through mtldirs and find directories that have Payette control
        files

        """
        if path is None:
            path = self.mtldirs
        if not isinstance(path, (list, tuple)):
            path = [path]

        mtldirs = []
        R0 = re.compile(r"\.[(svn)(git)]+")
        R1 = re.compile("Build_[0-9]{4}\.[0-9]{2}\.[0-9]{2}")
        R2 = re.compile(r"(?i).*_control.xml")
        for mtldir in path:
            mtldirs.extend([dirnam for dirnam, dirs, files in os.walk(mtldir)
                            if (not R0.search(dirnam) and
                                not R1.search(dirnam) and
                                R2.search(" ".join(files)))])
            continue
        return list(set(mtldirs))

    def expand_tests(self):
        """Walk through tests and find directories that have __test_dir__.py
        files

        """
        tests = []
        for d in self.tests:
            tests.extend([dirnam for dirnam, dirs, files in os.walk(d)
                          if "__test_dir__.py" in files])
        self.tests = list(set(tests))
        return

    def exists(self, paths):
        """ check if item exists on file system """
        retval = True
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        for path in paths:
            if dirnam is '':
                continue
            if not os.path.exists(path):
                retval = False
                self.increment_error_count("'{0}' not found".format(path))
            continue
        return retval

    def setup_from_user_env(self):
        """Extend the base configuration by including information from the
        user environment

        """
        _user_tests = USER_ENV["BENCHDIR"]
        if _user_tests is not None:
            self.add_user_tests(_user_tests.split(os.pathsep))

        _user_mtls = USER_ENV["MTLDIR"]
        if _user_mtls is not None:
            _user_mtls = [x for x in _user_mtls.split(os.pathsep)]
            self.add_user_mtls(_user_mtls)

        return

    def add_user_tests(self, user_tests):
        """Extend the base configuration by including information from the
        user environment

        """
        # extend tests to include user tests
        if not isinstance(user_tests, (list, tuple)):
            user_tests = [user_tests]
        for dirnam in user_tests:
            if dirnam is '':
                continue
            if not os.path.isdir(dirnam):
                self.increment_error_count("'{0}' not found".format(dirnam))
                continue
            self.tests.append(dirnam)
            continue
        return

    def add_user_mtls(self, user_mtls):
        """Extend the base configuration by including information from the
        user environment

        """
        # extend materials to include user materials
        mtldirs = []
        if not isinstance(user_mtls, (list, tuple)):
            user_mtls = [user_mtls]

        for dirnam in user_mtls:
            if dirnam is '':
                continue
            if not os.path.isdir(dirnam):
                self.increment_error_count("'{0}' not found".format(dirnam))
                continue
            mtldirs.append(dirnam)
            continue
        self.mtldirs.extend(self.walk_mtldirs(mtldirs))
        return

    def setup_env(self):
        """get current environment

        """
        for item in self.envs:
            if item in os.environ:
                self.env[item] = os.environ[item]
            continue
        if sys.dont_write_bytecode:
            self.env["PYTHONDONTWRITEBYTECODE"]="X"
        else:
            self.env["PYTHONDONTWRITEBYTECODE"]=""

        # make sure root is first on PYTHONPATH
        pypath = []
        tmp = self.env.get("PYTHONPATH")
        if tmp is not None:
            for path in tmp.split(os.pathsep):
                if os.path.isdir(path) and path not in pypath:
                    pypath.append(path)
                continue
        if self.root not in pypath:
            pypath.append(self.root)
        self.env["PYTHONPATH"] = os.pathsep.join(pypath)
        return

    def write_cfg(self):
        """Write the configuration file"""
        self.expand_tests()
        self.expand_mtldirs()

        # setup lists of attributes for the global and user configuration files
        do_not_write = ("__doc__", "__module__", "env", "envs", "errors",
                        "dotpayette", "config_file",
                        "user_config_file", "use_env")
        user_attributes = ("prev_tests",)
        global_attributes = [x for x in sorted(dir(self)) if
                             x not in user_attributes and
                             x not in do_not_write]

        # remove current configuration files
        remove(self.config_file)
        remove(self.user_config_file)

        logmes("writing {0}".format(os.path.basename(self.config_file)),
               end="...  ")
        attributes = global_attributes
        with open(self.config_file, "w") as fobj:
            fobj.write("""\
# *************************************************************************** #
#                                                                             #
# This file was generated automatically by the Payette. It contains important #
# global Payette parameters that are configured at build time.                #
#                                                                             #
# This file is intended to be imported by files in Payette like               #
# "from config import XXXXX"                                                  #
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
""")
            fobj.write("from __user_config__ import *\n")
            for attrnam in attributes:
                attr = getattr(self, attrnam)
                if "instancemethod" in str(type(attr)):
                    continue
                attribute = repr(attr)
                fobj.write("{0} = {1}\n".format(attrnam.upper(), attribute))
                continue
            fobj.write("if ROOT not in sys.path: sys.path.append(ROOT)\n")
            # add all material directories to sys.path
            fobj.write("for PATH in MTLDIRS:\n"
                       "    if os.path.basename(PATH) != 'code': "
                       "sys.path.append(PATH)\n")
        logmes("{0} written".format(os.path.basename(self.config_file)),
               pre="")

        # write the user configuration file
        attributes = user_attributes
        user_config = os.path.join(
            self.hdir, os.path.basename(self.user_config_file) + ".copy")
        with open(user_config, "w") as fobj:
            fobj.write("""\
# *************************************************************************** #
#                                                                             #
# This file was generated automatically by the Payette. It contains important #
# global Payette parameters that are configured at build time.                #
#                                                                             #
# This file is intended to be imported by files in Payette like               #
# "from config import XXXXX"                                                  #
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
""")
            fobj.write(
                "DOTPAYETTE = os.path.dirname(os.path.realpath(__file__))\n")
            for attrnam in attributes:
                attr = getattr(self, attrnam)
                if "instancemethod" in str(type(attr)):
                    continue
                attribute = repr(attr)
                if self.dotpayette in attribute:
                    post = attribute.replace(self.dotpayette, "")
                    if post[1] == os.sep:
                        post = post.replace(os.sep, "")
                    attribute = "os.path.join(DOTPAYETTE, {0})".format(post)
                fobj.write("{0} = {1}\n".format(attrnam.upper(), attribute))
                continue
            fobj.write(
                "if DOTPAYETTE not in sys.path: sys.path.append(DOTPAYETTE)\n")
        if not self.deployed:
            shutil.copyfile(user_config, self.user_config_file)
        return

    def config_sage(self):
        """configure for running with sage"""
        # get the sage environment to save
        for skey, sval in os.environ.items():
            if "sage" in sval.lower() or "sage" in skey.lower():
                self.envs.append(skey)
                continue
            sage_local = os.environ.get("SAGE_LOCAL")
            if sage_local is not None:
                sage_gfortran = os.path.join(sage_local, "gfortran")
                if os.path.isfile(sage_gfortran):
                    self.fcompiler = sage_gfortran
            continue
        return

    def check_viz(self):
        # --- visualization check
        display = os.environ.get("DISPLAY")
        logmes(
            "Checking if visualizaton suite is supported by Python distribution")
        if display is not None:
            try:
                # see if enthought tools are installed
                from enthought.traits.api import (
                    HasStrictTraits, Instance, String, Button,
                    Bool, List, Str, Property, HasPrivateTraits, on_trait_change)
                from enthought.traits.ui.api import (
                    View, Item, HSplit, VGroup, Handler,
                    TabularEditor, HGroup, UItem)
                from enthought.traits.ui.tabular_adapter import TabularAdapter
                logmes("Visualizaton suite supported by Python distribution")
                self.viz_compatible = True

            except ImportError:
                logmes("Visualizaton suite not supported by Python distribution")

        else:
            logmes("DISPLAY not set, visualization suite cannot be imported")

    def set_callback_mode(self, mode):
        """

        Parameter
        ---------
        mode : bool
        """
        self.f2py["callback"] = mode
        return

    def increment_error_count(self, message):
        self.errors += 1
        logerr(message)
        return

    def write_exes(self):
        """ create the Payette executables """

        logmes("Writing executable scripts")

        # message for executables that require Payette be built
        do_not_include = ("pythonpath", )
        for executable in self.executables:
            name = executable["name"]
            pyfile = executable["pyfile"]
            path = os.path.join(self.toolset, name)

            # remove the executable first
            remove(path)
            logmes("writing {0}".format(name), pre="indent", end="...  ")
            with open(path, "w") as fobj:
                fobj.write("#!/bin/sh -f\n")
                pypath = self.env["PYTHONPATH"]
                if not self.deployed:
                    fobj.write("DOTPAYETTE={0}\n".format(self.dotpayette))
                fobj.write(self.get_dotpayette())
                fobj.write("export PYTHONPATH={0}:$DOTPAYETTE\n".format(pypath))
                for key, val in self.env.items():
                    if key.lower() in do_not_include:
                        continue
                    fobj.write("export {0}={1}\n".format(key, val))
                    continue
                fobj.write("PYTHON={0}\n".format(self.pyint))
                fobj.write("PYFILE={0}\n".format(pyfile))
                fobj.write("$PYTHON $PYFILE $*\n")

            os.chmod(path, 0o750)
            logmes("{0} script written".format(name), pre="")
            continue

        path = os.path.join(self.toolset, "pconfigure")
        logmes("writing pconfigure", pre="indent", end="...  ")
        with open(path, "w") as fobj:
            fobj.write("#!/bin/sh -f\n")
            fobj.write("cd {0}\n".format(self.root))
            fobj.write("{0} {1}\n".format(self.pyint, " ".join(sys.argv)))
        os.chmod(path, 0o750)
        logmes("pconfigure written", pre="")

        logmes("Executable scripts written")

        return

    def setup_f2py(self, fcompiler):
        """Setup the f2py executable

        """
        # get fortran executables
        fortran = FORTRAN_VENDORS.get(fcompiler)
        if fortran is None:
            raise SystemExit(
                "{0} not a supported fortran vendor".format(fcompiler))
        pypath = os.path.dirname(self.pyint)
        fexe = get_exe_path(fortran["exe"], path=pypath)
        if fexe is None:
            fexe = get_exe_path(fortran["exe"])
            if fexe is None:
                raise SystemExit(
                    "fortran executable {0} not found".format(fortran["exe"]))

        self.f2py["fexe"] = fexe
        return

    def get_dotpayette(self):
        return """\
DIR={0}
if [ -z "${{DOTPAYETTE:+x}}" ]; then
    ostype=`uname -a | tr '[A-Z]' '[a-z]'`
    if [ -n "$(echo $ostype | grep 'darwin')" ];
    then
        DOTPAYETTE=$HOME/Library/Preferences/Payette
    else
        DOTPAYETTE=$HOME/.payette
    fi
fi
if [ ! -d "$DOTPAYETTE" ]; then
    echo "INFO: Creating $DOTPAYETTE"
    mkdir -p $DOTPAYETTE
fi
if [ ! -f "$DOTPAYETTE/__user_config__.py" ]; then
    echo "INFO: Copying __user_config__.py to $DOTPAYETTE"
    cp $DIR/../.payette/__user_config__.py.copy "$DOTPAYETTE/__user_config__.py"
fi
""".format(self.toolset)

def clean_payette():
    root = os.path.dirname(os.path.realpath(__file__))
    cp = os.path.join(root, "Toolset/cleanPayette")
    subprocess.call(cp, shell=True)
    return


def remove(paths):
    """Remove paths"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    for path in paths:
        pyc = path + ".c" if path.endswith(".py") else None
        try: os.remove(path)
        except OSError: pass
        try: os.remove(pyc)
        except OSError: pass
        except TypeError: pass
        continue
    return


def logmes(message, pre="INFO: ", end="\n", skip=True):
    """ log info """
    if pre.lower() == "indent":
        pre = " " * 6
    for line in message.split("\n"):
        if not line.split() and skip:
            continue
        sys.stdout.write("{0}{1}{2}".format(pre, line, end))
    return


def logerr(message, pre="ERROR: ", end="\n"):
    """ log error """
    logmes(message, pre=pre, end=end)
    return


def get_exe_path(exenam, syspath=[], path=None):
    """Get the full path to exenam"""
    if not syspath:
        syspath.append(os.getenv("PATH", "").split(os.pathsep))
    search_path = syspath[0] if path is None else [path]
    for dirnam in search_path:
        exepath = os.path.join(dirnam, exenam)
        if os.path.isfile(exepath):
            return exepath
    else:
        return None

if __name__ == "__main__":
    sys.exit(configure(sys.argv[1:]))
