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
import shutil
import imp
import re
import tempfile
import subprocess as sbp
from distutils import sysconfig
from copy import deepcopy
from numpy.f2py import main as f2py

import Source.__config__ as cfg
import Source.Payette_utils as pu


class MaterialBuilder(object):

    def __init__(self, name, libname, srcd, libdir, compiler_info,
                 sigf=None, incdirs=None, libdirs=None, libs=None):
        """
        Initialize the MaterialBuilder object.

        Parameters
        ----------
        name : str
           name of material
        libname : str
           name of library to be built
        srcd : path
           path to source directory
        libdir : path
           path to library directory where finished library will be copied
        compiler_info : dict
           information about compiler
        sigf : path, optional
           path to f2py signature file
        incdirs : path, optional
           path to include directories
        libdirs : path, optional
           path to library directories
        libs : path, optional
           libraries to use when compiling

        """

        self.name = name
        self.needs_mig_utils = True
        self.source_directory = srcd
        self.source_files = []
        self.libname = libname
        self.pre_directives = []
        self.use_lapack = False

        # directory to copy libraries
        if not os.path.isdir(libdir):
            pu.report_and_raise_error(
                "library directory {0} not found".format(libdir))
        self.payette_libdir = libdir

        # signature file and migutils
        self.signature_file = sigf

        self.nocallback_file = os.path.join(self.source_directory,
                                            "nocallback.signature.pyf")

        # format the signature file
        if self.signature_file is not None:
            self.format_signature_file()

            if not cfg.F2PY["callback"]:
                self.pre_directives.append("-DNOPYCALLBACK")

            if not os.path.isfile(self.signature_file):
                pu.report_and_raise_error(
                    "{0} signature file not found".format(self.name), errno=40)

        # include directories
        self.incdirs = [".", "{0}".format(self.source_directory),
                        "{0}".format(cfg.INCLUDES)]
        if incdirs is not None:
            if not isinstance(incdirs, (list, tuple)):
                incdirs = [incdirs]
            for incdir in incdirs:
                if not os.path.isdir(incdir):
                    pu.report_and_raise_error(
                        "incdir {0} not found".format(incdir), errno=60)
                self.incdirs.append(incdir)
        self.libdirs = []
        if libdirs is not None:
            if not isinstance(libdirs, (list, tuple)):
                libdirs = [libdirs]
            for libdir in libdirs:
                if not os.path.isdir(libdir):
                    pu.report_and_raise_error(
                        "libdir {0} not found".format(libdir), errno=60)
                self.libdirs.append(libdir)
        self.libs = []
        if libs is not None:
            if not isinstance(libs, (list, tuple)):
                libs = [libs]
            self.libdirs.extend(libs)

        # f2py opts
        f2py_opts = compiler_info["f2py"]["options"]
        self.f2pyopts = ["f2py", "-c"] + f2py_opts
        pass

    def add_compile_directives(self, D):
        self.pre_directives.extend(["-D{0}".format(x) for x in D])
        return

    def build_extension_module(self):
        pu.report_and_raise_error(
            "fortran build script must provide this function", errno=1)

    def build_extension_module_with_f2py(self):
        fcn = "build_extension_module_with_f2py"

        if not self.source_files:
            pu.report_and_raise_error(
                "no source files sent to {0}".format(fcn), errno=1)

        elif not isinstance(self.source_files, list):
            self.source_files = [self.source_files]

        for srcf in self.source_files:
            if not os.path.isfile(srcf):
                pu.report_error("source file {0} not found".format(srcf))
            continue

        if pu.error_count():
            pu.report_and_raise_error(
                "Stopping due to previous errors.", errno=10)

        # remove extension module files if they exist
        for d in [self.source_directory, self.payette_libdir]:
            try:
                os.remove(os.path.join(d, self.libname))
            except OSError:
                pass

            continue

        self.pre_directives.append("-DPAYETTE_SINGLE_CELL_DRIVER")
        self.pre_directives = list(set(self.pre_directives))

        # f2py pulls its arguments from sys.argv. Here, we build sys.argv to what
        # f2py expects. Later sys.argv will be restored.
        ffiles = [x for x in self.source_files]
        if self.needs_mig_utils:
            ffiles.append(cfg.MIG_UTILS)
        incsearch = ["-I{0}".format(x) for x in self.incdirs]
        libsearch = ["-L{0}".format(x) for x in self.libdirs]
        libs = ["-l{0}".format(x) for x in self.libs]
        f2pycmd = self.f2pyopts + incsearch + libsearch + libs
        f2pycmd.extend(["-m", self.name, self.signature_file])
        f2pycmd.extend(self.pre_directives)
        if self.use_lapack:
            f2pycmd.append("--link-lapack_opt")
        f2pycmd.extend(ffiles)

        tmp = deepcopy(sys.argv)
        sys.argv = f2pycmd
        sys.argv.append("--quiet")

        if os.access(self.source_directory, os.W_OK):
            echo = os.path.join(self.source_directory, "build.echo")
        else:
            echo = os.path.join(tempfile.gettempdir(), "build.echo")

        # f2py returns none if successful, it is an exception if not
        # successful
        try:
            built = not f2py()
        except BaseException as e:
            msg = re.sub(r"error: ", "", e.message)
            built = False
        except:
            msg = ("failed to build {0} with f2py, see {1}"
                   .format(self.libname, echo))
            built = False

        # restore sys.{argv, stdout, stderr}
        sys.argv = deepcopy(tmp)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        sys.stdout.flush()
        sys.stderr.flush()

        # remove nocallback_file, if it exists
        try:
            os.remove(self.nocallback_file)
        except OSError:
            pass

        # remove object files
        for f in self.source_files:
            try:
                os.remove(os.path.splitext(f)[0] + ".o")
            except OSError:
                pass
            continue

        if not built:
            pu.report_and_raise_error(msg, errno=2)

        # make sure the module is loadable
        try:
            py_mod, py_path = pu.get_module_name_and_path(self.libname)
            fp, pathname, description = imp.find_module(py_mod, py_path)
            build = imp.load_module(py_mod, fp, pathname, description)
            fp.close()
        except ImportError:
            new_name = os.path.join(cfg.AUX, self.libname)
            shutil.move(self.libname, new_name)
            pu.report_and_raise_error(
                "\n\tWARNING: {0} failed to import.  ".format(self.libname) +
                "\n\tTo diagnose, go to: {0}".format(new_name) +
                "\n\tand manually import the file in a python interpreter",
                errno=20)

        # copy the extension module file to the library directory
        shutil.move(self.libname,
                    os.path.join(self.payette_libdir, self.libname))

        try:
            os.remove(echo)
        except OSError:
            pass

        return 0

    def format_signature_file(self):

        """ format signature file from original sigature file """

        if cfg.F2PY["callback"]:
            return

        sigf_lines = open(self.signature_file, "r").readlines()

        in_callback = False
        user_routines = []
        lines = []
        for line in sigf_lines:

            if not line.split():
                continue

            if "end python module payette__user__routines" in line:
                in_callback = False
                continue

            if "use payette__user__routines" in line:
                continue

            if "python module payette__user__routines" in line:
                in_callback = True
                continue

            if in_callback:
                if line.split() and line.split()[0] == "subroutine":
                    sub = line.split(
                        "(")[0].strip()[len("subroutine"):].strip()
                    user_routines.append(sub)

                continue

            lines.append(line)

            continue

        self.signature_file = self.nocallback_file

        with open(self.signature_file, "w") as f:

            for line in lines:
                if [x for x in user_routines if x in line.split()]:
                    continue
                f.write(line)
                continue

        return
