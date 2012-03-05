from __future__ import print_function
import os,sys,shutil,imp
import subprocess as sbp
from distutils import sysconfig
from copy import deepcopy
from numpy.f2py import main as f2py

from Toolset.Payette_config import *
from Toolset.buildPayette import BuildError,get_module_name_and_path

class MaterialBuilder():

    def __init__(self, name, libname, srcd, compiler_info, sigf=None):

        self.name = name
        self.source_directory = srcd
        self.source_files = []
        self.libname = libname

        # signature file and migutils
        if not sigf and Payette_F2Py_Callback:
            self.signature_file = os.path.join(srcd, self.name + ".python-2.pyf")
            self.pre_directives = ["-UNOPYCALLBACK"]
        elif not sigf:
            self.signature_file = os.path.join(srcd, self.name + ".python-3.pyf")
            self.pre_directives = ["-DNOPYCALLBACK"]
        else:
            self.signature_file = sigf
            self.pre_directives = []
            pass

        if not os.path.isfile(self.signature_file):
            raise BuildError("{0} signature file not found".format(self.name),40)

        self.incdirs = [ ".", "{0}".format(self.source_directory),
                         "{0}".format(Payette_Materials_Fortran_Includes) ]
        self.libdirs = []
        self.libs = []

        # f2py opts
        f2py_opts = compiler_info["f2py"]
        self.f2pyopts = [f2py_opts["compiler"],"-c"] + f2py_opts["options"]
        pass

    def build_extension_module(self):
        raise BuildError("build script must provide this function",1)

    def build_extension_module_with_f2py(self):
        fcn = "build_extension_module_with_f2py"

        if not self.source_files:
            BuildError("no source files sent to {0}".format(fcn),1)

        elif not isinstance(self.source_files,list):
            self.source_files = [self.source_files]

        else: pass

        for srcf in self.source_files:
            if not os.path.isfile(srcf):
                BuildError("{0}: source file {1} not found".format(fcn,srcf),1)
            continue

        # remove extension module files if they exist
        for d in [self.source_directory, Payette_Materials_Library]:
            try: os.remove(os.path.join(d,self.libname))
            except: pass
            continue

        # f2py pulls its arguments from sys.argv. Here, we build sys.argv to what
        # f2py expects. Later sys.argv will be restored.
        ffiles = self.source_files + [ Payette_MIG_Utils ]
        incsearch = ["-I{0}".format(x) for x in self.incdirs]
        libsearch = ["-L{0}".format(x) for x in self.libdirs]
        libs = ["-l{0}".format(x) for x in self.libs]
        f2pycmd = self.f2pyopts + incsearch + libsearch + libs
        f2pycmd.extend(["-m",self.name,self.signature_file])
        f2pycmd.extend(self.pre_directives)
        f2pycmd.extend(ffiles)

        tmp = deepcopy(sys.argv)
        sys.argv = f2pycmd

        try:
            echo = os.path.join(self.source_directory,"build.echo")
            with open(echo,"w") as sys.stdout:
                with open(echo,"a") as sys.stderr:
                    # f2py returns none if successful, it is an exception if not
                    # successful
                    built = not f2py()
                    pass
                pass
        except:
            built = False
            pass

        # restore sys.{argv,stdout,stderr}
        sys.argv = deepcopy(tmp)
        sys.stdout, sys.stderr  = sys.__stdout__, sys.__stderr__

        if not built:
            raise BuildError("failed to build {0} with f2py, see {1}"
                             .format(self.libname,echo),2)

        # make sure the module is loadable
        try:
            py_mod, py_path = get_module_name_and_path(self.libname)
            fp, pathname, description = imp.find_module(py_mod,py_path)
            build = imp.load_module(py_mod,fp,pathname,description)
            fp.close()
        except ImportError, e:
            raise BuildError("{0}".format(str(e)),35)


        # copy the extension module file to the library directory
        shutil.move(self.libname,
                    os.path.join(Payette_Materials_Library,self.libname))

        try: os.remove(echo)
        except: pass

        return 0

