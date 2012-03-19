from __future__ import print_function
import os,sys,shutil,imp
import subprocess as sbp
import os.path as osp
from distutils import sysconfig
from copy import deepcopy
from numpy.f2py import main as f2py

from Payette_config import *
from Source.Payette_utils import BuildError
from Source.Materials.Payette_build_material import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        pass

    def build_extension_module(self):

        # domain switching model requires nlopt
        if not Payette_nlopt:
            raise BuildError("{0} environment variable not set, skipping {1}.\n"
                             .format("NLOPTLOC",self.libname),5)
        elif not os.path.isdir(Payette_nlopt):
            raise BuildError("{0} not found, skipping {1}\n."
                             .format(Payette_nlopt,self.libname),10)

        self.incdirs.append(os.path.join(Payette_nlopt,"include"))
        self.libdirs.append(os.path.join(Payette_nlopt,"lib"))
        self.libs.extend(["nlopt","m"])

        # fortran source files
        if not Payette_AlegraNevada:
            raise BuildError("{0} environment variable not found, skipping {1}"
                             .format("PAYETTE_ALEGRA",self.libname),5)

        emechd = os.path.join(Payette_AlegraNevada,
                              "alegra/material_libs/electromech")
        if  not os.path.isdir(emechd):
            raise BuildError("{0} not found, skipping {1}"
                             .format(emechd,self.libname),10)

        srcs = ["domain_switching_modules.f90","emech7.f90","domain_switching.f90"]
        self.source_files.extend([os.path.join(emechd,x) for x in srcs])

        self.build_extension_module_with_f2py()

        return 0

if __name__ == "__main__":
    sys.exit("Script must be called through buildPayette")
