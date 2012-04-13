from __future__ import print_function
import os,sys

from Payette_config import *
from Source.Payette_utils import BuildError
from Source.Materials.Payette_build_material import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        # domain switching model requires nlopt
        if not PC_NLOPT:
            raise BuildError("{0} environment variable not set, skipping {1}.\n"
                             .format("NLOPTLOC",self.libname),5)
        elif not os.path.isdir(PC_NLOPT):
            raise BuildError("{0} not found, skipping {1}\n."
                             .format(PC_NLOPT,self.libname),10)

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        pass

    def build_extension_module(self):

        self.incdirs.append(os.path.join(PC_NLOPT,"include"))
        self.libdirs.append(os.path.join(PC_NLOPT,"lib"))
        self.libs.extend(["nlopt","m"])

        # fortran source files
        if not PC_ALEGRANEVADA:
            raise BuildError("{0} environment variable not found, skipping {1}"
                             .format("PAYETTE_ALEGRA",self.libname),5)

        emechd = os.path.join(PC_ALEGRANEVADA,
                              "alegra/material_libs/electromech")
        if  not os.path.isdir(emechd):
            raise BuildError("{0} not found, skipping {1}"
                             .format(emechd,self.libname),10)

        srcs = ["slsqp_optmz.f", "domain_switching.f90"]
        self.source_files.extend([os.path.join(emechd,x) for x in srcs])

        self.build_extension_module_with_f2py()

        return 0

if __name__ == "__main__":
    sys.exit("Script must be called through buildPayette")
