from __future__ import print_function
import os,sys

from Payette_config import *
from Source.Payette_utils import BuildError
from Source.Materials.Payette_build_material import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        # fortran source files
        if not PAYETTE_ALEGRANEVADA:
            raise BuildError("{0} environment variable not found, skipping {1}"
                             .format("PAYETTE_ALEGRA",self.libname),5)

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        pass

    def build_extension_module(self):

        emechd = os.path.join(PAYETTE_ALEGRANEVADA,
                              "alegra/material_libs/electromech")
        if  not os.path.isdir(emechd):
            raise BuildError("{0} not found, skipping {1}"
                             .format(emechd,self.libname),10)

        srcs = ["emech2.F","piezo_ceramic.F"]
        self.source_files.extend([os.path.join(emechd,x) for x in srcs])

        self.build_extension_module_with_f2py()

        return 0

if __name__ == '__main__':
    build()
