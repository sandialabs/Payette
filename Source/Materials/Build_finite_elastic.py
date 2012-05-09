from __future__ import print_function

import os,sys

from Payette_config import *
from Source.Payette_utils import BuildError
from Source.Payette_material_builder import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))
        self.fdir, self.fnam = fdir, fnam

        # initialize base class
        srcd = os.path.join(fdir, "Fortran")
        sigf = os.path.join(fdir, "Payette_finite_elastic.pyf")
        MaterialBuilder.__init__(
            self, name, libname, srcd, compiler_info, sigf=sigf)

        pass

    def build_extension_module(self):

        # fortran files
        srcs = ["finite_elastic.f90"]
        self.source_files = [os.path.join(self.source_directory, x)
                             for x in srcs]

        self.build_extension_module_with_f2py()

        return 0
