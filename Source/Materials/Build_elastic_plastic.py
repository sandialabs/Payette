from __future__ import print_function
import os,sys

from Payette_config import *
from Source.Payette_utils import BuildError
from Source.Payette_material_builder import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        srcd = os.path.join(fdir, "Fortran")
        sigf = os.path.join(fdir, "Payette_elastic_plastic.pyf")
        # initialize base class
        MaterialBuilder.__init__(
            self, name, libname, srcd, compiler_info, sigf=sigf)

        pass

    def build_extension_module(self):

        # fortran files
        srcs = ["elastic_plastic.F"]
        self.source_files = [os.path.join(self.source_directory, x)
                             for x in srcs]

        self.build_extension_module_with_f2py()

        return 0

if __name__ == '__main__':
    sys.exit("Script must be called through buildPayette")
