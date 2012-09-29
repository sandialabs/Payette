import os, sys

import Source.__config__ as cfg
from Source.Payette_material_builder import MaterialBuilder
from Source.Payette_build import BuildError as BuildError

class Build(MaterialBuilder):

    def __init__(self, name, libname, libdir, compiler_info):

        fdir, fnam = os.path.split(os.path.realpath(__file__))

        # initialize base class
        srcd = os.path.join(fdir)
        sigf = os.path.join(fdir, "Payette_elastic.pyf")
        MaterialBuilder.__init__(
            self, name, libname, srcd, libdir, compiler_info, sigf=sigf)

        pass

    def build_extension_module(self):

        # fortran files
        srcs = ["elastic.f90"]
        self.source_files = [os.path.join(self.source_directory, x)
                             for x in srcs]
        self.source_files.append(
            os.path.join(cfg.FORTRAN, "tensor_toolkit.f90"))

        try:
            retval = self.build_extension_module_with_f2py()
        except BuildError as error:
            sys.stderr.write("ERROR: {0}".format(error.message))
            retval = error.errno

        return retval
