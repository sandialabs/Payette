import os, sys

from Source.Payette_material_builder import MaterialBuilder
from Source.Payette_build import BuildError as BuildError

class Build(MaterialBuilder):

    def __init__(self, name, libname, libdir, compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        srcd = fdir
        sigf = os.path.join(fdir, "Payette_elastic_plastic.pyf")
        # initialize base class
        MaterialBuilder.__init__(
            self, name, libname, srcd, libdir, compiler_info, sigf=sigf)

        pass

    def build_extension_module(self):

        # fortran files
        srcs = ["elastic_plastic.F"]
        self.source_files = [os.path.join(self.source_directory, x)
                             for x in srcs]

        try:
            retval = self.build_extension_module_with_f2py()
        except BuildError as error:
            sys.stderr.write("ERROR: {0}".format(error.message))
            retval = error.errno

        return retval
