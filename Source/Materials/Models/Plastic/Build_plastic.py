import os, sys
from shutil import copyfile

import Source.__config__ as cfg
from Source.Payette_material_builder import MaterialBuilder
from Source.Payette_build import BuildError as BuildError

class Build(MaterialBuilder):

    def __init__(self, name, libname, libdir, compiler_info):

        fdir, fnam = os.path.split(os.path.realpath(__file__))

        # initialize base class
        srcd = fdir
        sigf = os.path.join(fdir, "Payette_plastic.pyf")
        MaterialBuilder.__init__(
            self, name, libname, srcd, libdir, compiler_info, sigf=sigf)

        pass

    def build_extension_module(self):

        # fortran files
        eos_file = os.getenv("EOS_MODULI_FILE")
        if eos_file is None:
            eos_file = os.path.join(self.source_directory, "plastic_eos.f90")

        tens_tools = os.path.join(cfg.FORTRAN, "tensor_toolkit.f90")
        self.source_files.extend([tens_tools, eos_file])
        f_srcs = ["plastic_mod.f90", "plastic.f90"]
        self.source_files.extend([os.path.join(self.source_directory, x)
                                  for x in f_srcs])

        try:
            retval = self.build_extension_module_with_f2py()

        except BuildError as error:
            sys.stderr.write("ERROR: {0}".format(error.message))
            retval = error.errno

        return retval
