import os, sys

from Source.Payette_material_builder import MaterialBuilder
from Source.Payette_build import BuildError as BuildError

class Build(MaterialBuilder):

    def __init__(self, name, libname, libdir, compiler_info):

        fdir, fnam = os.path.split(os.path.realpath(__file__))

        # initialize base class
        srcd = os.path.join(fdir)
        MaterialBuilder.__init__(
            self, name, libname, srcd, libdir, compiler_info)

        pass
    def build_extension_module(self, *args, **kwargs):
        pass
