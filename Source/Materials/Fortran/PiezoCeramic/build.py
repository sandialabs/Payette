from __future__ import print_function
import os,sys,shutil,imp
import subprocess as sbp
import os.path as osp
from distutils import sysconfig
from copy import deepcopy
from numpy.f2py import main as f2py

from Toolset.Payette_config import *
from Toolset.buildPayette import BuildError
from Source.Materials.Payette_build_material import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        pass

    def build_extension_module(self):

        # fortran source files
        if not Payette_AlegraNevada:
            raise BuildError("{0} environment variable not found, skipping {1}"
                             .format("PAYETTE_ALEGRA",self.libname),5)

        emechd = os.path.join(Payette_AlegraNevada,
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
