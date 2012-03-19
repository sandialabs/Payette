from __future__ import print_function
import os,sys,shutil
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
        self.fdir, self.fnam = fdir, fnam

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        pass

    def build_extension_module(self):

        # fortran files
        self.source_files = [osp.join(self.fdir,x) for x in os.listdir(self.fdir)
                             if x.endswith(".F")]

        self.build_extension_module_with_f2py()

        return 0

if __name__ == '__main__':
    build()
