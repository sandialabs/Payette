from __future__ import print_function
import os,sys,shutil
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
        self.fdir = fdir

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        pass

    def build_extension_module(self):

        # fortran files
        self.source_files = [osp.join(self.fdir,x) for x in os.listdir(self.fdir)
                             if x.endswith(".F")]
        kerleyd = Payette_Kayenta
        if kerleyd:
            kerley = osp.join(kerleyd,"Kerley_eos.F")
            if os.path.isfile(kerley):
                self.pre_directives.append("-DKERLEY_EOS_RTNS")
                self.source_files.append(kerley)
                pass
            pass

        self.build_extension_module_with_f2py()

        return 0

if __name__ == '__main__':
    sys.exit("Script must be called through buildPayette")
