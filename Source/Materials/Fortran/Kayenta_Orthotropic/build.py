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

        self.kayenta_directives = ["-DKMM_ORTHOTROPIC"]
        self.pre_directives.extend(self.kayenta_directives)

        pass

    def build_extension_module(self):

        fdir,fnam = os.path.split(os.path.realpath(__file__))
        # get the Kayenta source directory
        if not Payette_Kayenta:
            raise BuildError("{0} environment variable not found, {1} not built"
                             .format("PAYETTE_KAYENTA",self.libname),5)
        elif not os.path.isdir(Payette_Kayenta):
            raise BuildError("{0} not found, {1} not built"
                             .format(Payette_Kayenta,self.libname),10)
        else:
            self.source_files = self.get_kayenta_source_files()
            pass

        self.build_extension_module_with_f2py()

        # remove cruft
        for srcf in self.source_files:
            try: os.remove(srcf)
            except: pass
            continue
        try: os.remove(os.path.join(fdir,"host_defines.h"))
        except: pass

        return 0

    def get_kayenta_source_files(self):

        # use kayenta release script to get the source file
        fdir = os.path.dirname(os.path.realpath(__file__))
        releasef = os.path.join(Payette_Kayenta,"release.py")
        py_mod = os.path.splitext(os.path.basename(releasef))[0]
        py_path = os.path.dirname(releasef)
        sys.path.append(py_path)
        fp, pathname, description = imp.find_module(py_mod,[py_path])
        release = imp.load_module(py_mod,fp,pathname,description)
        sys.path.pop()
        fp.close()
        kmmcmd = ["-t","payette","-b",fdir] + self.kayenta_directives
        retcode, kmmsrcf = release.main(kmmcmd)
        if retcode != 0:
            raise BuildError("kayenta release script failed",10)

        # remove extension module files if they exist
        for d in [self.source_directory,Payette_Materials_Library]:
            try: os.remove(os.path.join(d,self.libname))
            except: pass
            continue

        return kmmsrcf


if __name__ == "__main__":
    sys.exit("Script must be called through buildPayette")
