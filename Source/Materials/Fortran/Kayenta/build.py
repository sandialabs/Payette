from __future__ import print_function
import os,sys,imp

from Payette_config import *
from Source.Payette_utils import BuildError
from Source.Materials.Payette_build_material import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        # check if Kayenta source directory is found
        if not PC_KAYENTA:
            raise BuildError("{0} environment variable not found, {1} not built"
                             .format("PC_KAYENTA",libname),5)
        elif not os.path.isdir(PC_KAYENTA):
            raise BuildError("{0} not found, {1} not built"
                             .format(PC_KAYENTA,libname),10)

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        self.kayenta_directives = ["-DTENSILE_CONTROL"]
        self.pre_directives.extend(self.kayenta_directives)

        pass

    def build_extension_module(self):

        self.source_files = self.get_kayenta_source_files()

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
        releasef = os.path.join(PC_KAYENTA,"release.py")
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

        return kmmsrcf


if __name__ == "__main__":
    sys.exit("Script must be called through buildPayette")
