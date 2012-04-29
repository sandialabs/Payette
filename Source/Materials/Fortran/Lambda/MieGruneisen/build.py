from __future__ import print_function
import os,sys,imp

from Payette_config import *
from Source.Payette_utils import BuildError
from Source.Payette_material_builder import MaterialBuilder

class Build(MaterialBuilder):

    def __init__(self,name,libname,compiler_info):

        fdir,fnam = os.path.split(os.path.realpath(__file__))

        # check if Kayenta source directory is found
        if not PC_LAMBDA_MGRUN:
            raise BuildError("{0} environment variable not found, {1} not built"
                             .format("LAMBDA_MGRUN",libname),5)
        elif not os.path.isdir(PC_LAMBDA_MGRUN):
            raise BuildError("{0} not found, {1} not built"
                             .format(PC_LAMBDA_MGRUN,libname),10)

        # initialize base class
        MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

        pass

    def build_extension_module(self):

        self.source_files = self.get_source_files()

        self.build_extension_module_with_f2py()

        # remove cruft
        for srcf in self.source_files:
            try: os.remove(srcf)
            except: pass
            continue

        return 0


    def get_source_files(self):

        # use kayenta release script to get the source file
        fdir = os.path.dirname(os.path.realpath(__file__))
        source_d = PC_LAMBDA_MGRUN
        full_source = os.path.join(fdir, "MieGruneisen.F")

        source_files = ["eosmgi.F", "eosmgj.F", "eosmgK.F", "eosmgp.F",
                        "eosmgr.F", "eosmgv.F", "eosmgx.F", "eosmgy.F"]

        source_files = [os.path.join(source_d, x) for x in source_files]

        for source in source_files:
            if not os.path.isfile(source):
                raise BuildError("Lambda MGRUN file does not exist: {0}".format(source),10)


        CATFILE = open(full_source,"w")
        for source in source_files:
            CATFILE.write(open(source,"r").read())
            CATFILE.write("\n")
        CATFILE.close()

        return full_source


if __name__ == "__main__":
    sys.exit("Script must be called through buildPayette")
