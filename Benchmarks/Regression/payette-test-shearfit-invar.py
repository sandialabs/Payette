#!/usr/bin/env python

from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir, self.name))
        self.outfile = "{0}.opt/{0}.opt".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir, self.name))
        self.keywords = ["builtin","fast", "regression", "optimization", "shearfit"]
        self.runcommand = ["runPayette", "--no-writeprops", "--no-restart",
                           self.infile]
        self.aux_files = [os.path.join(self.tdir, "shearfit_invar.tbl"), ]

        self.compare_method = self.diff_files

        self.owner = 'Tim Fuller'
        self.date = 'March 30, 2012'
        self.description = """Test of the shearfit optimization """

        if check:
            self.check_setup()

        pass


if __name__ == '__main__':
    import time

    test = Test()

    test.runFromTerminal(sys.argv[1:])
