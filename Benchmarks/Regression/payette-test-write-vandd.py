#!/usr/bin/env python

from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):


        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = ["{0:s}.dgold".format(os.path.join(self.tdir,self.name)),
                         "{0:s}.vgold".format(os.path.join(self.tdir,self.name))]
        self.tables = [self.name + ".dtable", self.name + ".vtable"]
        self.keywords = ["builtin", "payette", "regression",
                         "fast", "vtable", "dtable"]
        self.runcommand = ["runPayette", "--no-writeprops", "--no-restart",
                           "--write-vandd", self.infile]
        self.material = "elastic"

        self.owner = "Tim Fuller"
        self.date = "February 26, 2012"
        self.description = """ Test of velocity and displacement table creation """

        if check:
            self.check_setup()

        pass

    def runTest(self):

        """ run the test """
        perform_calcs = self.run_command(self.runcommand)

        if perform_calcs != 0:
            return self.failcode

        diff = self.diff_files(self.baseline,self.tables)

        if diff:
            return self.failcode

        return self.passcode

if __name__ == "__main__":
    import time

    test = Test()

    t0 = time.time()
    print("%s RUNNING"%test.name)
    run_test = test.run_command(test.runcommand)
    dtp = time.time()-t0

    if run_test == test.passcode:
        print("%s PASSED(%fs)"%(test.name,dtp))
    else:
        print("%s FAILED(%fs)"%(test.name,dtp))
