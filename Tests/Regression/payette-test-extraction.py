#!/usr/bin/env python

import subprocess

from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self):

        # initialize the base class
        PayetteTest.__init__(self)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.xout".format(self.name)
        self.baseline = "{0}.xgold".format(os.path.join(self.tdir,self.name))
        self.keywords = ["payette","regression","fast","extraction"]
        self.runcommand = ["extractPayette.py",self.infile,r"@time",
                           r"@strain11",r"@sig11",r"2*@strain11",r"2*@sig11",r"2*%2"]

        self.owner = "Tim Fuller"
        self.date = "February 28, 2012"
        self.description = """ Test of extractPayette.py """

        pass

    def runTest(self):
        """ run the test """

        run_command = self.run_command(self.runcommand,echof=self.outfile)

        if run_command != 0:
            return self.failtoruncode

        diff = self.diff_files(self.baseline,self.outfile)
        if diff:
            return self.failcode

        return self.passcode

if __name__ == "__main__":
    import time
    test = Test()
    if "--cleanup" in sys.argv:
        for ext in ["xout","diff","pyc","echo"]:
            try: os.remove("%s.%s"%(test.name,ext))
            except: pass
            continue
        pass

    else:
        t0 = time.time()
        print("%s RUNNING"%test.name)
        run_test = test.runTest()
        dtp = time.time()-t0
        if run_test == test.passcode:
            print("%s PASSED(%fs)"%(test.name,dtp))
        else:
            print("%s DATA EXTRACTION DIFFED(%fs)"%(test.name,dtp))
