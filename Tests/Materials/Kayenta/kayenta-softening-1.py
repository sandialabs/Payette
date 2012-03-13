#!/usr/bin/env python
import os,sys

from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self):

        # initialize the base class
        PayetteTest.__init__(self)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.keywords = ['fast', 'verification', 'kayenta', 'softening', 'med']
        self.runcommand = ["runPayette","--no-restart",self.infile]

        self.owner = "M. Scot Swan"
        self.date = "June 20, 2011"
        self.description = """
Uniaxial strain compression with damage with the following steps:

    Step 1 (t=0.-10.): prescribed uniaxial strain to eps_11 = 0.2

    Results from the output are compared against results from an equivalent
    simulation performed in the MED driver.
 """
        pass

if __name__ == "__main__":

    test = Test()

    test.runFromTerminal(sys.argv[1:])
