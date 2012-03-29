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
        self.keywords = ["domain_switching_ceramic", "dsc", "prototype",
                         "fast", "piezo ceramic", "electromech"]
        self.runcommand = ["runPayette", "--no-restart",
                           "--no-writeprops", self.infile]

        self.owner = "Tim Fuller"
        self.date = "March 11, 2012"
        self.description = """ Uniaxial strain of multi domain ceramic """
        pass

if __name__ == "__main__":
    import time

    test = Test()

    test.runFromTerminal(sys.argv[1:])
