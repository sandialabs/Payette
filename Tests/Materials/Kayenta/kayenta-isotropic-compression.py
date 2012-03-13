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
        self.keywords = ['kayenta', 'medium', 'isotropic compression',
                         'verification', 'hardening']
        self.runcommand = ["runPayette","--no-restart",self.infile]

        self.owner = "Tim Fuller"
        self.date = "June 25, 2011"
        self.description = """
Test isotropic compression of kayenta material baseline file is from equivalent
MED simulation

                             **********NOTE************
      This test fails!  The offending variable is KAPPA.  When the problem
      with Kayenta is resolved that is causing KAPPA to fail, the Payette
      output can be copied to the baseline file.
 """
        pass

if __name__ == "__main__":

    test = Test()

    test.runFromTerminal(sys.argv[1:])
