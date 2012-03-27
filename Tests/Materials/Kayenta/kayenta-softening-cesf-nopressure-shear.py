#!/usr/bin/env python

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
        self.keywords = ["kayenta","long","verification","cesf"]
        self.compare_method = self.compare_constant_strain_at_failure
        self.runcommand = ["runPayette","--no-restart",self.infile]
        self.failtol = 2.e-2
        self.difftol = 1.e-2

        self.owner = "Scot Swan"
        self.date = "February 03, 2012"
        self.description = """
   Perform verification of the Constant Equivalent Strain-to-Failure model.
   This test runs a simulation using parameters for Al 2024-T351 with
   proportional loading (constant ratio of stresses) defined by:
       Stress Triaxiality = 0.0
       Lode Parameter =  0.0
   which correlate to a state of shear that has no hydrostatic pressure.

   This test only looks at ACCSTRAIN verses COHER.

   For reference:
       Stress Triaxiality = sigma_m / sigma_eqv = I1/(3*ROOT3*ROOTJ2)
       Lode Parameter     = sin(3\lodeangle)
"""
        pass

if __name__ == "__main__":

    test = Test()

    test.runFromTerminal(sys.argv[1:])
