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

        self.compare_method = self.compare_out_to_baseline_rms_general
        self.rms_sets = [ ["I1","ROOTJ2"] ]
        self.difftol = 1.0e-6
        self.failtol = 1.0e-5

        self.keywords = ['verification', 'kayenta', 'fast', 'ldp']
        self.runcommand = ["runPayette","--no-restart",self.infile]

        self.owner = "Scot Swan"
        self.date = "April 16, 2012"
        self.description = """
This simulation has the following characteristics:
* The shear modulus was calculated from the bulk modulus of aluminum
  and a poisson's ratio of -0.1 .
* The simulation is uniaxial extension.
* The CTPS surface is put at (nearly) infinity.
* The limit surface has been set to mimic the CTPS surface when
  CTPS = 1.0e8 (these parameters are only good for uniaxial extension).

In ROOTJ2 vs I1 space, this simulation will approach the limit surface and
the return algorithm should make the stress state move away from the I1 tensile
vertex.
 """
        pass

if __name__ == "__main__":

    test = Test()

    test.runFromTerminal(sys.argv[1:])
