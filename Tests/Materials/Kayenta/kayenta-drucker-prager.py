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
        self.keywords = ['fast', 'verification', 'kayenta', 'Drucker Prager']
        self.runcommand = ["runPayette","--no-restart",self.infile]

        self.owner = "Tim Fuller"
        self.date = "June 24, 2011"
        self.description = """ Non-associative Drucker-Prager test.

    The baseline file came from an equivalent MED simulation. But, becasue the
    MED takes the number of divisions of the simulation and interpolates the
    strain table at that many divisions, there can be a certain amount of error
    in the strain value at locations in the strain table where there are sharp
    corners. This test has some sharp corners and, therefore, some error in the
    MED simulation in the strain variable at those corners. Payette, on the other
    hand, only interpolates between legs and uses the exact strain at the
    beginning and end of leg, so there isn't error at those corners. That being,
    the Payette simulation was compared against the MED simulation and the
    Payette output then used as the gold file after verifying that the solutions
    are in agreement everywhere but at corners where the MED solution is not so
    good.
"""
        pass

if __name__ == "__main__":

    test = Test()

    test.runFromTerminal(sys.argv[1:])
