#!/usr/bin/env python
import os,sys

from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):


        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.keywords = ['kayenta', 'prototype', 'fast', 'rate dependent',
                         'hardening']
        self.runcommand = ["runPayette","--no-restart",self.infile]

        self.owner = "Tim Fuller"
        self.date = "June 28, 2011"
        self.description = """
Rate dependence with hardening.

    The gold is from an equivalent MED simulation run a year or so ago.

                                  **** NOTE ****

    This test does not pass. The HK variable is off between the out and gold.
"""

        if check:
            self.check_setup()

        pass

if __name__ == "__main__":

    test = Test()

    test.runFromTerminal(sys.argv[1:])
