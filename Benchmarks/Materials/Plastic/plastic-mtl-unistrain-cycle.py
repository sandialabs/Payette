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
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.keywords = ["medium", "verification", "plastic", "uniaxial strain",
                         "builtin"]
        self.runcommand = ["runPayette", "--no-writeprops",
                           "--no-restart", self.infile]
        self.material = "plastic"

        self.owner = 'Tim Fuller'
        self.date = 'June 1, 2011'
        self.description = """
    In this test, an plastic material is cycled through a uniaxial
    strain deformation path in the following steps:

    Step 1 (t=0.-1.): prescribed uniaxial strain to eps_11 = 0.1
    Step 2 (t=1.-2.): prescribed uniaxial strain to eps_11 = 0.0
    Step 3 (t=2.-3.): prescribed stress resulting in same strain
                      path as Step 1
    Step 4 (t=3.-4.): prescribed stress resulting in same strain
                      path as Step 2
    Step 5 (t=4.-5.): prescribed strain rate resulting in same strain
                      path as Step 1
    Step 6 (t=5.-6.): prescribed strain rate resulting in same strain
                      path as Step 2
    Step 7 (t=6.-7.): prescribed stress rate resulting in same strain
                      path as Step 1
    Step 8 (t=7.-8.): prescribed stress rate resulting in same strain
                      path as Step 2
    Step 9 (t=8.-9.): prescribed def gradient resulting in same strain
                      path as Step 1
    Step 10 (t=9.-10.): prescribed def gradient resulting in same strain
                        path as Step 2
"""


        if check:
            self.check_setup()

        pass

if __name__ == '__main__':
    import time

    test = Test()

    test.runFromTerminal(sys.argv[1:])
