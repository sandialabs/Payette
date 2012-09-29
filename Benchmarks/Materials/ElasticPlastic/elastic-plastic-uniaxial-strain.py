#!/usr/bin/env python
import os, sys

import Source.__config__ as cfg
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = True
        self.enabled = False

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.keywords = ["elastic_plastic", "verification", "fast",
                         "uniaxial", "builtin"]
        self.runcommand = ["payette","--no-writeprops",self.infile]
        self.material = "elastic_plastic"

        self.owner = 'Tim Fuller'
        self.date = 'February 26, 2012'
        self.description = ''' elastic_plastic material model, uniaxial strain '''


        if check:
            self.check_setup()

        pass

if __name__ == '__main__':
    test = Test()

    test.runFromTerminal(sys.argv[1:])
