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
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.keywords = ['medium','validation','kayenta_ortho','joints']
        self.runcommand = ["runPayette","--no-writeprops","--no-restart",self.infile]

        self.owner = 'Tim Fuller'
        self.date = 'January 30, 2012'
        self.description = """
Stress driven unconfined compression of Salem Limestone using orthotropic
kayenta model.
"""
        pass

if __name__ == '__main__':
    test = Test()

    test.runFromTerminal(sys.argv[1:])
