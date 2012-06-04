#!/usr/bin/env python

from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = False # change to True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.keywords = ["a","list","of","keywords"]
        self.runcommand = ["runPayette","--no-writeprops","--no-restart",self.infile]

        self.owner = "Your Name"
        self.date = "Month Day, Year"
        self.description = """ A good description """


        if check:
            self.check_setup()

        pass

    def runTest(self):

        """ run the test """

        perform_calcs = self.run_command(self.runcommand)

        if perform_calcs != 0:
            return 2

        compare = self.compare_out_to_baseline_rms()

        return compare

if __name__ == '__main__':
    test = Test()

    test.runFromTerminal(sys.argv[1:])
