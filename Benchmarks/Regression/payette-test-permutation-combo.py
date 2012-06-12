#!/usr/bin/env python

from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)
        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir, self.name))
        self.keywords = ["builtin", "medium", "regression", "elastic", "using",
                         "insert", "permutation", "combination"]
        self.runcommand = ["runPayette", "--no-writeprops",
                           self.infile]
        self.material = "elastic"
        self.aux_files = [os.path.join(self.tdir, "regression_tests.tbl"), ]

        self.owner = 'Tim Fuller'
        self.date = 'March 30, 2012'
        self.description = """Test of the "using dt strain" input method """

        if check:
            self.check_setup()

        pass

    def runTest(self):
        """ run the test """

        perform_calcs = self.run_command(self.runcommand)

        # if the test ran to completion, that is good enough
        if perform_calcs != 0:
            return self.failcode

        return self.passcode


if __name__ == '__main__':
    import time

    test = Test()

    test.runFromTerminal(sys.argv[1:])
