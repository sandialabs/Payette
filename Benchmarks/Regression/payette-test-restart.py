#!/usr/bin/env python

from config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.restartfile = self.name + ".prf"
        self.runcommand = ["payette", "--no-writeprops", "--write-restart",
                           "--test-restart", self.infile]
        self.restartcommand = ["payette", "--no-writeprops", "--write-restart",
                               "--test-restart", self.restartfile]
        self.material = "elastic"

        self.keywords = ["builtin", "payette", "restart", "regression", "fast"]
        self.compare_method = self.compare_out_to_baseline_rms
        self.owner = "Tim Fuller"
        self.date = "February 25, 2012"
        self.description = """ Test of restart capabilities """

        if check:
            self.check_setup()

        pass

    def runTest(self):
        """ run the test """
        perform_calcs = self.run_command(self.runcommand)
        if perform_calcs != 76:
            return self.failcode

        # now run the restart file
        perform_calcs = self.run_command(self.restartcommand)
        if perform_calcs != 0:
            return self.failcode

        # now check the output
        compare = self.compare_method()

        return compare


if __name__ == "__main__":

    import time

    test = Test()

    t0 = time.time()
    print("{0} RUNNING".format(test.name))
    run_test = test.run_command(test.runcommand)
    dtp = time.time()-t0
    if run_test != 76:
        print("{0} FAILED TO RUN TO COMPLETION ON FIRST LEG".format(test.name))
        sys.exit()
        pass
    # now run the restart file
    run_test = test.run_command(test.restartcommand)
    t1 = time.time()
    dta = time.time()-t1
    if run_test == test.passcode:
        print("%s PASSED(%fs)".format(test.name,dtp+dta))
    elif run_test == test.diffcode:
        print("{0} DIFFED({1}s)".format(test.name,dtp+dta))
    else:
        print("{0} FAILED({1}s)".format(test.name,dtp+dta))
