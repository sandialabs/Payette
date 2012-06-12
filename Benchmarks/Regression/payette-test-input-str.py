#!/usr/bin/env python

from Payette_config import *
from Source.Payette_test import PayetteTest
from Source.Payette_run import run_payette

class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.input_string = self.get_input_string()
        self.runcommand = ["--no-writeprops", "-v", "0",
                           "--input-str={0}".format(self.input_string)]
        self.material = "elastic"
        self.keywords = ["payette", "input_str", "regression",
                         "fast", "builtin"]

        self.owner = "Tim Fuller"
        self.date = "February 25, 2012"
        self.description = """ Test of input string capabilities """


        if check:
            self.check_setup()

        pass

    def get_input_string(self):

        input_string = """begin simulation payette-test-input-str
  begin material
    constitutive model elastic
    B0=11.634e9
    G0=10.018e9
  end material
  begin boundary
    estar = -1.
    kappa = 0.
    tstar = 1.
    ampl= .01
    begin legs
      0,     0.,   0, 222222, 0., 0., 0., 0., 0., 0.
      1,     1.,   1, 222222, 1., 0., 0., 0., 0., 0.
      2,     2.,   1, 222222, 0., 0., 0., 0., 0., 0.
    end legs
  end boundary
end simulation

begin simulation payette-test-input-str-1
  begin material
    constitutive model elastic
    B0=11.634e9
    G0=10.018e9
  end material
  begin boundary
    estar = -1.
    kappa = 0.
    tstar = 1.
    ampl= .01
    begin legs
      0,     0.,    0, 222222, 0., 0., 0., 0., 0., 0.
      1,     1.,   10, 222222, 1., 0., 0., 0., 0., 0.
      2,     2.,   10, 222222, 0., 0., 0., 0., 0., 0.
    end legs
  end boundary
end simulation
"""
        return input_string

    def runTest(self):

        """ run the test """

        # run the test directly through run_payette

        perform_calcs = run_payette(self.runcommand)

        if perform_calcs != 0:
            return self.failtoruncode

        compare = self.compare_method()

        return compare


if __name__ == "__main__":
    import time

    test = Test()

    print("RUNNING: {0}".format(test.name))
    run_test = test.runTest()

    if run_test == test.passcode:
        print("PASSED")
    elif run_test == test.diffcode:
        print("DIFF")
    elif run_test == test.failcode:
        print("FAIL")
        pass


