#!/usr/bin/env python

from Payette_config import *
from Source.Payette_test import PayetteTest
from Source.Payette_main import runPayette

class Test(PayetteTest):

    def __init__(self):

        # initialize the base class
        PayetteTest.__init__(self)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir,self.name))
        self.input_string = self.get_input_string()
        self.runcommand = ["--no-restart","--no-writeprops","-v","0",
                           "--input-str={0}".format(self.input_string)]
        self.keywords = ["payette","input_str","regression","fast"]

        self.owner = "Tim Fuller"
        self.date = "February 25, 2012"
        self.description = """ Test of input string capabilities """

        pass

    def get_input_string(self):

        mtl_props= """
AN=1.
B0=11.634e9
G0=10.018e9
G1=5
A1=8.27e6
A4=0.23
R0=3500
T0=298
GP=1.0
S1=1
CV=800
TM=1e99
IDK=1
IDG=1
"""

        input_string = """begin simulation payette-test-input-str
  begin material
    constitutive model elastic_plastic
    {0}
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
    constitutive model elastic_plastic
    {0}
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
""".format(mtl_props)
        return input_string

    def runTest(self):

        """ run the test """

        # run the test directly through runPayette

        perform_calcs = runPayette(len(self.runcommand),self.runcommand)

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


