#!/usr/bin/env python
# Copyright (2011) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
# rights in this software.

# The MIT License

# Copyright (c) Sandia Corporation

# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os
import sys

import Source.__config__ as cfg
from Source.Payette_test import PayetteTest
from Source.Payette_run import run_payette


class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir, self.name))
        self.input_string = self.get_input_string()
        self.siminp = self.input_string.split("\n")
        self.material = "elastic"
        self.keywords = [
            "payette", "input_str", "regression", "fast", "builtin",
            "insert_block"]

        self.owner = "Tim Fuller"
        self.date = "May 21, 2012"
        self.description = """ Test of 'insert block' capabilities """

        if check:
            self.check_setup()

        pass

    def get_input_string(self):

        input_string = """\
begin simulation payette-test-use-block
  nowriteprops
  insert boundary_001
  begin material
    constitutive model elastic
    insert material_001
  end material
end simulation

begin simulation payette-test-use-block-1
  nowriteprops
  insert boundary_001
  begin material
    constitutive model elastic
    insert material_001
  end material
end simulation

begin material_001
  # some fake parameters
  B0=11.634e9
  G0=10.018e9
end material_001

begin boundary_001
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
end boundary_001
"""
        return input_string

    def runTest(self):

        """ run the test """

        # run the test directly through run_payette

        perform_calcs = run_payette(siminp=self.siminp, verbosity=0, disp=0)

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
