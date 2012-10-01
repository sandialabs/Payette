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


import os, sys
import subprocess

import Source.__config__ as cfg
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.gold".format(os.path.join(self.tdir, self.name))
        self.outfile = "{0}.xout".format(self.name)
        self.baseline = "{0}.xgold".format(os.path.join(self.tdir, self.name))
        self.keywords = ["payette", "regression", "fast", "extraction", "builtin"]
        self.runcommand = ["extractPayette", self.infile, r"@time",
                           r"@strain11", r"@sig11", r"2*@strain11",
                           r"2*@sig11", r"2*%2"]

        self.owner = "Tim Fuller"
        self.date = "February 28, 2012"
        self.description = """ Test of extractPayette.py """


        if check:
            self.check_setup()

        pass

    def runTest(self):
        """ run the test """

        run_command = self.run_command(self.runcommand, echof=self.outfile)

        if run_command != 0:
            return self.failtoruncode

        diff = self.diff_files(self.baseline, self.outfile)
        if diff:
            return self.failcode

        return self.passcode

if __name__ == "__main__":
    import time
    test = Test()
    if "--cleanup" in sys.argv:
        for ext in ["xout", "diff", "pyc", "echo"]:
            try: os.remove("%s.%s"%(test.name, ext))
            except: pass
            continue
        pass

    else:
        t0 = time.time()
        print("%s RUNNING"%test.name)
        run_test = test.runTest()
        dtp = time.time()-t0
        if run_test == test.passcode:
            print("%s PASSED(%fs)"%(test.name, dtp))
        else:
            print("%s DATA EXTRACTION DIFFED(%fs)"%(test.name, dtp))
