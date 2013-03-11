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
import time

import Source.__config__ as cfg
from Source.Payette_test import PayetteTest


class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir, self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir, self.name))
        self.restartfile = self.name + ".prf"
        self.runcommand = ["payette", "--no-writeprops", "--write-restart",
                           "--test-restart", self.infile]
        self.restartcommand = ["payette", "--no-writeprops", "--write-restart",
                               "--test-restart", self.restartfile]
        self.material = "elastic"

        self.keywords = ["builtin", "payette", "restart", "regression", "fast"]
        self.owner = "Tim Fuller"
        self.date = "February 25, 2012"
        self.description = """ Test of restart capabilities """

        if check:
            self.check_setup()

        pass

    def run_test(self):
        """ run the test """
        d = os.getcwd()
        os.chdir(self.results_directory())
        t0 = time.time()
        perform_calcs = self.run_command(self.runcommand)
        if perform_calcs != 76:
            retcode = self.failcode
        else:
            # now run the restart file
            perform_calcs = self.run_command(self.restartcommand)
            if perform_calcs != 0:
                retcode = self.failcode
            else:
                # now check the output
                retcode = self.compare_method()
        tc = time.time()
        self.completion_time(tc - t0)
        self.retcode = retcode
        self.status = self.get_status()
        os.chdir(d)
        return

    def compare_method(self):
        return self.compare_out_to_baseline_rms()


if __name__ == "__main__":

    import time

    test = Test()

    t0 = time.time()
    print("{0} RUNNING".format(test.name))
    run_test = test.run_command(test.runcommand)
    dtp = time.time() - t0
    if run_test != 76:
        print("{0} FAILED TO RUN TO COMPLETION ON FIRST LEG".format(test.name))
        sys.exit()
        pass
    # now run the restart file
    run_test = test.run_command(test.restartcommand)
    t1 = time.time()
    dta = time.time() - t1
    if run_test == test.passcode:
        print("%s PASSED(%fs)".format(test.name, dtp + dta))
    elif run_test == test.diffcode:
        print("{0} DIFFED({1}s)".format(test.name, dtp + dta))
    else:
        print("{0} FAILED({1}s)".format(test.name, dtp + dta))
