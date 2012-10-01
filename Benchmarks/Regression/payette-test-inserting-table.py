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
        self.keywords = ["fast", "verification", "elastic", "using", "insert",
                         "builtin"]
        self.runcommand = ["payette", "--no-writeprops",
                           self.infile]
        self.material = "elastic"
        self.aux_files = [os.path.join(self.tdir, "regression_tests.tbl"), ]

        self.owner = 'Tim Fuller'
        self.date = 'March 30, 2012'
        self.description = """Test of the "using dt strain" input method """

        if check:
            self.check_setup()

        pass

if __name__ == '__main__':
    import time

    test = Test()

    test.runFromTerminal(sys.argv[1:])
