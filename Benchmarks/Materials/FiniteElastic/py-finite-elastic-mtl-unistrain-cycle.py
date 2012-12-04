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


class Test(PayetteTest):

    def __init__(self, check=True):
        super(Test, self).__init__(check)

        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir, self.name))
        self.outfile = "{0}.out".format(self.name)
        self.baseline = "{0}.gold".format(os.path.join(self.tdir, self.name))
        self.keywords = ["medium", "verification", "finite_elastic",
                         "uniaxial_strain", "python_model", "builtin"]
        self.runcommand = ["payette", "--no-writeprops", self.infile]
        self.material = "finite_elastic"

        self.owner = 'Tim Fuller'
        self.date = 'June 1, 2011'
        self.description = """
    In this test, a finite elastic material is cycled through a uniaxial
    strain deformation path in the following steps:

    Step 1 (t=0.-1.): prescribed uniaxial strain to eps_11 = 0.1
    Step 2 (t=1.-2.): prescribed uniaxial strain to eps_11 = 0.0
    Step 3 (t=2.-3.): prescribed stress resulting in same strain
                      path as Step 1
    Step 4 (t=3.-4.): prescribed stress resulting in same strain
                      path as Step 2
    Step 5 (t=4.-5.): prescribed strain rate resulting in same strain
                      path as Step 1
    Step 6 (t=5.-6.): prescribed strain rate resulting in same strain
                      path as Step 2
    Step 7 (t=6.-7.): prescribed stress rate resulting in same strain
                      path as Step 1
    Step 8 (t=7.-8.): prescribed stress rate resulting in same strain
                      path as Step 2
    Step 9 (t=8.-9.): prescribed def gradient resulting in same strain
                      path as Step 1
    Step 10 (t=9.-10.): prescribed def gradient resulting in same strain
                        path as Step 2
"""

        if check:
            self.check_setup()

        pass


if __name__ == '__main__':
    import time

    test = Test()

    test.runFromTerminal(sys.argv[1:])
