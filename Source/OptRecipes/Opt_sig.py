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

"""Provides a recipe for computing the maximum error between the
expected stress and the simulated

"""
import os
import sys
import numpy as np
import Source.Payette_utils as pu
import Source.Payette_extract as pe


class ObjectiveFunction(object):

    def __init__(self, *args):
        """With this objective function, the maximum root mean squared error
        between SIG11, SIG22, and SIG33 from the simulation output and the gold
        result is returned as the error.

        Parameters
        ----------
        args[0] : str
            The gold file. Must contain columns labeled SIG11, SIG22, SIG33.

        """

        self.minvars = ["@sig11", "@sig22", "@sig33"]
        gold_f = args[0]
        if gold_f is None:
            pu.report_and_raise_error("No gold file given for Opt_sig.py")

        elif not os.path.isfile(gold_f):
            pu.report_and_raise_error("{0} not found".format(gold_f))

        # extract only what we want from the gold and output files
        exargs = [gold_f] + self.minvars
        self.gold_data = np.array(pe.extract(exargs, silent=True))
        pass

    def evaluate(self, *args):
        """Evaluates the error between the simulation output and the "gold" answer

        Parameters
        ----------
        args : tuple
            args[0] : output file from simulation

        Returns
        -------
        error : float
            The error between the output and the "gold" answer

        """
        out_f = args[0]
        if not os.path.isfile(out_f):
            pu.report_and_raise_error("{0} not found".format(out_f))

        # extract only what we want from the gold and output files
        exargs = [out_f] + self.minvars
        out_data = np.array(pe.extract(exargs, silent=True))

        # do the comparison
        anrmsd = []
        for idx in range(3):
            rmsd = np.sqrt(
                np.mean((self.gold_data[:, idx] - out_data[:, idx]) ** 2))
            dnom = abs(np.amax(out_data[:, idx]) - np.amin(out_data[:, idx]))
            nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
            anrmsd.append(nrmsd)
            continue

        error = np.amax(np.abs(np.array(anrmsd)))
        return error
