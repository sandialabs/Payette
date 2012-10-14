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

"""Provides a recipe for computing the maximum error between the values of
variables from a known Payette result and a simulated result. Using "versus"
time computes error of slope.

"""
import os, sys
import numpy as np
import Source.Payette_utils as pu
import Source.Payette_extract as pe


class ObjectiveFunction(object):

    def __init__(self, *args):
        """Initialize data needed to compute the error

        """

        # Do operations on the gold file here so that they are only done once
        self.gold_f = args[0]
        if self.gold_f is None:
            pu.report_and_raise_error("No gold file given for Opt_legacy")

        elif not os.path.isfile(self.gold_f):
            pu.report_and_raise_error("{0} not found".format(self.gold_f))

        # get vars to minimize
        self.abscissa = args[1]
        self.minvars = args[2]
        self.nc = len(self.minvars)

        if self.abscissa is not None:
            self.minvars = [self.abscissa] + self.minvars

        # extract only what we want from the gold and output files
        exargs = [self.gold_f] + self.minvars
        self.gold_data = np.array(pe.extract(exargs, silent=True))

        pass

    def evaluate(self, *args):
        """Evaluates the error between the simulation output and the "gold"
        answer

        Parameters
        ----------
        args : tuple
            args[0] : output file from simulation

        Returns
        -------
        error : float
            The error between the output and the "gold" answer

        Notes
        -----
        With this objective function, the maximum root mean squared error
        between SIG11, SIG22, and SIG33 from the simulation output and the
        gold result is returned as the error.

        """

        # extract only what we want from the gold and output files
        out_f = args[0]
        exargs = [out_f] + self.minvars
        out_data = np.array(pe.extract(exargs, silent=True))
        if self.abscissa is not None:
            to, xo = out_data[:, 0], out_data[:, 1:]
            tg, xg = self.gold_data[:, 0], self.gold_data[:, 1:]
        else:
            xo = data
            xg = self.gold_data

        # do the comparison
        anrmsd = np.empty(self.nc)
        if self.abscissa is not None:
            for idx in range(self.nc):
                rmsd, nrmsd = pu.compute_rms(tg, xg[:, idx], to, xo[:, idx])
                anrmsd[idx] = nrmsd
                continue
        else:
            for idx in range(self.nc):
                rmsd = np.sqrt(np.mean((xg[:, idx] - xo[:, idx]) ** 2))
                dnom = abs(np.amax(xo[:, idx]) - np.amin(xo[:, idx]))
                nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
                anrmsd[idx] = nrmsd
                continue

        return np.amax(np.abs(anrmsd))
