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

"""Provides a recipe for computing the maximum error between the expected
stress vs. time slope and the simulated.

"""
import os, sys
import numpy as np
import Source.Payette_utils as pu
import Source.Payette_extract as pe


class ObjectiveFunction(object):

    def __init__(self, *args):
        """With this objective function, average of the maximum root mean
        squared error between SIG11 vs STRAIN11 for the elastic portion and
        ROOTJ2 for the plastic portion is returned as the error.

        Parameters
        ----------
        args[0] : str
            The gold file. Must contain columns labeled TIME, SIG11, SIG22,
            SIG33, SIG12, SIG23, SIG13, STRAIN11.

        """

        self.minvars = ["@time", "@strain11",
                        "@sig11", "@sig22", "@sig33",
                        "@sig12", "@sig23", "@sig13"]
        gold_f = args[0]
        if gold_f is None:
            pu.report_and_raise_error("No gold file given for Opt_sig_v_time.py")

        elif not os.path.isfile(gold_f):
            pu.report_and_raise_error("{0} not found".format(gold_f))

        # extract only what we want from the gold and output files
        exargs = [gold_f] + self.minvars
        self.gold_data = np.array(pe.extract(exargs, silent=True))

        # find Young's modulus and value of sqrt(J2) at yield
        self.Eg, self.RTJ2g = _find_e_and_y(self.gold_data)
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

        # extract only what we want from the gold and output files
        out_f = args[0]
        if not os.path.isfile(out_f):
            pu.report_and_raise_error("{0} not found".format(out_f))
        exargs = [out_f] + self.minvars
        out_data = np.array(pe.extract(exargs, silent=True))
        E, rtj2 = _find_e_and_y(out_data)
        error = np.sqrt(np.mean(np.array([self.Eg - E, self.RTJ2g - rtj2]) ** 2))
        return error


def _find_e_and_y(data):
    """Find the Young's modulus"""
    eps, sig = data[:, 1], data[:, 2]
    youngs = [(sig[1] - sig[0]) / (eps[1] - eps[0])]
    for i in range(1, len(sig) - 1):
        deps = eps[i + 1] - eps[i]
        dsig = sig[i + 1] - sig[i]
        if abs(deps) > 1.e-16:
            Ec = dsig / deps
            if Ec < .97 * np.mean(np.array(youngs)):
                break
            youngs.append(Ec)
        continue
    youngs = np.mean(np.array(youngs))

    # stresses after inelastic transition
    s = data[i:, 2:]
    j2 = (1. / 6. * ((s[0] - s[1]) ** 2 +
                     (s[1] - s[2]) ** 2 +
                     (s[2] - s[0]) ** 2) +
          s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
    rtj2 = np.mean(np.sqrt(j2))
    return youngs, rtj2
