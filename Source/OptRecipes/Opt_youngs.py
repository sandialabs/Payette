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

"""Provides a recipe for computing the error between the expected Young's
modulus and the simulated.

"""
import os, sys
import numpy as np
import Source.Payette_utils as pu
import Source.Payette_extract as pe


def exargs(fnam):
    return fnam, "@strain11", "@sig11"

def init(*args):
    """Initialize data needed to compute the error

    """

    # Do operations on the gold file here so that they are only done once
    gold_f = args[0]
    if gold_f is None:
        pu.report_and_raise_error("no obj_dat given for Opt_youngs")

    elif not os.path.isfile(gold_f):
        pu.report_and_raise_error("{0} not found".format(gold_f))

    # extract only what we want from the gold and output files
    xg = np.array(pe.extract(exargs(gold_f), silent=True))

    # find the Young's modulus
    Eg = []
    eps, sig = xg[:, 0], xg[:, 1]
    for idx in range(len(sig) - 1):
        deps = eps[idx + 1] - eps[idx]
        dsig = sig[idx + 1] - sig[idx]
        if abs(deps) > 1.e-16:
            Eg.append(dsig / deps)
        continue
    Eg = np.mean(np.array(Eg))
    _Eg(initial=Eg)
    return


def _Eg(Eg=[None], initial=None):
    """Manage the Young's modulus from the gold file

    Parameters
    ----------
    Eg : list
        Eg[0] is the Young's modulus
    initial : None or float, optional
        if float, set the intial value of Eg

    Returns
    -------
    Eg[0] : float
        the Young's modulus

    """
    if initial is not None:
        Eg[0] = float(initial)
    return Eg[0]


def obj_fn(*args):
    """Evaluates the error between the simulation output and the "gold" answer

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
    With this objective function, the maximum root mean squared error between
    the Young's modulus computed from the gold file and the output file is
    returned as the error.

    """

    out_f = args[0]

    # extract only what we want from the gold and output files
    xo = np.array(pe.extract(exargs(out_f), silent=True))

    # do the comparison
    Eo = []
    for idx in range(len(xo[:, 0]) - 1):
        deps = xo[:, 0][idx + 1] - xo[:, 0][idx]
        dsig = xo[:, 1][idx + 1] - xo[:, 1][idx]
        if abs(deps) > 1.e-16:
            Eo.append(dsig / deps)
        continue
    Em = np.mean(np.array(Eo))
    error = np.abs((_Eg() - Em) / _Eg())
    return error
