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
import os, sys
import numpy as np
import Source.Payette_utils as pu
import Source.Payette_extract as pe

#
NC = 3

def exargs(fnam):
    """Argument to send to pe.extract

    Note
    ----
    These are the variables that will be extracted and minimized during the
    optimization routine

    """
    return fnam, "@sig11", "@sig22", "@sig33"


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

    _xg(initial=xg)
    return


def _xg(xg=[None], initial=None):
    """Manage the gold file data

    Parameters
    ----------
    xg : list
        xg[0] is the gold file data
    initial : None or array, optional
        if not None, set the intial value of xg

    Returns
    -------
    xg[0] : array
        the gold file data

    """
    if initial is not None:
        xg[0] = initial
    return np.array(xg[0])




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
    SIG11, SIG22, and SIG33 from the simulation output and the gold result is
    returned as the error.

    """

    out_f = args[0]

    # extract only what we want from the gold and output files
    xo = np.array(pe.extract(exargs(out_f), silent=True))

    # do the comparison
    anrmsd = []
    xg = _xg()
    for idx in range(NC):
        rmsd = np.sqrt(np.mean((xg[:, idx] - xo[:, idx]) ** 2))
        dnom = abs(np.amax(xo[:, idx]) - np.amin(xo[:, idx]))
        nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
        anrmsd.append(nrmsd)
        continue

    error = np.amax(np.abs(np.array(anrmsd)))
    return error
