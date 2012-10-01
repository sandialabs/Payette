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


HAS_ABSCISSA = False
NC = None


def init(*args):
    """Initialize data needed to compute the error

    """

    global NC, HAS_ABSCISSA

    # Do operations on the gold file here so that they are only done once
    gold_f = args[0]
    if gold_f is None:
        pu.report_and_raise_error("no obj_dat given for Opt_youngs")

    elif not os.path.isfile(gold_f):
        pu.report_and_raise_error("{0} not found".format(gold_f))

    # get vars to minimize
    abscissa = args[1]
    minvars = args[2]
    NC = len(minvars)

    if abscissa is not None:
        HAS_ABSCISSA = True
        minvars = [abscissa] + minvars

    # extract only what we want from the gold and output files
    xg = np.array(pe.extract(exargs(gold_f, initial=minvars), silent=True))

    _xg(initial=xg)
    return


def exargs(fnam, mv=[None], initial=None):
    if initial is not None:
        if not isinstance(initial, (list, tuple)):
            initial = [initial]
        mv[0] = initial
    return [fnam] + mv[0]


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
        xg[0] = np.array(initial)
    if HAS_ABSCISSA:
        return xg[0][:, 0], xg[0][:, 1:]
    return xg[0]


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

    # extract only what we want from the gold and output files
    out_f = args[0]
    dat = np.array(pe.extract(exargs(out_f), silent=True))
    gdat = _xg()
    if HAS_ABSCISSA:
        to, xo = dat[:, 0], dat[:, 1:]
        tg, xg = gdat[0], gdat[1]
    else:
        xo = dat
        xg = gdat

    # do the comparison
    anrmsd = np.empty(NC)
    if HAS_ABSCISSA:
        for idx in range(NC):
            rmsd, nrmsd = pu.compute_rms(tg, xg[:, idx], to, xo[:, idx])
            anrmsd[idx] = nrmsd
            continue
    else:
        for idx in range(NC):
            rmsd = np.sqrt(np.mean((xg[:, idx] - xo[:, idx]) ** 2))
            dnom = abs(np.amax(xo[:, idx]) - np.amin(xo[:, idx]))
            nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
            anrmsd[idx] = nrmsd
            continue

    return np.amax(np.abs(anrmsd))
