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
import logging
import re
import math
import linecache
import numpy as np
import pickle
import inspect
from textwrap import fill as textfill

import Source.__config__ as cfg
import Source.Payette_extract as pe
import Source.__runopts__ as ro

SIMLOG = None
EPSILON = np.finfo(np.float).eps


'''
NAME
   Payette_utils

PURPOSE
   This file provides a number of functions and common variables used throughout
   the Payette application. The documentation is sparse and will be updated as time
   allows.

AUTHORS
   Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
'''


class DummyHolder:
    pass


def whoami():
    """ return name of calling function """
    return inspect.stack()[1][3]


def who_is_calling():
    """return the name of the calling function"""
    stack = inspect.stack()[2]
    return "{0}.{1}".format(
        os.path.splitext(os.path.basename(stack[1]))[0], stack[3])


def log_message(message, pre="INFO: ", end="\n", noisy=False, beg=""):
    """Report message to screen and write to log file if open"""
    message = "{0}{1}{2}{3}".format(beg, pre, message, end)
    if SIMLOG is not None:
        SIMLOG.write(message)
    if noisy or ro.VERBOSITY > 0:
        sys.stdout.write(message)
    return


# the following methods define error logging, counting
def reset_error_and_warnings():
    __count_error(reset=True)
    __count_warning(reset=True)
    return


def error_count():
    """Return the current number of errors"""
    return __count_error(inquire=True)


def __count_error(ecount=[0], inquire=False, reset=False):
    """Count the number of errors"""
    if reset:
        ecount = [0]
        return
    if inquire:
        return ecount[0]
    ecount[0] += 1
    return


def report_error(message, count=True):
    """Report error to screen and write to log file if open"""
    if count:
        __count_error()
    stack = inspect.stack()[1]
    message = "ERROR: {0} [reported by: {1}]\n".format(message,
                                                       who_is_calling())
    if SIMLOG is not None:
        SIMLOG.write(message)
    sys.stdout.flush()
    sys.stderr.write(message)
    return


def report_and_raise_error(message, tracebacklimit=None, caller=None):
    """Report and raise an error"""
    if caller is None:
        caller = who_is_calling()
    from Source.Payette_container import PayetteError as PayetteError
    raise PayetteError(message, caller=caller)


# the following methods define warning logging, counting
def warn_count():
    return __count_warning(inquire=True)


def __count_warning(wcount=[0], inquire=False, reset=False):
    if reset:
        wcount = [0]
        return
    if inquire:
        return wcount[0]
    wcount[0] += 1
    return


def log_warning(message, limit=False, caller=None, pre="WARNING: ",
                beg="", end="\n"):
    """Report warning to screen and write to log file if open"""

    if ro.WARNING == "error":
        report_and_raise_error(message, caller=who_is_calling())

    __count_warning()

    if ro.WARNING == "ignore":
        return

    if limit and ro.WARNING != "all":
        max_warn = 1000
        wcount = warn_count()
        if wcount <= max_warn / 200 - 1:
            thresh = wcount + 1
        elif wcount > max_warn / 200 and wcount < max_warn / 100:
            thresh = max_warn / 100
        elif wcount > max_warn / 100 and wcount < max_warn / 10:
            thresh = max_warn / 10
        else:
            thresh = max_warn
        if wcount != thresh:
            return

    if caller is None:
        caller = who_is_calling()
    elif caller.lower() == "anonymous":
        caller = ""

    if caller:
        caller = " [reported by: {0}]".format(caller)

    message = (textfill("{0}{1}{2}{3}".format(beg, pre, message, caller),
                        subsequent_indent=" " * (len(beg) + len(pre))) + end)
    if SIMLOG is not None:
        SIMLOG.write(message)
    sys.stdout.flush()
    sys.stderr.write(message)
    return


def write_to_simlog(msg):
    """ write message to simulation log """
    msg = "{0:s}\n".format(msg)
    SIMLOG.write(msg)
    return


def parse_token(n, stringa, token=r'|'):
    """ python implementation of mig partok """
    parsed_string = []
    i = 0
    while i < n:
        stat = False
        for s in stringa:
            try:
                s = s.decode('utf-8')
            except:
                s = str(s)
            if not stat:
                x = ''
                stat = True

            if s != token:
                x += s
            else:
                parsed_string.append(str(x))
                i += 1
                stat = False

            continue
        continue
    return parsed_string


def check_py_version():
    """ check python version compatibility """
    (major, minor, micro, releaselevel, serial) = sys.version_info
    if (major != 3 and major != 2) or (major == 2 and minor < 6):
        raise SystemExit("Payette requires Python >= 2.6\n")
    return


def setup_logger(logfile, mode="w"):
    """ set up the simulation logger """
    global SIMLOG
    SIMLOG = open(logfile, mode)
    if mode == "w":
        SIMLOG.write(cfg.INTRO + "\n")
    return


def textformat(var):
    """
    textformat() is used to get consistent formatting for all of
    the variables that are printed to the output file. Most of the
    possible variables are of numpy-derived types.

    Created: 17 June 2011 by mswan
    """
    try:
        return "{0:<20.10E}".format(float(var))
    except:
        return "{0:<20s}".format(str(var))


def close_aux_files():
    """ close auxilary files """
    global SIMLOG
    try:
        SIMLOG.close()
    except OSError:
        pass
    SIMLOG = None
    return


def flatten(x):
    """flatten(sequence) -> list

    Returns a single, flat list which contains all elements retrieved
    from the sequence and all recursively contained sub-sequences
    (iterables).

    Examples:
    >>> [1, 2, [3,4], (5,6)]
    [1, 2, [3, 4], (5, 6)]
    >>> flatten([[[1,2,3], (42,None)], [4,5], [6], 7, MyVector(8,9,10)])
    [1, 2, 3, 42, None, 4, 5, 6, 7, 8, 9, 10]"""

    if isinstance(x, (float, int, str, bool)):
        return x
    result = []
    for el in x:
        # if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
        continue
    return result


def get_module_name_and_path(py_file):
    """ return module name and path as list """
    return (os.path.splitext(os.path.basename(py_file))[0],
            [os.path.dirname(py_file)])


def get_module_name(py_file):
    """ return the module name of py_file """
    return get_module_name_and_path(py_file)[0]


def get_header(fpath):
    """Get the header of f

    Parameters
    ----------
    fpath : str
        Path to file

    Returns
    -------
    head : array_like
        list of strings containing column names

    """
    line = linecache.getline(fpath, 1)
    if line.strip().startswith(ro.CCHAR):
        return line.split()[1:]
    return line.split()


def read_data(fpath):
    """Reads in a whitespace-delimed data file f with numpy.loadtxt.

    It is assumed that the first line contains text (the name of the column).
    All other lines contain floats.

    Parameters
    ----------
    fpath : str
        Path to file

    Returns
    -------
    data : array_like
        m x n dimensional numpy ndarray where m is the number of data points
        (rows) and n the number of data fields (columns)

    """
    return np.loadtxt(fpath, skiprows=1)


def compute_rms_closest_point_residual(set1x, set1y, set2x, set2y):
    r"""Compute the root mean square difference between data in set1{x, y} and
    set2{x, y} by taking each point of set2 and calculating the closest point
    distance to any given line segment of set1.

    "set2{x, y}" is compared against "set1{x, y}".

    len(set1x) == len(set1y)
    len(set2x) == len(set2y)

    Parameters
    ----------
    set1x : array_like
        Abscissa of set 1
    set1y : array_like
        Range of set 1
    set2x : array_like
        Abscissa of set 2
    set2y : array_like
        Range of set 2

    Returns
    -------
    rmsd : float
        root mean square difference between gold and out
    nrmsd : float
        normalized root mean square difference between gold and out

    """

    # check the lengths of the arrays passed
    lset1x, lset1y, lset2x, lset2y = [len(x)
                                      for x in [set1x, set1y, set2x, set2y]]
    if lset1x != lset1y:
        report_and_raise_error("len(set1x) != len(set1y)")
    if lset2x != lset2y:
        report_and_raise_error("len(set2x) != len(set2y)")

    if lset1x < 2:
        report_and_raise_error("set1 must have at least two points.")
    if lset2x < 1:
        report_and_raise_error("set2 must have at least one point.")

    dx = max(set1x) - min(set1x)
    dy = max(set1y) - min(set1y)
    dd = math.sqrt(dx * dx + dy * dy)

    dist_pt_to_pt = lambda x0, y0, x1, y1: math.sqrt(
        (x1 - x0) ** 2 + (y1 - y0) ** 2)
    # compute the running square of the difference
    err = 0.0
    for idx in range(0, lset2x):
        tmp_arr = []
        for jdx in range(0, lset1x - 1):
            kdx = jdx + 1
            dist_from_pt0 = dist_pt_to_pt(
                set1x[jdx], set1y[jdx], set2x[idx], set2y[idx])
            dist_from_pt1 = dist_pt_to_pt(
                set1x[kdx], set1y[kdx], set2x[idx], set2y[idx])

            # use dot(a,b)/(mag(a)*mag(b)) = cos(theta) to find the distance
            # from the line.
            vec_a_x = set1x[jdx] - set1x[kdx]
            vec_a_y = set1y[jdx] - set1y[kdx]
            vec_b_x = set2x[idx] - set1x[kdx]
            vec_b_y = set2y[idx] - set1y[kdx]
            mag_a = math.sqrt(vec_a_x ** 2 + vec_a_y ** 2)
            mag_b = math.sqrt(vec_b_x ** 2 + vec_b_y ** 2)

            if mag_a == 0.0 or mag_b == 0.0:
                tmp_arr.append(min(dist_from_pt0, dist_from_pt1))
                continue

            costheta = (vec_a_x * vec_b_x + vec_a_y * vec_b_y) / mag_a / mag_b

            if costheta < 0.0 or mag_b * costheta > mag_a:
                tmp_arr.append(min(dist_from_pt0, dist_from_pt1))
                continue

            theta = math.acos(max(min(1.0, costheta), -1.0))
            dist_from_line = mag_b * math.sin(theta)

            dist = min(dist_from_line, min(dist_from_pt0, dist_from_pt1))
            tmp_arr.append(dist)
            continue
        err += min(tmp_arr)
        continue

#    rmsd = math.sqrt(err / float(lset1x))
#    dnom = abs(np.amax(set1y) - np.amin(set1y))
#    nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
#    return rmsd, nrmsd
    return err, err / dd


def compute_rms(set1x, set1y, set2x, set2y, step=1):
    r"""Compute the root mean square difference between data in set1{x, y} and
    set2{x, y}.

    Determines the rms of the error between two piecewise functions. In almost
    all cases, the abscissa variables (set1x and set2x) will be time.
    Currently, the abscissa variables are required to be monotonically
    increasing (for easier computing). Interpolations are used to compare data
    points where data is missing in the other set.

    "set2{x, y}" is compared against "set1{x, y}".

    len(set1x) == len(set1y)
    len(set2x) == len(set2y)

    Parameters
    ----------
    set1x : array_like
        Abscissa of set 1
    set1y : array_like
        Range of set 1
    set2x : array_like
        Abscissa of set 2
    set2y : array_like
        Range of set 2
    step : {1,}, optional
        Used for large data sets. This gives the benchmark developer more
        control over how the error is computed (choosing accuracy vs. speed).

    Returns
    -------
    rmsd : float
        root mean square difference between gold and out
    nrmsd : float
        normalized root mean square difference between gold and out

    """

    # check the lengths of the arrays passed
    lset1x, lset1y, lset2x, lset2y = [len(x)
                                      for x in [set1x, set1y, set2x, set2y]]
    if lset1x != lset1y:
        report_and_raise_error("len(set1x) != len(set1y)")
    if lset2x != lset2y:
        report_and_raise_error("len(set2x) != len(set2y)")

    # Use a shortcut if the lengths of the x and y data sets are the same.
    # Also, before using the shortcut, do a quick check by computing the RMS
    # on every 10th x value. The RMS check on the x-values only gets computed
    # if len(set1y) == len(set2y). This check allows for the possibility of
    # the lengths of the data to be equal but for the spacing to be not equal.
    if (lset1y == lset2y and
            np.sum((set1x[::10] - set2x[::10]) ** 2) < 1.0e-6 * np.amax(set1x)):
        return compute_fast_rms(set1y, set2y)

    else:

        # compute the running square of the difference
        err = 0.0
        for i in range(0, lset2x, step):

            for j in range(0, lset1x - 1, step):

                # mss: Very expensive look. Optimize.
                # if set2x is between set1x pair, do linear interpolation
                if set1x[j] <= set2x[i] <= set1x[j + 1]:
                    x_0 = set1x[j]
                    y_0 = set1y[j]
                    x_1 = set1x[j + 1]
                    y_1 = set1y[j + 1]
                    y_f = y_0 + (set2x[i] - x_0) * (y_1 - y_0) / (x_1 - x_0)

                    # Add the square of the difference to err
                    err += (set2y[i] - y_f) ** 2
                    break
                continue
            continue

    rmsd = math.sqrt(err / float(lset1x))
    dnom = abs(np.amax(set1y) - np.amin(set1y))
    nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
    return rmsd, nrmsd


def compute_fast_rms(gold, out):
    r"""Determines the RMS of the error between two piecewise functions of same
    length.

    This should only be called if the timestep sizes of the gold simulation
    and the simulation that produced out are the same. This function makes no
    attempt to make that determination, it is up to the caller.

    Parameters
    ----------
    gold : str
        Path to gold file
    out : str
        Path to out file

    Returns
    -------
    rmsd : float
        root mean square difference between gold and out
    nrmsd : float
        normalized root mean square difference between gold and out

    """
    rmsd = math.sqrt(np.sum((gold - out) ** 2) / float(len(gold)))
    dnom = abs(np.amax(gold) - np.amin(gold))
    nrmsd = rmsd / dnom if dnom >= EPSILON else rmsd
    return rmsd, nrmsd


def compare_out_to_gold_rms(gold_f, out_f, to_skip=None):
    r"""Compare results from out file to those in baseline (gold) file

    Parameters
    ----------
    gold_f : str
        Path to gold file
    out_f : str
        Path to out file

    Returns
    -------
    returncode : int
        0 success, !=0 otherwise
    armsd : array_like
        accumulated root mean square error
    anrmsd : array_like
        accumulated normalized root mean square error

    """

    if not os.path.isfile(gold_f):
        report_error("gold file {0} not found".format(gold_f))

    if not os.path.isfile(out_f):
        report_error("output file {0} not found".format(out_f))

    if error_count():
        return error_count(), None, None

    # read in header
    out_h = [x.lower() for x in get_header(out_f)]
    gold_h = [x.lower() for x in get_header(gold_f)]

    if out_h[0] != "time":
        report_error("time not in outfile {0}".format(out_f))

    if gold_h[0] != "time":
        report_error("time not in gold file {0}".format(gold_f))

    if error_count():
        return error_count(), None, None

    # read in data
    out = read_data(out_f)
    gold = read_data(gold_f)

    # check that time is same (lengths must be the same)
    if len(gold[:, 0]) == len(out[:, 0]):
        rmsd, nrmsd = compute_fast_rms(gold[:, 0], out[:, 0])

    else:
        rmsd, nrmsd = 1.0e99, 1.0e99

    if nrmsd > EPSILON:
        report_error(
            "time step error between {0} and {1}".format(out_f, gold_f))

    if error_count():
        return 1, None, None

    # get items to skip and compare
    if to_skip is None:
        to_skip = []

    to_compare = [x for x in out_h if x in gold_h if x not in to_skip]

    # do the comparison
    anrmsd, armsd = [], []
    for val in to_compare:
        gidx = gold_h.index(val)
        oidx = out_h.index(val)
        rmsd, nrmsd = compute_rms(gold[:, 0], gold[:, gidx],
                                  out[:, 0], out[:, oidx])
        anrmsd.append(nrmsd)
        armsd.append(rmsd)
        continue

    return 0, np.array(anrmsd), np.array(armsd)


def compare_file_cols(file_1, file_2, cols=["all"]):
    r"""Compare columns in file_1 to file_2

    Parameters
    ----------
    file_1 : str
        Path to first file
    file_2 : str
        Path to second file
    cols : array_like
        List of columns to compare

    Returns
    -------
    returncode : int
        0 success, !=0 otherwise
    armsd : array_like
        accumulated root mean square error
    anrmsd : array_like
        accumulated normalized root mean square error

    """

    if not os.path.isfile(file_1):
        report_error("file {0} not found".format(file_1))

    if not os.path.isfile(file_2):
        report_error("file {0} not found".format(file_2))

    if error_count():
        return error_count(), None, None

    # read in header
    head_1 = [x.lower() for x in get_header(file_1)]
    head_2 = [x.lower() for x in get_header(file_2)]

    # read in data
    dat_1 = read_data(file_1)
    dat_2 = read_data(file_2)

    # for now, the number of rows must be the same
    if dat_1.shape[0] != dat_2.shape[0]:
        report_error("shape of {0} != shape of {1}".format(file_1, file_2))
        return 1, None, None

    to_compare = [x for x in head_2 if x in head_1]  # if x not in to_skip]

    # do the comparison
    anrmsd, armsd = [], []
    for val in to_compare:
        idx_1 = head_1.index(val)
        idx_2 = head_2.index(val)
        rmsd = math.sqrt(np.mean((dat_2[:, idx_2] - dat_1[:, idx_1]) ** 2))
        dnom = abs(np.amax(dat_1[:, idx_1]) - np.amin(dat_1[:, idx_1]))
        nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
        anrmsd.append(nrmsd)
        armsd.append(rmsd)
        continue

    return 0, np.array(anrmsd), np.array(armsd)


def get_super_classes(data):
    """ return the super class name from data """

    super_class_names = []
    for super_class in data.super:
        if super_class == "object":
            continue
        if isinstance(super_class, basestring):
            super_class_names.append(super_class)
        else:
            super_class_names.append(super_class.name)
        continue
    return super_class_names
