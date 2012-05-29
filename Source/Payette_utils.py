# The MIT License

# Copyright (c) 2011 Tim Fuller

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
from __future__ import print_function

import os, sys
import logging
import re
import math
import linecache
import numpy as np
import pickle
import imp
from inspect import stack

import Payette_config as pc
import Source.Payette_extract as pe

if not os.path.isfile(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),"../Payette_config.py")):
    sys.exit("ERROR: Payette_config.py must be written by configure.py")

EPSILON = np.finfo(np.float).eps
ACCLIM = .0001 / 100.


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


class CountCalls(object):
   "Decorator that keeps track of the number of times a function is called."

   __instances = {}

   def __init__(self, f):
      self.__f = f
      self.__numcalls = 0
      CountCalls.__instances[f] = self

   def __call__(self, *args, **kwargs):
      self.__numcalls += 1
      return self.__f(*args, **kwargs)

   def count(self):
      "Return the number of times the function f was called."
      return CountCalls.__instances[self.__f].__numcalls

   @staticmethod
   def counts():
      "Return a dict of {function: # of calls} for all registered functions."
      return dict([(f.__name__, CountCalls.__instances[f].__numcalls)
                   for f in CountCalls.__instances])


class PayetteError(Exception):
    def __init__(self, msg, tracebacklimit=None):

        if tracebacklimit is not None:
            sys.tracebacklimit = tracebacklimit

        l = 79 # should be odd number
        st, stsp = '*'*l + '\n', '*' + ' '*(l-2) + '*\n'
        psf = 'Payette simulation failed'
        ll = (l - len(psf) - 2)/2
        psa = '*' + ' '*ll + psf + ' '*ll + '*\n'
        head = '\n\n' + st + stsp + psa + stsp + st
        Exception.__init__(self, head+msg)


def whoami():
    """ return name of calling function """
    return stack()[1][3]


@CountCalls
def reportMessage(f, msg, pre="INFO: "):
    """ report message to log """
#    msg = 'INFO: {0} (reported from [{1}])\n'.format(msg, f)
    msg = '{0}{1}\n'.format(pre, msg)
    simlog.write(msg)
    if loglevel > 0: sys.stdout.write(msg)
    return


@CountCalls
def reportWarning(f, msg, limit=False):
    """ report warning to log """
    wcount = CountCalls.counts()[whoami()]
    if limit:
        max_warn = 1000
        if wcount <= max_warn / 200 - 1:
            thresh = wcount + 1
        elif wcount > max_warn / 200 and wcount < max_warn / 100:
            thresh = max_warn / 100
        elif wcount > max_warn / 100 and wcount < max_warn / 10:
            thresh = max_warn / 10
        else:
            thresh = max_warn
        wcount += 1
        if wcount != thresh:
            return

    msg = 'WARNING: {0} (reported from [{1}])\n'.format(msg, f)
    simlog.write(msg)
    if loglevel > 0:
        sys.stdout.write(msg)
    return


@CountCalls
def reportError(iam, msg, tracebacklimit=None):
    """ report error to log and raise error """
    iam = _adjust_nam_length(iam)
    msg = '{0} (reported from [{1}])'.format(msg, iam)
    raise PayetteError(msg, tracebacklimit)
    return


def migMessage(msg):
    """ mig interface to reportMessage """
    reportMessage("MIG", msg)
    return


def migError(msg):
    """ mig interface to reportError """
    msg = ' '.join([x for x in msg.split(' ') if x])
    reportError("MIG", msg, tracebacklimit=0)
    return


def migWarning(msg):
    """ mig interface to reportWarning """
    reportWarning("MIG", msg)
    return


def write_msg_to_screen(iam, msg):
    """ write a message to stdout """
    sys.stdout.write("INFO: {0:s}\n".format(msg))
    return


def write_wrn_to_screen(iam, msg):
    """ write warning to stdout """
    sys.stdout.write("WARNING: {0:s}\n".format(msg))
    return


def write_to_simlog(msg):
    """ write message to simulation log """
    msg = "{0:s}\n".format(msg)
    simlog.write(msg)
    return


def _adjust_nam_length(nam):
    """ adjust the length of a file name """
    if not os.path.isfile(nam):
        return nam
    return os.path.splitext(os.path.basename(nam))[0]


def parse_token(n,stringa,token=r'|'):
    """ python implementation of mig partok """
    parsed_string = []
    i = 0
    while i < n:
        stat = False
        for s in stringa:
            try: s = s.decode('utf-8')
            except: s = str(s)
            if not stat:
                x = ''
                stat = True
                pass
            if s != token:
                x += s
            else:
                parsed_string.append(str(x))
                i += 1
                stat = False
                pass
            continue
        continue
    return parsed_string

def check_py_version():
    """ check python version compatibility """
    (major, minor, micro, releaselevel, serial) = sys.version_info
    if (major != 3 and major != 2) or (major == 2 and minor < 6):
        raise SystemExit("Payette requires Python >= 2.6\n")
    return


def setup_logger(logfile, level, mode="w"):
    """ set up the simulation logger """
    global loglevel, simlog
    loglevel = level
    simlog = open(logfile, mode)
    if mode == "w":
        simlog.write(pc.PC_INTRO + "\n")
    return


def parse_error(message):
    """ quite on parsing error """
    sys.exit("ERROR: {0}".format(message))


def read_input(user_input, user_cchar=None):
    """ read a list of user inputs and return """

    if not user_input:
        parse_error("no user input sent to read_input")
        pass

    # comment characters
    cchars = ['#','$']
    if user_cchar is not None:
        cchars.append(user_cchar)

    # get all of the input blocks for the file
    input_lines = _get_input_lines(user_input, cchars)
    input_sets = _get_blocks(input_lines)

    # input_sets contains a list of all blocks in the file, parse it to make
    # sure that a simulation is given
    recognized_blocks = ("simulation", "boundary", "legs", "material",
                         "optimization", "permutation", "enumeration",
                         "mathplot", "name", "content", "extraction",
                         "output")
    incompatible_blocks = (("visualization", "optimization", "enumeration"),)

    user_dict = {}
    errors = 0

    for input_set in input_sets:

        if "simulation" not in input_set:
            errors += 1
            logerr("no simulation block found")
            continue

        simkey = input_set["simulation"]["name"]
        if not simkey:
            errors += 1
            logerr('did not find simulation name.  Simulation block '
                   'must be for form:\n'
                   '\tbegin simulation simulation name ... end simulation')
            continue

        # check for incompatibilities
        bad_blocks = [x for x in input_set["simulation"]
                      if x not in recognized_blocks]

        if bad_blocks:
            errors += 1
            logerr("unrecognized blocks: {0}".format(", ".join(bad_blocks)))

        for item in incompatible_blocks:
            bad_blocks = [x for x in input_set["simulation"] if x in item]
            if len(bad_blocks) > 1:
                errors += 1
                logerr("{0} blocks incompatible, choose one"
                       .format(", ".join(bad_blocks)))
            continue

        user_dict[simkey] = input_set["simulation"]

        continue

    if errors:
        parse_error("resolve previous errors")

    return user_dict


def _get_blocks(user_input):
    """ Find all input blocks in user_input.

    Input blocks are blocks of instructions in

        begin keyword [title]
               .
               .
               .
        end keyword

    blocks.

    Parameters
    ----------
    user_input : array_like
        Split user_input

    Returns
    -------
    blocks : dict
        Dictionary containing all blocks.
        keys:
            simulation
            boundary
            legs
            special
            mathplot

    """

    block_tree = []
    block_stack = []
    block = {}

    for iline, line in enumerate(user_input):

        split_line = line.strip().lower().split()

        if not split_line:
            continue

        if split_line[0] == "begin":
            # Encountered a new block. Before continuing, we need to decide
            # what to do with it. Possibilities are:
            #
            #    1. Start a new block dictionary if this block is not nested
            #    2. Append this block to the previous if it is nested

            # get the block type
            try:
                block_typ = split_line[1]

            except ValueError:
                parse_error("encountered a begin directive with no block type")

            # get the (optional) block name
            try:
                block_nam = "_".join(split_line[2:])

            except ValueError:
                block_nam = None

            new_block = {block_typ: {"name": block_nam, "content": [], }}

            if not block_stack:
                # First encountered block, old block is now done, store it
                if block:
                    block_tree.append(block)

                block = new_block

            else:
                if block_typ in block[block_stack[0]]:
                    parse_error("duplicate block \"{0}\" encountered"
                                .format(block_typ))

                block[block_stack[0]][block_typ] = new_block[block_typ]

            # Append the block type to the block stack. The block stack is a
            # list of blocks we are currently in.
            block_stack.append(block_typ)
            continue

        elif split_line[0] == "end":

            # Reached the end of a block. Make sure that it is the end of the
            # most current block.
            try:
                block_typ = split_line[1]
            except ValueError:
                parse_error("encountered a end directive with no block type")

            if block_stack[-1] != block_typ:
                parse_error('unexpected "end {0}" directive, expected "end {1}"'
                            .format(block_typ, block_stack[-1]))

            # Remove this block from the block stack
            block_stack.pop()
            try:
                block_typ = block_stack[-1]

            except IndexError:
                block_typ = None

            continue

        # Currently in a block,
        if not block_stack:
            continue

        if block_stack[0] == block_typ:
            block[block_typ]["content"].append(line)

        else:
            block[block_stack[0]][block_typ]["content"].append(line)

        continue

    block_tree.append(block)

    return block_tree


def _remove_block(input_lines, block):
    """ remove the requested block from input_lines """
    idx_0, idx_f, lines = _find_block(input_lines, block)
    del input_lines[idx_0:idx_f + 1]
    return input_lines


def _find_block(input_lines, block):
    """ find block in input_lines """
    block_lines = []
    idx_0, idx_f = None, None
    for idx, line in enumerate(input_lines):
        sline = line.split()
        if sline[0].lower() == "begin":
            if sline[1] == block:
                idx_0 = idx
        elif sline[0].lower() == "end":
            if sline[1] == block:
                idx_f = idx
        continue

    if idx_0 is not None and idx_f is not None:
        block_lines = input_lines[idx_0 + 1:idx_f]

    return idx_0, idx_f, block_lines


def textformat(var):
    """
    textformat() is used to get consistent formatting for all of
    the variables that are printed to the output file. Most of the
    possible variables are of numpy-derived types.

    Created: 17 June 2011 by mswan
    """
    if isinstance(var,(int,float,np.float32,np.float64)):
        return "{0:20.10E}".format(float(var))
    elif isinstance(var,str):
        return "{0:>20s}".format(str(var))
    elif isinstance(var,(np.int,   np.int8,    np.int16,
                         np.int32, np.int64,   np.uint8,
                         np.uint16,np.uint32,  np.uint64,
                         np.float)):
        return "{0:20.10E}".format(float(var))
    else:
        return "{0:>20}".format(str(var))


def close_aux_files():
    """ close auxilary files """
    try:
        simlog.close()

    except OSError:
        pass


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

    if isinstance(x,(float,int,str,bool)): return x

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
        continue
    return result

class BuildError(Exception):
    def __init__(self, message, errno):
        # errno:
        # 1: bad input files
        # 2: f2py failed
        #  5 = environmental variable not found (not error)
        # 10 = source files/directories not found
        # 35 = Extension module not imported
        # 40 = Bad/no sigfile
        # 66 = No build attribute
        self.message = message
        self.errno = errno
        logwrn(message)
        pass

    def __repr__(self):
        return self.__name__

    def __str__(self):
        return repr(self.errno)


def get_module_name_and_path(py_file):
    """ return module name and path as list """
    return (os.path.splitext(os.path.basename(py_file))[0],
            [os.path.dirname(py_file)])


def get_module_name(py_file):
    """ return the module name of py_file """
    return get_module_name_and_path(py_file)[0]


def begmes(msg, pre="", end="  ", verbose=True):
    """ begin a message to stdout """
    if not verbose:
        return
    print("{0}{1}...".format(pre, msg), end=end)
    return


def endmes(msg, pre="", end="\n", verbose=True):
    """ end message to stdout """
    if not verbose:
        return
    print("{0}{1}".format(pre, msg), end=end)
    return


def loginf(msg, pre="", end="\n", caller=None, verbose=True):
    """ log an info message to stdout """
    if not verbose:
        return
    if caller is not None:
        msg = "{0} [reported by {1}]".format(msg, caller)
    print("{0}INFO: {1}".format(pre, msg), end=end)
    return


def logmes(msg, pre="", end="\n", caller=None, verbose=True):
    """ log a message to stdout """
    if not verbose:
        return
    if caller is not None:
        msg = "{0} [reported by {1}]".format(msg, caller)
    print("{0}{1}".format(pre, msg), end=end)
    return


def logwrn(msg, pre="", end="\n", caller=None):
    """ log a warning to stdout """
    if caller is not None:
        msg = "{0} [reported by {1}]".format(msg, caller)

    print("{0}WARNING: {1}".format(pre, msg), end=end)
    return


def logerr(msg, pre="", end="\n", caller=None):
    """ log an error to stdout """
    if caller is not None:
        msg = "{0} [reported by {1}]".format(msg, caller)

    print("{0}ERROR: {1}".format(pre, msg), end=end)
    return

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
    return linecache.getline(fpath, 1).split()

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
        sys.exit("len(set1x) != len(set1y)")
    if lset2x != lset2y:
        sys.exit("len(set2x) != len(set2y)")

    if lset1x < 2:
        sys.exit("set1 must have at least two points.")
    if lset2x < 1:
        sys.exit("set2 must have at least one point.")

    dx = max(set1x)-min(set1x)
    dy = max(set1y)-min(set1y)
    dd = math.sqrt(dx*dx+dy*dy)

    dist_pt_to_pt = lambda x0, y0, x1, y1: math.sqrt((x1-x0)**2+(y1-y0)**2)
    # compute the running square of the difference
    err = 0.0
    for idx in range(0, lset2x):
        tmp_arr = []
        for jdx in range(0, lset1x - 1):
            kdx = jdx+1
            dist_from_pt0 = dist_pt_to_pt(
                set1x[jdx], set1y[jdx], set2x[idx], set2y[idx])
            dist_from_pt1 = dist_pt_to_pt(
                set1x[kdx], set1y[kdx], set2x[idx], set2y[idx])

            # use dot(a,b)/(mag(a)*mag(b)) = cos(theta) to find the distance
            # from the line.
            vec_a_x = set1x[jdx]-set1x[kdx]
            vec_a_y = set1y[jdx]-set1y[kdx]
            vec_b_x = set2x[idx]-set1x[kdx]
            vec_b_y = set2y[idx]-set1y[kdx]
            mag_a = math.sqrt(vec_a_x**2 + vec_a_y**2)
            mag_b = math.sqrt(vec_b_x**2 + vec_b_y**2)

            if mag_a == 0.0 or mag_b == 0.0:
                tmp_arr.append(min(dist_from_pt0,dist_from_pt1))
                continue

            costheta = (vec_a_x*vec_b_x+vec_a_y*vec_b_y)/mag_a/mag_b

            if costheta < 0.0 or mag_b*costheta > mag_a:
                tmp_arr.append(min(dist_from_pt0,dist_from_pt1))
                continue

            theta = math.acos( max(min(1.0,costheta),-1.0) )
            dist_from_line = mag_b*math.sin(theta)

            dist = min( dist_from_line, min(dist_from_pt0,dist_from_pt1) )
            tmp_arr.append(dist)
            continue
        err += min(tmp_arr)
        continue

#    rmsd = math.sqrt(err / float(lset1x))
#    dnom = abs(np.amax(set1y) - np.amin(set1y))
#    nrmsd = rmsd / dnom if dnom >= 2.e-16 else rmsd
#    return rmsd, nrmsd
    return err, err/dd



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
        sys.exit("len(set1x) != len(set1y)")
    if lset2x != lset2y:
        sys.exit("len(set2x) != len(set2y)")

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

    iam = "compare_out_to_gold_rms(gold_f, out_f)"
    errors = 0

    if not os.path.isfile(gold_f):
        logerr("gold file {0} not found".format(gold_f), caller=iam)
        errors += 1
        pass

    if not os.path.isfile(out_f):
        logerr("output file {0} not found".format(out_f), caller=iam)
        errors += 1
        pass

    if errors:
        return errors, None, None

    # read in header
    out_h = [x.lower() for x in get_header(out_f)]
    gold_h = [x.lower() for x in get_header(gold_f)]

    if out_h[0] != "time":
        errors += 1
        logerr("time not in outfile {0}".format(out_f), caller=iam)

    if gold_h[0] != "time":
        errors += 1
        logerr("time not in gold file {0}".format(gold_f), caller=iam)

    if errors:
        return errors, None, None

    # read in data
    out = read_data(out_f)
    gold = read_data(gold_f)

    # check that time is same (lengths must be the same)
    if len(gold[:, 0]) == len(out[:, 0]):
        rmsd, nrmsd = compute_fast_rms(gold[:, 0], out[:, 0])

    else:
        rmsd, nrmsd = 1.0e99, 1.0e99

    if nrmsd > EPSILON:
        errors += 1
        logerr("time step error between {0} and {1}".format(out_f, gold_f),
               caller=iam)

    if errors:
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

    iam = "compare_file_cols(file_1, file_2, cols)"
    errors = 0

    if not os.path.isfile(file_1):
        logerr("file {0} not found".format(file_1), caller=iam)
        errors += 1
        pass

    if not os.path.isfile(file_2):
        logerr("file {0} not found".format(file_2), caller=iam)
        errors += 1
        pass

    if errors:
        return errors, None, None

    # read in header
    head_1 = [x.lower() for x in get_header(file_1)]
    head_2 = [x.lower() for x in get_header(file_2)]

    # read in data
    dat_1 = read_data(file_1)
    dat_2 = read_data(file_2)

    # for now, the number of rows must be the same
    if dat_1.shape[0] != dat_2.shape[0]:
        logerr("shape of {0} != shape of {1}"
               .format(file_1, file_2), caller=iam)
        return 1, None, None

    to_compare = [x for x in head_2 if x in head_1] # if x not in to_skip]

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


def write_input_file(nam, inp_dict, inp_f):
    """ from an input dictionary, write the input file

    Parameters
    ----------
    nam : str
        The name of the simulation

    inp_dict : dict
        The input dictionary

    inp_f : str
        Path to input file to be written

    Returns
    -------
    None

    """

    req_blocks = ("content", "simulation", "boundary", "legs", "material", "name")
    with open(inp_f, "w") as fobj:
        fobj.write("begin simulation {0}\n".format(nam))
        content = inp_dict["content"]
        if content:
            fobj.write("\n".join(content) + "\n")
        # boundary block
        fobj.write("begin boundary\n")
        fobj.write("\n".join(inp_dict["boundary"]["content"]) + "\n")
        # legs block
        fobj.write("begin legs\n")
        fobj.write("\n".join(inp_dict["legs"]["content"]) + "\n")
        fobj.write("end legs\n")
        fobj.write("end boundary\n")
        # material block
        fobj.write("begin material\n")
        fobj.write("\n".join(inp_dict["material"]["content"]) + "\n")
        fobj.write("end material\n")
        for key, val in inp_dict.items():
            if key in req_blocks:
                continue
            fobj.write("begin {0}\n".format(key))
            fobj.write("\n".join(val["content"]) + "\n")
            fobj.write("end {0}\n".format(key) + "\n")
        fobj.write("end simulation")
    return


def _get_input_lines(raw_user_input, cchars):
    """Read the user input, inserting files if encountered

    Parameters
    ----------
    user_input : list
        New line separated list of the input file
    cchars : list
        List of comment characters

    Returns
    -------
    all_input : list
        All of the read input, including inserted files

    """

    insert_kws = ("insert", "include")

    used_blocks = []
    all_input = []
    iline = 0

    user_input = _remove_all_comments(raw_user_input, cchars)

    # infinite loop for reading the input file and getting all inserted files
    while True:

        # if we have gone through one time, reset user_input to be all_input
        # and run through again until all inserted files have been read in
        if iline == len(user_input):
            if [x for x in all_input if x.split()[0] in insert_kws]:
                user_input = [x for x in all_input]
                all_input = []
                iline = 0
            else:
                break

        # get the next line of input
        line = user_input[iline]

        # check for internal "use" directives
        if line.split()[0] == "use":
            block = " ".join(line.split()[1:])
            # check if insert is given in file
            idx_0, idx_f, block_insert = _find_block(user_input, block)
            if idx_0 is None:
                parse_error("'use' block '{0:s}' not found".format(block))
            elif idx_f is None:
                parse_error("end of 'use' block '{0:s}' not found"
                            .format(block))

            used_blocks.append(block)
            all_input.extend(block_insert)

        # check for inserts
        elif line.split()[0] in insert_kws:
            insert = " ".join(line.split()[1:])

            if not os.path.isfile(insert):
                parse_error("inserted file '{0:s}' not found".format(insert))

            insert_lines = open(insert, "r").readlines()
            all_input.extend(_remove_all_comments(insert_lines, cchars))

        else:
            all_input.append(line)

        iline += 1

        continue

    for block in list(set(used_blocks)):
        all_input = _remove_block(all_input, block)

    return all_input


def _remove_all_comments(lines, cchars):
    """ remove all comments from lines """
    stripped_lines = []
    for line in lines:
        line = line.strip()
        # skip blank and comment lines
        if not line.split() or line[0] in cchars:
            continue

        # remove inline comments
        for cchar in cchars:
            line = line.split(cchar)[0]

        stripped_lines.append(line)
        continue
    return stripped_lines


def get_constitutive_model(model_name):
    """ get the constitutive model dictionary of model_name """
    constitutive_model = None
    constitutive_models = get_installed_models()

    for key, val in constitutive_models.items():
        if model_name == key or model_name in val["aliases"]:
            constitutive_model = val
        continue

    if constitutive_model is None:
        logerr("constitutive model {0} not found".format(model_name))

    return constitutive_model


def get_constitutive_model_object(model_name):
    """ get the actual model object """
    constitutive_model = get_constitutive_model(model_name)
    if constitutive_model is None:
        sys.exit("stopping due to previous errors")
    py_mod = constitutive_model["module"]
    py_path = [os.path.dirname(constitutive_model["file"])]
    cls_nam = constitutive_model["class name"]
    fobj, pathname, description = imp.find_module(py_mod, py_path)
    py_module = imp.load_module(py_mod, fobj, pathname, description)
    fobj.close()
    cmod = getattr(py_module, cls_nam)
    return cmod


def get_installed_models():
    """ return a list of all installed models """
    with open(pc.PC_MTLS_FILE, "rb") as fobj:
        constitutive_models = pickle.load(fobj)
    return constitutive_models


# the following are being kept around for back compatibiltiy
def parseToken(*args, **kwargs):
    return parse_token(*args, **kwargs)
def checkPythonVersion():
    return check_py_version()
def writeMessage(*args):
    return write_msg_to_screen(*args)
def writeWarning(*args):
    return write_wrn_to_screen(*args)
def writeToLog(*args):
    return write_to_simlog(*args)

