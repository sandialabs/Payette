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

import Payette_config as pc
import Source.Payette_extract as pe

if not os.path.isfile(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),"../Payette_config.py")):
    sys.exit("ERROR: Payette_config.py must be written by configure.py")

debug = False
wcount, wthresh, wmax = 0, 0, 1000
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


def reportError(f, msg, tracebacklimit=None):
    f = fixFLength(f)
    msg = '{0} (reported from [{1}])'.format(msg, f)
    raise PayetteError(msg, tracebacklimit)
    return


def reportWarning(f, msg, limit=False):
    global wcount, wthresh, wmax
    if limit:
        if wcount <= wmax/200-1: wthresh = wcount+1
        elif wcount > wmax/200 and wcount < wmax/100: wthresh = wmax/100
        elif wcount > wmax/100 and wcount < wmax/10: wthresh = wmax/10
        else: wthresh = wmax
        wcount += 1
        if wcount != wthresh: return
        pass
    msg = 'WARNING: {0} (reported from [{1}])\n'.format(msg, f)
    simlog.write(msg)
    if loglevel > 0: sys.stdout.write(msg)
    return


def writeMessage(f, msg):
    sys.stdout.write("INFO: {0:s}\n".format(msg))
    return


def writeWarning(f, msg):
    sys.stdout.write("WARNING: {0:s}\n".format(msg))
    return


def reportMessage(f, msg, pre="INFO: "):
#    msg = 'INFO: {0} (reported from [{1}])\n'.format(msg, f)
    msg = '{0}{1}\n'.format(pre, msg)
    simlog.write(msg)
    if loglevel > 0: sys.stdout.write(msg)
    return


def writeToLog(msg):
    msg = "{0:s}\n".format(msg)
    simlog.write(msg)
    return


def migMessage(msg):
    reportMessage("MIG", msg)
    return


def migError(msg):
    msg = ' '.join([x for x in msg.split(' ') if x])
    reportError("MIG", msg, tracebacklimit=0)
    return


def migWarning(msg):
    reportWarning("MIG", msg)
    return


def fixFLength(f):
    if not os.path.isfile(f): return f
    f = os.path.split(f)[1]
    basename,fext = os.path.splitext(f)
    return basename


def payetteParametersDir():
    lpd = os.path.join(pc.PC_AUX,'MaterialsDatabase')
    if not os.path.isdir(lpd):
        reportError(__file__,'Aux/MaterialsDatabase directory not found')
        return 1
    else:
        return lpd


def epsilon():
    return np.finfo(np.float).eps


def accuracyLim():
    acc_per = .0001
    return acc_per/100.


def parseToken(n,stringa,token=r'|'):
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


def checkPythonVersion():
    (major, minor, micro, releaselevel, serial) = sys.version_info
    if (major != 3 and major != 2) or (major == 2 and minor < 6):
        raise SystemExit("Payette requires Python >= 2.6\n")


def setupLogger(logfile,level,mode="w"):
    global loglevel,simlog
    loglevel = level
    simlog = open(logfile,mode)
    if mode == "w": simlog.write(pc.PC_INTRO + "\n")
    return


def parse_error(message):
    sys.exit("ERROR: {0}".format(message))


def read_input(user_input, user_cchar=None):
    """
       read a list of user inputs and return
    """

    if not user_input:
        parse_error("no user input sent to read_input")
        pass

    # comment characters
    cchars = ['#','$']
    if user_cchar is not None:
        cchars.append(user_cchar)

    # get all of the input blocks for the file
    input_sets = get_blocks(get_input_lines(user_input, cchars))

    # input_sets contains a list of all blocks in the file, parse it to make
    # sure that a simulation is given
    recognized_blocks = ("simulation", "boundary", "legs", "material",
                         "optimization", "permutation", "enumeration",
                         "mathplot", "name", "content", "extraction")
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


def get_blocks(user_input):
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

            new_block = {block_typ: {"name": block_nam, "content": []}}

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


def setupOutputFile(simdat, matdat, restart):

    global ofile,vtable,dtable
    if restart:
        ofile = open(simdat.OUTFILE,'a')
    else:
        ofile = open(simdat.OUTFILE,'w')

        # get the plot keys from the simvars
        plot_keys = simdat.plotKeys()
        plot_keys.extend(matdat.plotKeys())

        for head in plot_keys: ofile.write(textformat(head))
        ofile.write('\n')
        ofile.flush()
        pass

    if simdat.WRITE_VANDD_TABLE:
        vname = os.path.splitext(simdat.OUTFILE)[0] + ".vtable"
        dname = os.path.splitext(simdat.OUTFILE)[0] + ".dtable"

        # set up velocity and displacement table files
        if restart:
            vtable = open(vname,'a')
        else:
            vtable = open(vname,'w')
            default = ['time','v1','v2','v3']
            for head in default: vtable.write(textformat(head))
            vtable.write('\n')
            pass

        if restart:
            dtable = open(dname,'a')
        else:
            dtable = open(dname,'w')
            default = ['time','d1','d2','d3']
            for head in default: dtable.write(textformat(head))
            dtable.write('\n')
            dtable.write(textformat(0.))
            for j in range(3): dtable.write(textformat(0.))
            dtable.write("\n")
            pass
        pass

    writeAvailableDataToLog(simdat,matdat)

    return


def writeState(simdat,matdat):

    """ write the simulation and material data to the output file """
    plot_data = simdat.plotData()
    plot_data.extend(matdat.plotData())

    for x in plot_data:
        ofile.write(textformat(x))
        continue
    ofile.write('\n')

    return None


def writeMathPlot(simdat,matdat):

    """
    Write the $SIMNAME.math1 file for mathematica post processing
    """

    iam = "writeMathPlot"

    math1 = simdat.MATH1
    math2 = simdat.MATH2
    outfile = simdat.OUTFILE
    plotable = simdat.MATHPLOT_VARS
    parameter_table = matdat.PARAMETER_TABLE

    # get the plot keys from the simvars
    plot_keys = simdat.plotKeys()
    plot_keys.extend(matdat.plotKeys())

    # math1 is a file containing user inputs, and locations of simulation output
    # for mathematica to use
    with open( math1, "w" ) as f:
        # write out user given input
        for item in parameter_table:
            key = item["name"]
            val = "{0:12.5E}".format(item["initial value"]).replace("E","*^")
            f.write("{0:s}U={1:s}\n".format(key,val))
            continue

        # write out checked, possibly modified, input
        for item in parameter_table:
            key = item["name"]
            val = "{0:12.5E}".format(item["adjusted value"]).replace("E","*^")
            f.write("{0:s}M={1:s}\n".format(key,val))
            continue

        # write out user requested plotable output
        f.write('simdat = Delete[Import["{0:s}", "Table"],-1];\n'
                .format(outfile))
        sig_idx = None

        for i, item in enumerate(plot_keys):
            if item == "SIG11": sig_idx = i + 1
            f.write('{0:s}=simdat[[2;;,{1:d}]];\n'.format(item,i+1))
            continue

        # a few last ones...
        if sig_idx != None:
            pres=("-(simdat[[2;;,{0:d}]]".format(sig_idx) +
                  "+simdat[[2;;,{0:d}]]".format(sig_idx + 1) +
                  "+simdat[[2;;,{0:d}]])/3;".format(sig_idx+2))
            f.write('PRES={0}\n'.format(pres))
            pass
        f.write("lastep=Length[{0}]\n".format(simdat.getPlotKey("time")))

        pass

    # math2 is a file containing mathematica directives to setup default plots
    # that the user requested
    lowhead = [x.lower() for x in plot_keys]
    lowplotable = [x.lower() for x in plotable]
    with open( math2, "w" ) as f:
        f.write('showcy[{0},{{"cycle","time"}}]\n'
                .format(simdat.getPlotKey("time")))

        tmp = []
        for item in plotable:
            try:
                name = plot_keys[lowhead.index(item.lower())]
                f.write('grafhis[{0:s},"{0:s}"]\n'.format(name))
            except:
                tmp.append(item)
                continue
            continue

        if tmp:
            msg = (
                "requested plot variable{0:s} {1:s} not available for mathplot"
                .format("s" if len(tmp) > 1 else "", ", ".join(tmp)))
            logwrn(msg, caller=iam)
            pass

        pass

    return


def writeAvailableDataToLog(simdat,matdat):

    plotable = simdat.MATHPLOT_VARS
    lowplotable = [x.lower() for x in plotable]

    def write_plotable(idx,key,isplotable,name,val):
        # write to the logfile the available variables
        if isplotable: token = "plotable"
        else: token = "no request"
        writeToLog("{0:<3d} {1:<10s}: {2:<10s} = {3:<50s} = {4:12.5E}"
                   .format(idx,token,key,name,val))
        return

    # write to the log file what is plotable and not requested, along with inital
    # value
    writeToLog("Summary of available output")

    # time
    time = simdat.getData("time")
    key = simdat.getPlotKey("time")
    name = simdat.getPlotName("time")
    idx, val = 1, time
    write_plotable(idx,key,key.lower() in lowplotable,name,val)

    # stress
    sig = matdat.getData("stress")
    keys = matdat.getPlotKey("stress")
    names = matdat.getPlotName("stress")
    for i, val in enumerate(sig):
        idx = idx+1
        write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # dstress/dt
    dsigdt = matdat.getData("stress rate")
    keys = matdat.getPlotKey("stress rate")
    names = matdat.getPlotName("stress rate")
    for i, val in enumerate(dsigdt):
        idx = idx+1
        write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # strain
    eps = simdat.getData("strain")
    keys = simdat.getPlotKey("strain")
    names = simdat.getPlotName("strain")
    for i, val in enumerate(eps):
        idx = idx+1
        write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # sym(velocity gradient)
    d = simdat.getData("rate of deformation")
    keys = simdat.getPlotKey("rate of deformation")
    names = simdat.getPlotName("rate of deformation")
    for i, val in enumerate(d):
        idx = idx+1
        write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # deformation gradient
    defgrad = simdat.getData("deformation gradient")
    keys = simdat.getPlotKey("deformation gradient")
    names = simdat.getPlotName("deformation gradient")
    for i, val in enumerate(defgrad):
        idx = idx+1
        write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # extra variables
    if matdat.num_extra:
        ex = matdat.getData("extra variables")
        for i, val in enumerate(ex):
            idx = idx+1
            name = matdat.getExName(i)
            key = matdat.getPlotKey(name)
            write_plotable(idx,key,key.lower() in lowplotable,name,val)
            continue

    if simdat.EFIELD_SIM:
        # electric field
        efield = simdat.getData("electric field")
        keys = simdat.getPlotKey("electric field")
        names = simdat.getPlotName("electric field")
        for i, val in enumerate(efield):
            idx = idx+1
            write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
            continue

        # polarization
        polrzn = matdat.getData("polarization")
        keys = matdat.getPlotKey("polarization")
        names = matdat.getPlotName("polarization")
        for i, val in enumerate(polrzn):
            idx = idx+1
            write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
            continue

        # electric displacement
        edisp = matdat.getData("electric displacement")
        keys = matdat.getPlotKey("electric displacement")
        names = matdat.getPlotName("electric displacement")
        for i, val in enumerate(edisp):
            idx = idx+1
            write_plotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
            continue
        pass

    return


def closeFiles():
    try:
        ofile.write("\n")
        ofile.flush()
        ofile.close()
    except:
        pass
    try:
        simlog.close()
    except:
        pass


def writeVelAndDispTable(t0,tf,tbeg,tend,epsbeg,epsend,kappa):
    """
    NAME
        writeVelAndDispTable

    PURPOSE
        For each strain component, make a velocity and a displacement table and
        write it to a file. Useful for setting up simulations in other host
        codes.

    INPUT
        tbeg                time at beginning of step
        tend                time at end of step
        epsbeg              strain at beginning of step
        epsend              strain at end of step

    BACKGROUND
        python implementation of similar function in Rebecca Brannon's MED driver

    AUTHOR
        Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    dt = tend - tbeg
    dsp,vel = np.zeros(3),np.zeros(3)

#    # principal strains at beginning and end of leg
#    psbeg,v = la.eigh(np.array([ [ epsbeg[0], epsbeg[3], epsbeg[5] ],
#                                 [ epsbeg[3], epsbeg[1], epsbeg[4] ],
#                                 [ epsbeg[5], epsbeg[4], epsbeg[2] ] ] ))
#    psend,v = la.eigh(np.array([ [ epsend[0], epsend[3], epsend[5] ],
#                                 [ epsend[3], epsend[1], epsend[4] ],
#                                 [ epsend[5], epsend[4], epsend[2] ] ] ))

    psbeg,psend = epsbeg,epsend
    # determine average velocities that will ensure passing through the
    # exact stretches at each interval boundary.
    for j in range(3):
        if kappa != 0.:
            # Seth-Hill generalized strain is defined
            # strain = (1/kappa)*[(stretch)^kappa - 1]
            lam0 = (psbeg[j]*kappa + 1.)**(1./kappa)
            lam = (psend[j]*kappa + 1.)**(1./kappa)

        else:
            # In the limit as kappa->0, the Seth-Hill strain becomes
            # strain = ln(stretch).
            lam0 = math.exp(psbeg[j])
            lam = math.exp(psend[j])
            pass
        dsp[j] = lam - 1

        # Except for logarithmic strain, a constant strain rate does NOT
        # imply a constant boundary velocity. We will here determine a
        # constant boundary velocity that will lead to the correct value of
        # stretch at the beginning and end of the interval.
        vel[j] = (lam - lam0)/dt
        continue

    # displacement
    dtable.write(textformat(tend))
    for j in range(3): dtable.write(textformat(dsp[j]))
    dtable.write("\n")

    # jump discontinuity in velocity will be specified by a VERY sharp change
    # occuring from time = tjump - delt to time = tjump + delt
    delt = tf*1.e-9
    if tbeg > t0: tbeg += delt
    if tend < tf: tend -= delt

    vtable.write(textformat(tbeg))
    for j in range(3): vtable.write(textformat(vel[j]))
    vtable.write("\n")
    vtable.write(textformat(tend))
    for j in range(3): vtable.write(textformat(vel[j]))
    vtable.write("\n")

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

    if isinstance(x,(float,int,str,bool)): return x

    result = []
    for el in x:
        #if isinstance(el, (list, tuple)):
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
            pass
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
    return (os.path.splitext(os.path.basename(py_file))[0],
            [os.path.dirname(py_file)])


def get_module_name(py_file):
    """ return the module name of py_file """
    return get_module_name_and_path(py_file)[0]


def begmes(msg, pre="", end="  "):
    print("{0}{1}...".format(pre, msg), end=end)
    return


def endmes(msg, pre="", end="\n"):
    print("{0}{1}".format(pre, msg), end=end)
    return


def loginf(msg, pre="", end="\n", caller=None):
    if caller is not None:
        msg = "{0} [reported by {1}]".format(msg, caller)
    print("{0}INFO: {1}".format(pre, msg), end=end)
    return


def logmes(msg, pre="", end="\n", caller=None):
    if caller is not None:
        msg = "{0} [reported by {1}]".format(msg, caller)
    print("{0}{1}".format(pre, msg), end=end)
    return


def logwrn(msg, pre="", end="\n", caller=None):
    if caller is not None:
        msg = "{0} [reported by {1}]".format(msg, caller)

    print("{0}WARNING: {1}".format(pre, msg), end=end)
    return


def logerr(msg, pre="", end="\n", caller=None):
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
            dist_from_pt0 = dist_pt_to_pt(set1x[jdx],set1y[jdx],set2x[idx],set2y[idx])
            dist_from_pt1 = dist_pt_to_pt(set1x[kdx],set1y[kdx],set2x[idx],set2y[idx])

            # use dot(a,b)/(mag(a)*mag(b)) = cos(theta) to find the distance from the line.
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
        fobj.write("\n".join(inp_dict["content"]) + "\n")
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


def get_input_lines(user_input, cchars):
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

    all_input = []
    iline = 0

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
        line = user_input[iline].strip()

        # skip blank and comment lines
        if not line.split() or line[0] in cchars:
            iline += 1
            continue

        # remove inline comments
        for cchar in cchars:
            line = line.split(cchar)[0]

        # check for inserts
        if line.split()[0] in insert_kws:
            inserted_file = True
            insert = " ".join(line.split()[1:])
            if not os.path.isfile(insert):
                parse_error("inserted file '{0:s}' not found".format(insert))

            insert_lines = open(insert, "r").readlines()
            for insert_line in insert_lines:
                if not insert_line.split() or insert_line[0] in cchars:
                    continue
                for cchar in cchars:
                    insert_line = insert_line.split(cchar)[0]
                all_input.append(insert_line.strip())

        else:
            all_input.append(line)

        iline += 1

        continue

    return all_input



def write_extraction(simdat, matdat):
    """ write out the requested extraction """

    iam = "write_extraction"

    # make sure that any extraction arguments are in the output file
    lowhead = [x.lower() for x in simdat.plotKeys() + matdat.plotKeys()]
    lowextract = [x.lower() for x in simdat.EXTRACTION]

    exargs = [simdat.OUTFILE, "--silent", "--xout"]
    for idx, item in enumerate(lowextract):
        if not item[1:].isdigit() and item[1:] not in lowhead:
            if not [x for x in item if x in "+/*-"]:
                logwrn("ignoring bad extraction request " + item, caller=iam)
                continue
        exargs.append(item)
        continue

    error = pe.extract(exargs)

    return


def check_if_test_dir(dir_path):
    """ check if dir_path has __test_dir__.py """
    has__test_dir__ = False
    for dirnam, dirs, files in os.walk(dir_path):
        if "__test_dir__.py" in files:
            has__test_dir__ = True
            break
        continue

    return has__test_dir__
