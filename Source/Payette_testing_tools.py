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

import sys
import imp
import os
import linecache
import numpy as np
import math
import time

from Source.Payette_utils import *

def findTests(dirname,names,mykeys,mynegkeys,spectests):
    '''
    NAME
       findTests

    PURPOSE
       determine if any python files in names are Payette tests

    INPUT
       dirname: directory name where the files in names reside
       names: list of files in dirname
       mykeys - list of requested kw
       mynegkeys - list of negated kw
       spectests - list of specific tests to run

    OUTPUT
       tests: list of files that are
              conform - conforming Payette python test files
              nonconform - nonconforming Payette python test files

    AUTHORS
       Tim Fuller, Sandia, National Laboratories, tjfulle@sandia.gov
       M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    '''
    conform,nonconform = [],[]
    # mykeys are user specified keywords
    if mykeys: mykeys = [x.lower() for x in mykeys]
    if mynegkeys: mynegkeys = [x.lower() for x in mynegkeys]
    if spectests: spectests = [x.lower() for x in spectests]
    mykeys.sort()
    mynegkeys.sort()

    if '.svn' in dirname:
        return [], []

    for name in names:
        fpath = os.path.join(dirname,name)
        fbase,fext = os.path.splitext(name)

        if fext != ".py" or fbase[0] == "." or fbase == "__init__":
            # only want nonhidden python files
            continue

        # correct path and get full path to python file
        if dirname not in sys.path: sys.path.append(dirname)

        # load module
        with open(fpath) as f:
            fmod = imp.load_module(fbase,f,fpath,('py','U',imp.PY_SOURCE))
            pass

        if hasattr(fmod,'Payette_test') and fmod.Payette_test:

            # check that the module has all required attributes
            hasallattr,missingattr = checkTestAttributes(fmod,name)

            if missingattr:
                nonconform.append([name,missingattr]) # missing attributes
                continue

            fkeys = [x.lower() for x in fmod.keywords]

            if spectests:
                # user asked for specific tests
                if fmod.name.lower() in spectests: conform.append(fpath)
                else: pass

            elif not mykeys and not mynegkeys:
                # if user has not specified any kw or kw negations
                # append to conform
                conform.append(fpath)

            else:
                # user specified a kw or kw negation
                # find the intersection of fkeys and my{,neg}keys
                t0 = time.time()
                hasnegkey = bool([x for x in mynegkeys if x in fkeys])
                hasreqkey = True
                reqkeyl = [x for x in mykeys if x in fkeys]
                reqkeyl.sort()
                hasreqkey = reqkeyl == mykeys
                if hasnegkey: pass # kw negation takes precident
                elif not mykeys: conform.append(fpath)
                elif hasreqkey: conform.append(fpath)
                pass

            continue

    return conform,nonconform

def checkTestAttributes(module,filename):
    '''
    NAME
       checkTestAttributes

    PURPOSE
       check that the benchmark module given in filename contains the required
       attributes

    AUTHORS
       M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov
       Tim Fuller, Sandia, National Laboratories, tjfulle@sandia.gov
    '''

    missing = []
    reqattr = ['name','infile','outfile','keywords','executable',
               'runcommand','owner','date','description','runTest']
    for attr in reqattr:
        if not hasattr(module,attr): missing.append(attr)
        continue
    if missing: return False, ', '.join(missing)

    # check some attributes
    if module.infile and not os.path.isfile(module.infile):
        missing.append('infile %s does not exist'%module.infile)
        pass
    reqlkw = ['fast','medium','long']
    haslkw = False
    for kw in reqlkw:
        if kw in module.keywords:
            haslkw = True
            break
        continue
    if not haslkw: missing.append('must specify one of %s in keywords'%str(reqlkw))

    reqtkw = ['verification','validation','prototype','regression']
    hastkw = False
    for kw in reqtkw:
        if kw in module.keywords:
            hastkw = True
            break
        continue
    if not hastkw: missing.append('must specify one of %s in keywords'%str(reqtkw))

    if missing: return False, ', '.join(missing)
    return True,None

def getHeader(f):
    """
    NAME:
        getHeader

    PURPOSE:
        Reads the header row of a file f which (presumably) has the names of the
        fields in the output file

    OUTPUT
        n dimensional list of names of each data field (columns)

    AUTHORS:
        Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """
    return linecache.getline(f,1).split()

def readData(f):
    """
    NAME:
        readData

    PURPOSE:
        Reads in a whitespace-delimed data file f. It is assumed that the first
        line contains text (the name of the column). All other lines contain
        floats.

    OUTPUT
        m x n dimensional numpy ndarray where m is the number of data points
        (rows) and n the number of data fields (columns)

    AUTHORS:
        Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """
    return np.loadtxt(f,skiprows=1)

def computeRMS(set1x,set1y,set2x,set2y,step=1):
    """
    NAME:
        computeRMS( list(float)x4, step=int )

    PURPOSE:
        Determines the RMS of the error between two piecewise functions. In
        almost all cases, the abscissa variables (set1x and set2x) will be time.
        Currently, the abscissa variables are required to be monotonically
        increasing (for easier computing). Interpolations are used to compare
        data points where data is missing in the other set.

        'set2{x,y}' is compared against 'set1{x,y}'.

        len(set1x) == len(set1y)
        len(set2x) == len(set2y)

        The optional 'step' argument is used for large data sets. This gives
        the benchmark developer more control over how the error is computed
        (choosing accuracy vs. speed).

    AUTHORS:
        M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov

    CHANGELOG:
        20 Jun 2011   mswan   Created
    """

    # check the lengths of the arrays passed
    lset1x,lset1y,lset2x,lset2y = map(len,[set1x,set1y,set2x,set2y])
    if lset1x != lset1y: sys.exit("len(set1x) != len(set1y)")
    if lset2x != lset2y: sys.exit("len(set2x) != len(set2y)")

    # Use a shortcut if the lengths of the x and y data sets are the same. Also,
    # before using the shortcut, do a quick check by computing the RMS on every
    # 10th x value. The RMS check on the x-values only gets computed if
    # len(set1y) == len(set2y). This check allows for the possibility of the
    # lengths of the data to be equal but for the spacing to be not equal.
    if (lset1y == lset2y and
        np.sum((set1x[::10]-set2x[::10])**2) < 1.0e-6*np.amax(set1x)):
        return computeFastRMS(set1y,set2y)

    else:

        # compute the running square of the difference
        error = 0.0
        for i in range(0,lset2x,step):

            for j in range(0,lset1x-1,step):

                # mss: Very expensive look. Optimize. if set2x is between set1x
                # pair, do linear interpolation
                if set1x[j] <= set2x[i] <= set1x[j+1]:
                    x0 = set1x[j];   y0 = set1y[j]
                    x1 = set1x[j+1]; y1 = set1y[j+1]
                    f = y0 + (set2x[i]-x0)*(y1-y0)/(x1-x0)

                    # Add the square of the difference to error
                    error += (set2y[i] - f)**2
                    break
                continue
            continue
        pass
    rmsd = math.sqrt(error/float(lset1x))
    dnom = abs(np.amax(set1y) - np.amin(set1y))
    nrmsd = rmsd/dnom if dnom >= 2.e-16 else rmsd
    return rmsd,nrmsd

def computeFastRMS(gold,out):
    """
    NAME:
        computeFastRMS

    PURPOSE:
        Determines the RMS of the error between two piecewise functions of same
        length. This should only be called if the timestep sizes of the gold
        simulation and the simulation that produced out are the same. This
        function makes no attempt to make that determination, it is up to the
        caller.

    AUTHORS:
       Tim Fuller, Sandia, National Laboratories, tjfulle@sandia.gov

    """
    rmsd = math.sqrt(np.sum((gold - out)**2)/float(len(gold)))
    dnom = abs(np.amax(gold) - np.amin(gold))
    nrmsd = rmsd/dnom if dnom >= np.finfo(np.float).eps else rmsd
    return rmsd,nrmsd
