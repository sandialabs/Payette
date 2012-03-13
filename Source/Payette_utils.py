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
import numpy as np
import numpy.linalg as la

from Toolset.Payette_config import *

debug = False
wcount, wthresh, wmax = 0, 0, 1000

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
    def __init__(self,msg):
        if 'MIG' in msg: sys.tracebacklimit = 3
        l = 79 # should be odd number
        st,stsp = '*'*l + '\n', '*' + ' '*(l-2) + '*\n'
        psf = 'Payette simulation failed'
        ll = (l - len(psf) - 2)/2
        psa = '*' + ' '*ll + psf + ' '*ll + '*\n'
        head = '\n\n' + st + stsp + psa + stsp + st
        Exception.__init__(self,head+msg)

def reportError(f,msg):
    f = fixFLength(f)
    msg = '{0} (reported from [{1}])'.format(msg,f)
    raise PayetteError(msg)
    return

def reportWarning(f,msg,limit=False):
    global wcount, wthresh, wmax
    if limit:
        if wcount <= wmax/200-1: wthresh = wcount+1
        elif wcount > wmax/200 and wcount < wmax/100: wthresh = wmax/100
        elif wcount > wmax/100 and wcount < wmax/10: wthresh = wmax/10
        else: wthresh = wmax
        wcount += 1
        if wcount != wthresh: return
        pass
    msg = 'WARNING: {0} (reported from [{1}])\n'.format(msg,f)
    simlog.write(msg)
    sys.stderr.write(msg)
    return

def writeMessage(f,msg):
    sys.stdout.write("INFO: {0:s}\n".format(msg))
    return

def writeWarning(f,msg):
    sys.stdout.write("WARNING: {0:s}\n".format(msg))
    return

def reportMessage(f,msg):
#    msg = 'INFO: {0} (reported from [{1}])\n'.format(msg,f)
    msg = 'INFO: {0}\n'.format(msg)
    simlog.write(msg)
    if loglevel > 0: sys.stderr.write(msg)
    return

def writeToLog(msg):
    msg = "{0:s}\n".format(msg)
    simlog.write(msg)
    return

def migMessage(msg):
    reportMessage('MIG',msg)
    return

def migError(msg):
    msg = ' '.join([x for x in msg.split(' ') if x])
    reportError('MIG',msg)
    return

def migWarning(msg):
    reportWarning('MIG',msg)
    return

def fixFLength(f):
    if not os.path.isfile(f): return f
    f = os.path.split(f)[1]
    basename,fext = os.path.splitext(f)
    return basename

def payetteParametersDir():
    lpd = os.path.join(Payette_Aux,'MaterialsDatabase')
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

def shutdownLogger():
    simlog.close()
    return

def setupLogger(logfile,level):
    global loglevel,simlog
    loglevel = level
    simlog = open(logfile,"w")
    return

def readUserInput(user_input,user_cchar=None):
    """
       read a list of user inputs and return
    """
    # comment characters
    cchars = ['#','$']
    if user_cchar: cchars.append(user_cchar)

    # read file, sans comments, into a list
    all_input = []
    if not user_input:
        parse_error("no user input sent to readUserInput")
        pass
    for line in user_input:

        aline = line.strip()

        # skip blank and comment lines
        if not aline or aline[0] in cchars: continue

        # remove inline comments
        aline = removeComments(aline,cchars)

        # check for inserts
        inclines,inserts = checkForInserts(aline), []
        if inclines:
            for incline in inclines:
                incline = removeComments(incline,cchars)
                toomany = checkForInserts(incline)
                if toomany: parse_error("cannot have includes in includes")
                if incline: inserts.append(incline)
                continue
            all_input.extend(inserts)
        else: all_input.append(aline)

        continue

    # parse the list and put individual simulations into the user_dict
    user_dict = {}
    i, maxit = 0, len(all_input)
    while True:
        simulation,simid = findBlock(all_input,'simulation')
        if simulation and not simid:
            parse_error('did not find simulation name.  Simulation block '
                         'must be for form:\n'
                         '\tbegin simulation simulation name ... end simulation')

        elif simulation:
            simkey = simid.replace(' ','_')
            user_dict[simkey] = simulation
        else: break
        if i >= maxit: break
        i += 1
        continue
    return user_dict

def removeComments(aline,cchars):
    for cchar in cchars:
        while cchar in aline:
            i = aline.index(cchar)
            aline = aline[0:i].strip()
            continue
        continue
    return aline

def checkForInserts(aline):
    magic = ["insert","include"]
    qmarks = ['"',"'"]
    if not aline.strip(): pass
    elif aline.split()[0].lower() in magic:
        f = ' '.join(aline.split()[1:]).strip()
        if f[0] in qmarks:
            if f[-1] not in qmarks or f[-1] != f[0]:
                parse_error('inconsistent quotations around include file at "{0}"'
                            .format(aline))
            else: f = f[1:len(f)-1]
            pass

        f = os.path.realpath(f)

        try:
            return [line.strip() for line in open(f,'r').readlines()]
        except IOError as error:
            if error.errno == 2:
                parse_error('insert file {0} not found'.format(f))
            else: raise
        except: raise

    else: pass

    return None

def findBlock(ilist,keyword):
    '''
    NAME
       findBlock

    PURPOSE
       from the user input in <ilist>, find and return the <keyword> input
       block

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''
    kwd = str(keyword)
    index_beg = None
    index_end = None
    bid = None

    for item in ilist:
        if item[:len('begin {0}'.format(kwd))].strip() == 'begin {0}'.format(kwd):
            index_beg = ilist.index(item)+1
            bid = item[len('begin {0}'.format(kwd)):].strip()
        else: pass
        if index_beg: break
        continue
    for item in ilist[index_beg:]:
        if item.strip() == 'end {0}'.format(kwd):
            index_end = ilist.index(item)
        elif 'end {0}'.format(kwd) in item.strip():
            index_end = ilist.index(item)
        else: pass
        if index_end: break
        continue
    if not index_beg or not index_end:
        return None,None
    else:
        block = ilist[index_beg:index_end]
        del ilist[index_beg-1:index_end+1]
        pass
    return block,bid

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

def setupOutputFile(simdat,matdat,restart):

    global ofile,vtable,dtable,plot_keys
    if restart: ofile = open(simdat.outfile,'a')
    else: ofile = open(simdat.outfile,'w')

    # get the plot keys from the simvars
    plot_keys = simdat.plotKeys()
    plot_keys.extend(matdat.plotKeys())

    #    outfile.write('# ')
    for head in plot_keys: ofile.write(textformat(head))
    ofile.write('\n')

    if simdat.write_vandd_table:
        vname = os.path.splitext(simdat.outfile)[0] + ".vtable"
        dname = os.path.splitext(simdat.outfile)[0] + ".dtable"

        # set up velocity and displacement table files
        if restart: vtable = open(vname,'a')
        else: vtable = open(vname,'w')
        default = ['time','v1','v2','v3']
        for head in default: vtable.write(textformat(head))
        vtable.write('\n')

        if restart: dtable = open(dname,'a')
        else: dtable = open(dname,'w')
        default = ['time','d1','d2','d3']
        for head in default: dtable.write(textformat(head))
        dtable.write('\n')
        dtable.write(textformat(0.))
        for j in range(3): dtable.write(textformat(0.))
        dtable.write("\n")
        pass

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

    math1 = simdat.math1
    math2 = simdat.math2
    outfile = simdat.outfile
    plotable = simdat.mathplot_vars
    parameter_table = matdat.parameter_table

    # private functions
    def writePlotable(idx,key,isplotable,name,val):
        # write to the logfile the available variables
        if isplotable: token = "plotable"
        else: token = "no request"
        writeToLog("{0:<3d} {1:<10s}: {2:<10s} = {3:<50s} = {4:12.5E}"
                   .format(idx,token,key,name,val))
        return

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
        f.write('simdat = Delete[Import["{0:s}", "Table"],-1];\n'.format(outfile))
        sig_idx = None
        for i, item in enumerate(plot_keys):
            if item == "SIG11": sig_idx = i + 1
            f.write('{0:s}=simdat[[2;;,{1:d}]];\n'.format(item,i+1))
            continue

        # a few last ones...
        if sig_idx != None:
            pres=('-(simdat[[2;;,{0:d}]]+simdat[[2;;,{1:d}]]+simdat[[2;;,{2:d}]])/3;'
                  .format(sig_idx,sig_idx+1,sig_idx+2))
            f.write('PRES={0}\n'.format(pres))
            pass
        f.write("lastep=Length[{0}]\n".format(simdat.getPlotKey("time")))

        pass

    # math2 is a file containing mathematica directives to setup default plots
    # that the user requested
    lowhead = [x.lower() for x in plot_keys]
    lowplotable = [x.lower() for x in plotable]
    with open( math2, "w" ) as f:
        f.write('showcy[{0},{{"cycle","time"}}]\n'.format(simdat.getPlotKey("time")))

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
            msg = ("requested plot variable{0:s} {1:s} not available for mathplot"
                   .format("s" if len(tmp) > 1 else "", ", ".join(tmp)))
            reportWarning(__file__,msg)
            pass

        pass

    # write to the log file what is plotable and not requested, along with inital
    # value
    # time
    time = simdat.getData("time")
    key = simdat.getPlotKey("time")
    name = simdat.getPlotName("time")
    writeToLog("Summary of available output")
    idx, val = 1, time
    writePlotable(idx,key,key.lower() in lowplotable,name,val)

    # stress
    sig = matdat.getData("stress")
    keys = matdat.getPlotKey("stress")
    names = matdat.getPlotName("stress")
    for i, val in enumerate(sig):
        idx = idx+1
        writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # dstress/dt
    dsigdt = matdat.getData("stress rate")
    keys = matdat.getPlotKey("stress rate")
    names = matdat.getPlotName("stress rate")
    for i, val in enumerate(dsigdt):
        idx = idx+1
        writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # strain
    eps = simdat.getData("strain")
    keys = simdat.getPlotKey("strain")
    names = simdat.getPlotName("strain")
    for i, val in enumerate(eps):
        idx = idx+1
        writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # sym(velocity gradient)
    d = simdat.getData("rate of deformation")
    keys = simdat.getPlotKey("rate of deformation")
    names = simdat.getPlotName("rate of deformation")
    for i, val in enumerate(d):
        idx = idx+1
        writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # deformation gradient
    defgrad = simdat.getData("deformation gradient")
    keys = simdat.getPlotKey("deformation gradient")
    names = simdat.getPlotName("deformation gradient")
    for i, val in enumerate(defgrad):
        idx = idx+1
        writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
        continue

    # extra variables
    ex = matdat.getData("extra variables")
    for i, val in enumerate(ex):
        idx = idx+1
        name = matdat.getExName(i)
        key = matdat.getPlotKey(name)
        writePlotable(idx,key,key.lower() in lowplotable,name,val)
        continue

    if matdat.electric_field_model:
        # electric field
        efield = simdat.getData("electric field")
        keys = simdat.getPlotKey("electric field")
        names = simdat.getPlotName("electric field")
        for i, val in enumerate(efield):
            idx = idx+1
            writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
            continue

        # polarization
        polrzn = matdat.getData("polarization")
        keys = matdat.getPlotKey("polarization")
        names = matdat.getPlotName("polarization")
        for i, val in enumerate(polrzn):
            idx = idx+1
            writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
            continue

        # electric displacement
        edisp = matdat.getData("electric displacement")
        keys = matdat.getPlotKey("electric displacement")
        names = matdat.getPlotName("electric displacement")
        for i, val in enumerate(edisp):
            idx = idx+1
            writePlotable(idx,keys[i],keys[i].lower() in lowplotable,names[i],val)
            continue
        pass

    return

def closeFiles():
    ofile.write("\n")
    ofile.close()
    simlog.close()

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


def parse_error(message):
    sys.exit("ERROR: {0}".format(message))

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
