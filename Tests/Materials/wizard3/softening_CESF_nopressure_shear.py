#!/usr/bin/env python
import sys
import subprocess
import linecache
import os
import time
import math
import numpy as np

from tests_common import *

Payette_test = True
name = os.path.basename(os.path.realpath(__file__))
name = os.path.splitext(name)[0]
keywords = ['kayenta','medium','verification','cesf']
owner = 'Scot Swan'
date = '03 Feb 2012'

fdir = os.path.dirname(os.path.realpath(__file__))
infile = '%s.inp'%(os.path.join(fdir,name))
outfile = '%s.out'%(name)
executable = runPayette
runcommand = [executable,'--no-writeprops','--no-restart','--proportional',infile]

def get_strain_at_failure():
    eta = 0.0
    xi  = 0.0
    C1 = 0.21 # Strain at failure
    return C1

description = '''
   Perform verification of the Constant Equivalent Strain-to-Failure model.
   This test runs a simulation using parameters for Al 2024-T351 with
   proportional loading (constant ratio of stresses) defined by:
       Stress Triaxiality = 0.0
       Lode Parameter =  0.0
   which correlate to a state of shear that has no hydrostatic pressure.

   This test only looks at ACCSTRAIN verses COHER.

   For reference:
       Stress Triaxiality = sigma_m / sigma_eqv = I1/(3*ROOT3*ROOTJ2)
       Lode Parameter     = sin(3\lodeangle)
'''

def performCalcs():
    '''
    NAME
       performCalcs

    PURPOSE
       run the benchmark problem

    OUTPUT
       stat   0: problem ran successfully
           != 0: problem did not run successfully
    '''
    # run the problem
    with open('%s.echo'%(name),'w') as f:
        run = subprocess.Popen(runcommand,stdout=f,stderr=subprocess.STDOUT)
        run.wait()
        pass
    return run.returncode

def analyzeTest():
    '''
    NAME
       analyzeTest

    PURPOSE
       analyze the test results from performCalcs

    OUTPUT
       stat  0: passed
             1: diffed
             2: failed
       message
    '''
    import Payette_testing_tools as ptt

    # tolerances
    difftol = 1.e-2
    failtol = 2.e-2

    # open the log file
    LOG = open('%s.diff'%name,'w')

    # read in results
    outheader = ptt.getHeader(outfile)
    out = ptt.readData(outfile)

    # compare results
    LOG.write("\n\nTOLERANCES:\n")
    LOG.write("diff tol: {0:10e}\n".format(difftol))
    LOG.write("fail tol: {0:10e}\n".format(failtol))
    errors = []

    # Get the indicies for COHER and ACCSTRAIN. Then, verify COHER does
    # drop below 0.5. If it does not, then this test is a FAIL.
    coher_idx     = outheader.index("COHER")
    accstrain_idx = outheader.index("ACCSTRAIN")
    fail_idx = -1
    for i in range(0,len(out[:,coher_idx])):
        if out[i,coher_idx] < 0.5:
            fail_idx = i
            break

    if fail_idx == -1:
        LOG.write("COHER did not drop below 0.5.\n")
        LOG.write("Final value of COHER: {0}\n".format(out[-1,coher_idx]))
        return 2

    # Perform an interpolation between COHER-ACCSTRAIN sets to find the
    # ACCSTRAIN when COHER=0.5 . Then compute the absolute and relative
    # errors.
    x0, y0 = out[fail_idx-1,coher_idx],out[fail_idx-1,accstrain_idx]
    x1, y1 = out[fail_idx,  coher_idx],out[fail_idx,  accstrain_idx]

    strain_f = y0 + (0.5-x0)*(y1-y0)/(x1-x0)
    abs_err = abs(strain_f-get_strain_at_failure())
    rel_err = abs_err/abs(max(strain_f,get_strain_at_failure()))

    # Write to output.
    LOG.write("COHER absolute error: {0}\n".format(abs_err))
    LOG.write("COHER relative error: {0}\n".format(rel_err))
    if rel_err >= failtol:
        LOG.write("\n{0:=^72s}\n".format(" FAIL "))
        LOG.close()
        return 2
    elif rel_err >= difftol:
        LOG.write("\n{0:=^72s}\n".format(" DIFF "))
        LOG.close()
        return 1
    else:
        LOG.write("\n{0:=^72s}\n".format(" PASS "))
        LOG.close()
        return 0

def runTest():
    perform_calcs = performCalcs()
    if perform_calcs != 0: return 2
    return analyzeTest()

if __name__ == '__main__':
    import time
    if '--cleanup' in sys.argv:
        for ext in ['out','diff','log','prf','pyc','echo']:
            try: os.remove('%s.%s'%(name,ext))
            except: pass
            continue
        pass
    else:
        t0 = time.time()
        print('%s RUNNING'%name)
        perform_calcs = performCalcs()
        dtp = time.time()-t0
        if perform_calcs != 0:
            print('%s FAILED TO RUN TO COMPLETION'%(name))
            sys.exit()
            pass
        print('%s FINISHED'%name)
        print('%s ANALYZING'%name)
        t1 = time.time()
        run_test = analyzeTest()
        dta = time.time()-t1
        if run_test == 0:
            print('%s PASSED(%fs)'%(name,dtp+dta))
        elif run_test == 1:
            print('%s DIFFED(%fs)'%(name,dtp+dta))
        else:
            print('%s FAILED(%fs)'%(name,dtp+dta))
