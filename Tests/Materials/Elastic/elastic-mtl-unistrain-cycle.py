#!/usr/bin/env python

import sys,subprocess,linecache,os
import math
import numpy as np
import time

from tests_common import *

Payette_test = True
name = 'elastic-mtl-unistrain-cycle'
fdir = os.path.dirname(os.path.realpath(__file__))
infile = '%s.inp'%(os.path.join(fdir,name))
outfile = '%s.out'%(name)
keywords = ['fast','verification','elastic','uniaxial strain']
executable = runPayette

runcommand = [executable,'--no-writeprops','--no-restart',infile]
baseline = '%s.gold'%(os.path.join(fdir,name))
owner = 'Tim Fuller'
date = 'June 1, 2011'
description = '''
    In this test, an elastic material is cycled through a uniaxial
    strain deformation path in the following steps:

    Step 1 (t=0.-1.): prescribed uniaxial strain to eps_11 = 0.1
    Step 2 (t=1.-2.): prescribed uniaxial strain to eps_11 = 0.0
    Step 3 (t=2.-3.): prescribed stress resulting in same strain
                      path as Step 1
    Step 4 (t=3.-4.): prescribed stress resulting in same strain
                      path as Step 2
    Step 5 (t=4.-5.): prescribed strain rate resulting in same strain
                      path as Step 1
    Step 6 (t=5.-6.): prescribed strain rate resulting in same strain
                      path as Step 2
    Step 7 (t=6.-7.): prescribed stress rate resulting in same strain
                      path as Step 1
    Step 8 (t=7.-8.): prescribed stress rate resulting in same strain
                      path as Step 2
    Step 9 (t=8.-9.): prescribed def gradient resulting in same strain
                      path as Step 1
    Step 10 (t=9.-10.): prescribed def gradient resulting in same strain
                        path as Step 2
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
    '''
    import Payette_testing_tools as ptt

    # local variables
    difftol = 1.e-6
    failtol = 1.e-3

    # open the log file
    LOG = open('%s.diff'%name,'w')

    # read in results
    header = ptt.getHeader(baseline)
    gold = ptt.readData(baseline)
    out = ptt.readData(outfile)

    # check that number of fields and columns the same
    if out.shape != gold.shape:
        LOG.write('number of fields not the same between the current and baseline')
        LOG.write("\n{0:=^72s}\n".format(" FAIL "))
        LOG.close()
        return 2

    nfields,ncols = gold.shape

    # check that time is same
    rmsd, nrmsd = ptt.computeFastRMS(gold[:,0],out[:,0])
    if nrmsd > np.finfo(np.float).eps:
        LOG.write('time step error between the current and baseline')
        LOG.write("\n{0:=^72s}\n".format(" FAIL "))
        LOG.close()
        return 2

    # compare results
    LOG.write("\n\nTOLERANCES:\n")
    LOG.write("diff tol: {0:10e}\n".format(difftol))
    LOG.write("fail tol: {0:10e}\n".format(failtol))
    errors = []

    for i in range(1,ncols):
        rmsd,nrmsd = ptt.computeFastRMS(gold[:,i],out[:,i])

        # For good measure, print both the RMSD and normalized RMSD
        if nrmsd >= failtol: stat = 'FAIL'
        elif nrmsd >= difftol: stat = 'DIFF'
        else: stat = 'PASS'
        LOG.write("{0}:{1} error:\n".format(header[i],stat))
        LOG.write("    Unscaled: {0:.10e}\n".format(rmsd))
        LOG.write("    Scaled:   {0:.10e}\n".format(nrmsd))
        errors.append(nrmsd)
        continue

    if max(errors) >= failtol:
        LOG.write("\n{0:=^72s}\n".format(" FAIL "))
        LOG.close()
        return 2
    elif max(errors) >= difftol:
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
