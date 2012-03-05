#!/usr/bin/env python
import sys,subprocess,linecache,os
import time
import math
import numpy as np

from tests_common import *

Payette_test = True
name = 'kayenta-ctps-txci'
fdir = os.path.dirname(os.path.realpath(__file__))
infile = '%s.inp'%(os.path.join(fdir,name))
outfile = '%s.out'%(name)
keywords = ['medium','verification','kayenta','ctps','med']
executable = runPayette
runcommand = [executable,'--no-restart',infile]
baseline = '%s.gold'%(os.path.join(fdir,name))
medinpfile = '%s.MoS'%(name)
owner = "Tim Fuller Swan"
date = 'June 25, 2011'
description = '''
    Test of principal stress cut off

    The baseline file is from an equivalent simulation performed in the MED driver.
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
    difftol = 5.e-6
    failtol = 1.e-3

    # open the log file
    LOG = open('%s.diff'%name,'w')

    # read in results
    outheader,goldheader = ptt.getHeader(outfile),ptt.getHeader(baseline)
    gold = ptt.readData(baseline)
    out = ptt.readData(outfile)

    # check that time is same
    rmsd,nrmsd = ptt.computeFastRMS(gold[:,0],out[:,0])
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
    toskip = ['KAPPA','EOS1','EOS2','EOS3','EOS4','PLROOTJ2','SNDSP','ENRGY','RHO','TMPR']
    tocompare = [x for x in outheader if x in goldheader and x not in toskip]
    for val in tocompare:
        gidx = goldheader.index(val)
        oidx = outheader.index(val)
        rmsd,nrmsd = ptt.computeRMS(gold[:,0],gold[:,gidx],out[:,0],out[:,oidx])

        # For good measure, print both the RMSD and normalized RMSD
        LOG.write("{0} error:\n".format(val))
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
