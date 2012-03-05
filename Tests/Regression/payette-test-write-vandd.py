#!/usr/bin/env python
import sys,subprocess,linecache,os
import time
import math
import numpy as np

from tests_common import *

Payette_test = True
name = "payette-test-write-vandd"
keywords = ["payette","regression","fast","vtable","dtable"]
owner = "Tim Fuller"
date = "February 26, 2012"

fdir = os.path.dirname(os.path.realpath(__file__))
infile = "%s.inp"%(os.path.join(fdir,name))
outfile = "%s.out"%(name)
executable = runPayette
runcommand = [executable,"--no-restart","--no-writeprops","-w",infile]
baseline = ["%s.dgold"%(os.path.join(fdir,name)),
            "%s.vgold"%(os.path.join(fdir,name))]

description = """
    Test of velocity and displacement table creation
"""

def performCalcs():
    """
    NAME
       performCalcs

    PURPOSE
       run the benchmark problem

    OUTPUT
       stat   0: problem ran successfully
           != 0: problem did not run successfully
    """
    # run the problem
    with open("%s.echo"%(name),"w") as f:
        run = subprocess.Popen(runcommand,stdout=f,stderr=subprocess.STDOUT)
        run.wait()
        pass
    return run.returncode

def analyzeTest():
    """
    NAME
       analyzeTest

    PURPOSE
       analyze the test results from performCalcs

    OUTPUT
       stat  0: passed
             1: diffed
             2: failed
       message
    """
    import difflib

    errors = 0

    # open the log file
    LOG = open('%s.diff'%name,'w')

    # compare the displacement table
    dtable = open(name + ".dtable").readlines()
    dgold = open(baseline[0]).readlines()
    if dtable != dgold:
        ddiff = difflib.ndiff(dtable,dgold)
        errors += 1
        LOG.write("ERROR: Displacement tables diffed:\n\n")
        LOG.write("".join(ddiff))
    else:
        LOG.write("dtable PASSED")

    # compare the velocity table
    vtable = open(name + ".vtable").readlines()
    vgold = open(baseline[1]).readlines()
    if vtable != vgold:
        vdiff = difflib.ndiff(vtable,vgold)
        errors += 2
        LOG.write("ERROR: Velocity tables diffed:\n\n")
        LOG.write("".join(vdiff))
    else:
        LOG.write("vtable PASSED")

    return errors

def runTest():
    perform_calcs = performCalcs()
    if perform_calcs != 0: return 2
    return analyzeTest()

if __name__ == "__main__":
    import time
    if "--cleanup" in sys.argv:
        for ext in ["props","vtable","dtable","out","diff","log","prf","pyc","echo"]:
            try: os.remove("%s.%s"%(name,ext))
            except: pass
            continue
        pass
    else:
        t0 = time.time()
        print("%s RUNNING"%name)
        perform_calcs = performCalcs()
        dtp = time.time()-t0
        if perform_calcs == 1:
            print("%s FAILED TO RUN TO COMPLETION"%(name))
            sys.exit()
            pass
        print("%s FINISHED"%name)
        print("%s ANALYZING"%name)
        t1 = time.time()
        run_test = analyzeTest()
        dta = time.time()-t1
        if run_test == 0:
            print("%s PASSED(%fs)"%(name,dtp+dta))
        elif run_test == 1:
            print("%s DISPLACEMENT TABLES DIFFED(%fs)"%(name,dtp+dta))
        elif run_test == 2:
            print("%s VELOCITY TABLES DIFFED(%fs)"%(name,dtp+dta))
        elif run_test == 3:
            print("%s DISPLACEMENT & VELOCITY TABLES DIFFED(%fs)"%(name,dtp+dta))
        else:
            print("%s FAILED(%fs)"%(name,dtp+dta))
