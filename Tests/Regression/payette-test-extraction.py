#!/usr/bin/env python
import sys,subprocess,linecache,os
import time
import math
import numpy as np

from tests_common import *

Payette_test = True
name = "payette-test-extraction"
keywords = ["payette","regression","fast","extraction"]
owner = "Tim Fuller"
date = "February 28, 2012"

fdir = os.path.dirname(os.path.realpath(__file__))
infile = "%s.gold"%(os.path.join(fdir,name))
outfile = None
executable = os.path.join(Payette_Toolset,"extractPayette.py")
runcommand = [executable,infile,r"@time",
              r"@eps11",r"@sig11",r"2*@eps11",r"2*@sig11",r"2*%2"]
baseline = ["%s.xgold"%(os.path.join(fdir,name))]

description = """
    Test of extractPayette.py
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
    with open("%s.xout"%(name),"w") as f:
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

    # compare the xout files
    xout = open(name + ".xout").readlines()
    xgold = open(baseline[0]).readlines()
    if xout != xgold:
        ddiff = difflib.ndiff(xout,xgold)
        errors += 1
        LOG.write("ERROR: extracted data diffed:\n\n")
        LOG.write("".join(ddiff))
    else:
        LOG.write("PASSED")

    return errors

def runTest():
    perform_calcs = performCalcs()
    if perform_calcs != 0: return 2
    return analyzeTest()

if __name__ == "__main__":
    import time
    if "--cleanup" in sys.argv:
        for ext in ["xout","diff","pyc","echo"]:
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
        else:
            print("%s DATA EXTRACTION DIFFED(%fs)"%(name,dtp+dta))
