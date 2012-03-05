#!/usr/bin/env python
import sys,subprocess,linecache,os
import time
import math
import numpy as np

from tests_common import *

Payette_test = True   # change to true
name = "payette-test-restart"
keywords = ["payette","restart","regression","fast"] # add keywords
owner = "Tim Fuller"
date = "February 25, 2012"

fdir = os.path.dirname(os.path.realpath(__file__))
infile = "%s.inp"%(os.path.join(fdir,name))
restartfile = name + ".prf"
outfile = "%s.out"%(name)
executable = runPayette
runcommand = [executable,"--no-writeprops","--test-restart",infile]
restartcommand = [executable,"--no-writeprops","--test-restart",restartfile]
baseline = "%s.gold"%(os.path.join(fdir,name))

description = """
    Test of restart capabilities
"""

def performCalcs(cmd):
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
        run = subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
        run.wait()
        pass
    return run.returncode

def analyzeTest():
    return 0

def runTest():
    perform_calcs = performCalcs(runcommand)
    if perform_calcs != 76: return 2
    # now run the restart file
    perform_calcs = performCalcs(restartcommand)
    if perform_calcs != 0: return 2
    return analyzeTest()

if __name__ == "__main__":
    import time
    if "--cleanup" in sys.argv:
        for ext in ["out","res","log","prf","pyc","echo"]:
            try: os.remove("%s.%s"%(name,ext))
            except: pass
            continue
        pass
    else:
        t0 = time.time()
        print("%s RUNNING"%name)
        perform_calcs = performCalcs(runcommand)
        dtp = time.time()-t0
        if perform_calcs != 76:
            print("%s FAILED TO RUN TO COMPLETION ON FIRST LEG"%(name))
            sys.exit()
            pass
        # now run the restart file
        rfile = os.path.splitext(infile)[0] + ".prf"
        perform_calcs = performCalcs(restartcommand)
        if perform_calcs != 0:
            print("%s FAILED TO RUN TO COMPLETION ON RESTART"%(name))
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
            print("%s DIFFED(%fs)"%(name,dtp+dta))
        else:
            print("%s FAILED(%fs)"%(name,dtp+dta))
