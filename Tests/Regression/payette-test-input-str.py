#!/usr/bin/env python

import sys,subprocess,linecache,os
import time
import math
import numpy as np

from tests_common import *
import Payette

Payette_test = True   # change to true
name = "payette-test-input-str"
keywords = ["payette","input_str","regression","fast"] # add keywords
owner = "Tim Fuller"
date = "February 25, 2012"

fdir = os.path.dirname(os.path.realpath(__file__))
infile = "%s.py"%(os.path.join(fdir,name))
restartfile = "%s.prf"%(os.path.join(fdir,name))
outfile = "%s.out"%(name)
executable = runPayette
baseline = "%s.gold"%(os.path.join(fdir,name))

mtl_props= """
AN=1.
B0=11.634e9
G0=10.018e9
G1=5
A1=8.27e6
A4=0.23
R0=3500
T0=298
GP=1.0
S1=1
CV=800
TM=1e99
IDK=1
IDG=1
"""

inp="""begin simulation payette-test-input-str
  begin material
    constitutive model diamm
    {0}
  end material
  begin boundary
    estar = -1.
    kappa = 0.
    tstar = 1.
    ampl= .01
    begin legs
      0,     0.,   0, 222222, 0., 0., 0., 0., 0., 0.
      1,     1.,   1, 222222, 1., 0., 0., 0., 0., 0.
      2,     2.,   1, 222222, 0., 0., 0., 0., 0., 0.
    end legs
  end boundary
end simulation

begin simulation payette-test-input-str-1
  begin material
    constitutive model diamm
    {0}
  end material
  begin boundary
    estar = -1.
    kappa = 0.
    tstar = 1.
    ampl= .01
    begin legs
      0,     0.,    0, 222222, 0., 0., 0., 0., 0., 0.
      1,     1.,   10, 222222, 1., 0., 0., 0., 0., 0.
      2,     2.,   10, 222222, 0., 0., 0., 0., 0., 0.
    end legs
  end boundary
end simulation
""".format(mtl_props)
runcommand = ["--no-restart","--no-writeprops","-v","0","--input-str=%s"%inp]

description = """
    Test of input string capabilities
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
    run = Payette.runPayette(1,runcommand)
    return run

def analyzeTest():
    return 0


def runTest():
    perform_calcs = performCalcs()
    if perform_calcs != 0: return 2
    return analyzeTest()

    return input_string


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
        perform_calcs = performCalcs()
        dtp = time.time()-t0
        if perform_calcs != 0:
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
            print("%s DIFFED(%fs)"%(name,dtp+dta))
        else:
            print("%s FAILED(%fs)"%(name,dtp+dta))
