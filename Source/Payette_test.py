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
import subprocess
import time

if __name__ == "__main__":
    thisd = os.path.dirname(os.path.realpath(__file__))
    srcd = os.path.dirname(thisd)
    sys.path.append(srcd)

from Source.Payette_utils import *

try: from Source.Materials.Payette_installed_materials import *
except ImportError:
    reportError(__file__,"Toolset/buildPayette must be run to create "
                "Source/Materials/Payette_installed_materials.py")

speed_kws = ["fast","medium","long"]
type_kws = ["verification","validation","prototype","regression"]

class TestLogger(object):
    def __init__(self,name,mode="w"):
        if name: self.file = open(name,mode)
        else: self.file = sys.stdout
        pass
    def __del__(self):
        if self.file != sys.stdout: self.file.close()
        pass
    def write(self,message):
        self.file.write(message + "\n")
        pass
    def warn(self,caller,message):
        self.write("WARNING: {0} [reported by {1}]".format(message,caller))
        pass
    def error(self,caller,message):
        self.write("ERROR: {0} [reported by {1}]".format(message,caller))
        pass


class PayetteTest:

    def __init__(self):

        self.name = None
        self.tdir = None
        self.keywords = []
        self.owner = None
        self.date = None
        self.infile = None
        self.outfile = None
        self.compare_method = self.compare_out_to_baseline_rms
        self.runcommand = None
        self.baseline = None
        self.description = None
        self.enabled = False
        self.passcode = 0
        self.badincode = 1
        self.diffcode = 2
        self.failcode = 3
        self.failtoruncode = 4

        self.items_to_skip = []
        self.items_to_compare = []

        # tolerances
        self.difftol = 6.e-6
        self.failtol = 1.e-3

        pass

    def checkSetup(self):

        iam = "{0}.checkSetup(self)".format(self.name)

        errors = 0

        if not self.enabled:
            return errors

        if not self.name:
            errors += 1
            self.warn(iam,"no name given for test")
            pass

        if not self.tdir:
            errors += 1
            self.warn(iam,"no test directory given for test")
            pass

        if not self.keywords:
            errors += 1
            self.warn(iam,"no keywords given")
            pass

        if not isinstance(self.keywords,(list,tuple)):
            errors += 1
            self.warn(iam,"keywords must be list, got {0}".format(self.keywords))
            pass

        else:
            self.keywords = [x.lower() for x in self.keywords]
            lkw = 0
            tkw = 0
            for kw in self.keywords:
                if kw in speed_kws: lkw += 1
                if kw in type_kws: tkw += 1
                continue
            if not lkw:
                msg = "keywords must specify one of {0}".format(", ".join(speed_kws))
                self.warn(iam,msg)
            elif lkw > 1:
                msg = ("keywords must specify only one of {0}"
                       .format(", ".join(speed_kws)))
                self.warn(iam,msg)
                pass
            if not tkw:
                msg = "keywords must specify one of {0}".format(", ".join(type_kws))
                self.warn(iam,msg)
            elif tkw > 1:
                msg = ("keywords must specify only one of {0}"
                       .format(", ".join(type_kws)))
                self.warn(iam,msg)
                pass
            pass

        if not self.owner:
            errors += 1
            self.warn(iam,"no owner specified")
            pass

        if not self.date:
            errors += 1
            self.warn(iam,"no date given")
            pass

        if self.infile and not os.path.isfile(self.infile):
            errors += 1
            self.warn(iam,"infile {0} not found".format(self.infile))
            pass

        if self.baseline:
            if isinstance(self.baseline,list):
                for fff in self.baseline:
                    if not os.path.isfile(fff):
                        errors += 1
                        self.warn(iam,"baseline {0} not found".format(fff))
                        pass
                    continue
            elif not os.path.isfile(self.baseline):
                errors += 1
                self.warn(iam,"baseline {0} not found".format(self.baseline))
                pass
            pass

        if not self.description:
            errors += 1
            self.warn(iam,"no description given")
            pass

        if not isinstance(self.items_to_skip,(list,tuple)):
            errors += 1
            self.warn(iam,"items to skip must be list, got {0}"
                      .format(self.items_to_skip))
            pass


        if not isinstance(self.items_to_compare,(list,tuple)):
            errors += 1
            self.warn(iam,"items to compare must be list, got {0}"
                      .format(self.items_to_compare))
            pass

        # check for previous errors
        if errors:
            self.error(iam,"fix previous warnings")
            pass

        return errors

    def runTest(self):

        """ run the test """

        perform_calcs = self.run_command(self.runcommand)

        if perform_calcs != 0:
            return self.failtoruncode

        compare = self.compare_method()

        return compare

    def error(self,caller,message):
        print "ERROR: {0} [reported by {1}]".format(message,caller)
        return

    def warn(self,caller,message):
        print "WARNING: {0} [reported by {1}]".format(message,caller)
        return

    def inform(self,caller,message):
        print "INFO: {0} [reported by {1}]".format(message,caller)
        return

    def run_command(self,*cmd,**kwargs):

        cmd,error = self.build_command(*cmd)
        if error:
            return error

        try: echof = kwargs["echof"]
        except: echof = self.name + ".echo"
#        if "runPayette" in cmd[0]:
#            # run directly and not through a subprocess
#            sys.stdout = open(echof,"w")
#            sys.stderr = sys.stdout
#            returncode = runPayette(len(cmd[1:]),cmd[1:])
#            sys.stdout.close()
#            sys.stdout = sys.__stdout__
#            sys.stderr = sys.__stderr__
#            return returncode
        with open(echof,"w") as f:
            run = subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            run.wait()
            pass

        return run.returncode

    def build_command(self,cmd):

        iam = "{0}.build_command".format(self.name)

        if not isinstance(cmd,(list,tuple)):
            cmd = [cmd]
            pass

        exenam = cmd[0]
        found = False
        if os.path.isfile(exenam):
            found = True

        elif exenam in PAYETTE_EXECUTABLES:
            exenam = PAYETTE_EXECUTABLES[exenam]
            found = True

        else:
            path = os.getenv("PATH").split(os.pathsep)
            path.insert(0,PAYETTE_TOOLSET)
            for p in path:
                exenam = os.path.join(p,exenam)
                if os.path.isfile(exenam):
                    found = True
                    break
                continue
            pass

        if not found:
            error(iam,"executable {0} not found".format(exenam))
            return None,self.failcode

        cmd[0] = exenam

        return [ x for x in cmd ], self.passcode

    def compare_out_to_baseline_rms(self,baselinef=None,outf=None):
        """
            Compare results from out file to those in baseline

            OUTPUT
                0: passed
                1: bad input
                2: diffed
                3: failed
        """
        iam = "{0}.compare_out_to_baseline_rms(self)".format(self.name)
        errors = 0

        # open the log file
        log = TestLogger(self.name + ".diff","w")

        if not baselinef:
            baselinef = self.baseline
            pass

        if not os.path.isfile(baselinef):
            log.error(iam,"baseline file not found {0}".format(self.name))
            errors += 1
            pass

        if not outf:
            outf = self.outfile
            pass

        if not os.path.isfile(outf):
            log.error(iam,"output file not found for {0}".format(self.name))
            errors += 1
            pass

        if errors:
            return badincode

        # read in header
        outheader = [x.lower() for x in self.get_header(outf)]
        goldheader = [x.lower() for x in self.get_header(baselinef)]

        if outheader[0] != "time":
            errors += 1
            log.error(iam,"time not first column of {0} for {1}".format(outf,
                                                                        self.name))
            pass

        if goldheader[0] != "time":
            errors += 1
            log.error(iam,"time not first column of {0} for {1}".format(baselinef,
                                                                        self.name))
            pass

        if errors:
            return badincode

        # read in data
        out = self.read_data(outf)
        gold = self.read_data(baselinef)

        # check that time is same (lengths must be the same)
        if len(gold[:,0]) == len(out[:,0]):
            rmsd, nrmsd = self.compute_fast_RMS(gold[:,0], out[:,0])
        else:
            rmsd, nrmsd = 1.0e99, 1.0e99

        if nrmsd > np.finfo(np.float).eps:
            errors += 1
            log.error(iam,"time step error between {0} and {1}"
                      .format(outf,baselinef))
            log.write("\n{0:=^72s}".format(" FAIL "))
            pass

        if errors:
            del log
            return self.failcode

        # compare results
        log.write("Payette test results for: {0}\n".format(self.name))
        log.write("TOLERANCES:")
        log.write("  diff tol: {0:10e}".format(self.difftol))
        log.write("  fail tol: {0:10e}\n".format(self.failtol))
        errors = []

        # get items to skip
        # kayenta specific customization
        if "kayenta" in self.keywords:
            self.items_to_skip.extend(["KAPPA","EOS1","EOS2","EOS3","EOS4",
                                       "PLROOTJ2","SNDSP","ENRGY","RHO","TMPR"])
            pass
        to_skip = [x.lower() for x in self.items_to_skip]

        if not self.items_to_compare:
            to_compare = [ x for x in outheader if x in goldheader
                           and x not in to_skip ]
            pass
        else: to_compare = [x.lower() for x in self.items_to_compare]

        failed, diffed = False, False
        for val in to_compare:
            gidx = goldheader.index(val)
            oidx = outheader.index(val)
            rmsd, nrmsd = self.compute_RMS(gold[:,0],gold[:,gidx],
                                           out[:,0],out[:,oidx])

            # For good measure, write both the RMSD and normalized RMSD
            if nrmsd >= self.failtol:
                failed = True
                stat = "FAIL"
            elif nrmsd >= self.difftol:
                diffed = True
                stat = "DIFF"
            else:
                stat = "PASS"
                pass
            log.write("{0}: {1}".format(val,stat))
            log.write("  Unscaled error: {0:.10e}".format(rmsd))
            log.write("    Scaled error: {0:.10e}".format(nrmsd))
            errors.append(nrmsd)
            continue

        if failed:
            log.write("\n{0:=^72s}".format(" FAIL "))
            del log
            return self.failcode

        elif diffed:
            log.write("\n{0:=^72s}".format(" DIFF "))
            del log
            return self.diffcode

        else:
            log.write("\n{0:=^72s}".format(" PASS "))
            del log
            return self.passcode

        return

    def get_header(self,f):
        """ get the header of f """
        return linecache.getline(f,1).split()

    def clean_tracks(self):
        for ext in [".out",".diff",".log",".prf",".pyc",".echo",".props",".pyc"]:
            try: os.remove(self.name + ext)
            except: pass
            continue
        return

    def read_data(self,f):
        """
            Reads in a whitespace-delimed data file f. It is assumed that the
            first line contains text (the name of the column). All other lines
            contain floats.

            Returns m x n dimensional numpy ndarray where m is the number of data
            points (rows) and n the number of data fields (columns)
        """
        return np.loadtxt(f,skiprows=1)

    def compute_fast_RMS(self,gold,out):
        """
            Determines the RMS of the error between two piecewise functions of
            same length. This should only be called if the timestep sizes of the
            gold simulation and the simulation that produced out are the same.
            This function makes no attempt to make that determination, it is up
            to the caller.

            Tim Fuller, Sandia, National Laboratories, tjfulle@sandia.gov
        """
        rmsd = math.sqrt(np.sum((gold - out)**2)/float(len(gold)))
        dnom = abs(np.amax(gold) - np.amin(gold))
        nrmsd = rmsd/dnom if dnom >= np.finfo(np.float).eps else rmsd
        return rmsd, nrmsd

    def compute_RMS(self,set1x,set1y,set2x,set2y,step=1):
        """
        NAME:
            compute_RMS( list(float)x4, step=int )

        PURPOSE:
            Determines the RMS of the error between two piecewise functions. In
            almost all cases, the abscissa variables (set1x and set2x) will be
            time. Currently, the abscissa variables are required to be
            monotonically increasing (for easier computing). Interpolations are
            used to compare data points where data is missing in the other set.

            "set2{x,y}" is compared against "set1{x,y}".

            len(set1x) == len(set1y)
            len(set2x) == len(set2y)

            The optional "step" argument is used for large data sets. This gives
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

        # Use a shortcut if the lengths of the x and y data sets are the same.
        # Also, before using the shortcut, do a quick check by computing the RMS
        # on every 10th x value. The RMS check on the x-values only gets computed
        # if len(set1y) == len(set2y). This check allows for the possibility of
        # the lengths of the data to be equal but for the spacing to be not
        # equal.
        if (lset1y == lset2y and
            np.sum((set1x[::10]-set2x[::10])**2) < 1.0e-6*np.amax(set1x)):
            return self.compute_fast_RMS(set1y,set2y)

        else:

            # compute the running square of the difference
            err = 0.0
            for i in range(0,lset2x,step):

                for j in range(0,lset1x-1,step):

                    # mss: Very expensive look. Optimize.
                    # if set2x is between set1x pair, do linear interpolation
                    if set1x[j] <= set2x[i] <= set1x[j+1]:
                        x0 = set1x[j];   y0 = set1y[j]
                        x1 = set1x[j+1]; y1 = set1y[j+1]
                        f = y0 + (set2x[i]-x0)*(y1-y0)/(x1-x0)

                        # Add the square of the difference to err
                        err += (set2y[i] - f)**2
                        break
                    continue
                continue
            pass

        rmsd = math.sqrt(err/float(lset1x))
        dnom = abs(np.amax(set1y) - np.amin(set1y))
        nrmsd = rmsd/dnom if dnom >= 2.e-16 else rmsd
        return rmsd, nrmsd

    def compare_constant_strain_at_failure(self,outf=None,epsfail=None):
        """ compare the constant strain at failure with expected """

        iam = "{0}.compare_strain_at_failure(self)".format(self.name)
        errors = 0

        # open the log file
        log = TestLogger(self.name + ".diff","w")

        if outf:
            if not os.path.isfile(outf):
                errors += 1
                log.error(iam,"sent output file not found")
        else:
            outf = self.outfile
            pass

        if not outf:
            errors +=1
            log.error(iam,"not out file given")
            pass

        propf = self.name + ".props"
        if not os.path.isfile(propf):
            errors += 1
            log.error(iam,"{0} not found".format(propf))
            pass

        # get constant strain at failure
        if not epsfail:
            props = open(propf,"r").readlines()
            for prop in props:
                prop = [x.strip() for x in prop.split("=")]
                if prop[0].lower() == "fail2":
                    epsfail = float(prop[1])
                    break
                continue
            pass
        try: epsfail = float(epsfail)
        except TypeError:
            errors += 1
            log.error(iam,"epsfail must be float, got {0}".format(epsfail))
        except:
            errors += 1
            log.error(iam,"bad epsfail [{0}]".format(epsfail))
            pass

        # read in header
        outheader = [x.lower() for x in self.get_header(outf)]

        if outheader[0] != "time":
            errors += 1
            log.error(iam,"time not first column of {0}".format(outf))
            pass

        if errors:
            return self.badincode

        # read in the data
        out = self.read_data(outf)

        # compare results
        failed, diffed = False, False
        log.write("Payette test results for: {0}\n".format(self.name))
        log.write("TOLERANCES:")
        log.write("  diff tol: {0:10e}".format(self.difftol))
        log.write("  fail tol: {0:10e}\n".format(self.failtol))

        # Get the indicies for COHER and ACCSTRAIN. Then, verify COHER does drop
        # below 0.5. If it does not, then this test is a FAIL.
        coher_idx     = outheader.index("coher")
        accstrain_idx = outheader.index("accstrain")
        fail_idx = -1
        for idx, val in enumerate(out[:,coher_idx]):
            if val < 0.5:
                fail_idx = idx
                break
            continue

        if fail_idx == -1:
            log.error(iam,"COHER did not drop below 0.5.\n")
            log.error(iam,"Final value of COHER: {0}\n".format(out[-1,coher_idx]))
            return self.failcode

        # Perform an interpolation between COHER-ACCSTRAIN sets to find the
        # ACCSTRAIN when COHER=0.5 . Then compute the absolute and relative
        # errors.
        x0, y0 = out[fail_idx-1,coher_idx], out[fail_idx-1,accstrain_idx]
        x1, y1 = out[fail_idx,  coher_idx], out[fail_idx,  accstrain_idx]

        strain_f = y0 + (0.5-x0)*(y1-y0)/(x1-x0)
        abs_err = abs(strain_f-epsfail)
        rel_err = abs_err/abs(max(strain_f,epsfail))

        # Write to output.
        log.write("COHER absolute error: {0}\n".format(abs_err))
        log.write("COHER relative error: {0}\n".format(rel_err))
        if rel_err >= self.failtol:
            failed = True
            stat = "FAIL"

        elif rel_err >= self.difftol:
            diffed = True
            stat = "DIFF"
        else:
            stat = "PASS"
            pass

        if failed:
            log.write("\n{0:=^72s}".format(" FAIL "))
            del log
            return self.failcode

        elif diffed:
            log.write("\n{0:=^72s}".format(" DIFF "))
            del log
            return self.diffcode

        else:
            log.write("\n{0:=^72s}".format(" PASS "))
            del log
            return self.passcode

        return


    def runFromTerminal(self,argv,compare_method=None):

        if "--cleanup" in argv or "-c" in argv:
            self.clean_tracks()
            return

        t0 = time.time()

        if "--full" in argv:
            print("{0:s} RUNNING FULL TEST".format(self.name))
            result = self.runTest()
            dta = time.time() - t0
            dtp = dta

        else:
            print("{0:s} RUNNING".format(self.name))

            perform_calcs = self.run_command(self.runcommand)
            dtp = time.time() - t0

            if perform_calcs != 0:
                print("{0:s} FAILED TO RUN TO COMPLETION".format(self.name))
                sys.exit()
                pass

            print("{0:s} FINISHED".format(self.name))
            print("{0:s} ANALYZING".format(self.name))

            t1 = time.time()
            compare = self.compare_method()
            dta = time.time() - t1
            pass

        if compare == self.passcode:
            print("{0:s} PASSED({1:f}s)".format(self.name,dtp+dta))

        elif compare == self.badincode:
            print("{0:s} BAD INPUT({1:f}s)".format(self.name,dtp+dta))

        elif compare == self.diffcode:
            print("{0:s} DIFFED({1:f}s)".format(self.name,dtp+dta))

        elif compare == self.failcode:
            print("{0:s} FAILED({1:f}s)".format(self.name,dtp+dta))

        else:
            print("{0:s} UNKOWN STAT({1:f}s)".format(self.name,dtp+dta))
            pass

        return

    def diff_files(self,gold,out):

        """ compare gold with out """

        iam = self.name + "diff_files(self,gold,out)"
        import difflib

        # open the log file
        log = TestLogger(self.name + ".diff","w")

        one_gold, one_out = False, False
        errors = 0
        if not isinstance(gold,list):
            one_gold = True
            if not os.path.isfile(gold):
                log.error("{0} not found".format(gold))
                errors += 1
                pass
            pass
        else:
            for goldf in gold:
                if not os.path.isfile(goldf):
                    log.error(iam,"{0} not found".format(goldf))
                    errors += 1
                    pass
                continue
            pass

        if not isinstance(out,list):
            one_out = True
            if not os.path.isfile(out):
                log.error(iam,"{0} not found".format(out))
                errors += 1
                pass
            pass
        else:
            for outf in out:
                if not os.path.isfile(outf):
                    log.error(iam,"{0} not found".format(outf))
                    errors += 1
                    pass
                continue
            pass

        if one_out and not one_gold:
            errors += 1
            log.error(iam,"multiple gold for single out")

        elif one_gold and one_out:
            gold = [gold]
            out = [out]

        elif one_gold and not one_out:
            gold = [gold]*len(out)
            pass

        if len(gold) != len(out):
            errors += 1
            log.error(iam,"len(gold) != len(out)")
            pass

        if errors:
            del log
            return self.failcode

        diff = 0

        for idx in range(len(gold)):

            # compare the files
            goldf = gold[idx]
            outf = out[idx]

            bgold = os.path.basename(goldf)
            xgold = open(goldf).readlines()
            bout = os.path.basename(outf)
            xout = open(outf).readlines()

            if xout != xgold:
                ddiff = difflib.ndiff(xout,xgold)
                diff += 1
                log.write("ERROR: {0} diffed from {1}:\n".format(bout,bgold))
                log.write("".join(ddiff))
            else:
                log.write("PASSED")
                pass
            continue

        del log
        if diff:
            return self.failcode

        return self.passcode

def findTests(reqkws,unreqkws,spectests,test_dir=None):
    """
    NAME
       findTests

    PURPOSE
       determine if any python files in names are Payette tests

    INPUT
       dirname: directory name where the files in names reside
       names: list of files in dirname
       reqkws - list of requested kw
       unreqkws - list of negated kw
       spectests - list of specific tests to run

    OUTPUT
       found_tests: dictionary of found tests

    AUTHORS
       Tim Fuller, Sandia, National Laboratories, tjfulle@sandia.gov
       M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    """

    import pyclbr

    iam = "findTests"

    def get_module_name(py_file):
        return os.path.splitext(os.path.basename(py_file))[0]

    def get_super_classes(name, data):
        super_class_names = []
        for super_class in data.super:
            if super_class == "object":
                continue
            if isinstance(super_class, basestring):
                super_class_names.append(super_class)
            else:
                super_class_names.append(super_class.name)
                pass
            continue
        return super_class_names

    if test_dir:
        if not os.path.isdir(test_dir):
            sys.exit("test directory {0} not found".format(test_dir))
            pass
        pass
    else:
        test_dir = PAYETTE_TESTS
        pass

    # reqkws are user specified keywords
    errors = 0
    if reqkws: reqkws = [x.lower() for x in reqkws]
    if unreqkws: unreqkws = [x.lower() for x in unreqkws]
    if spectests: spectests = [x.lower() for x in spectests]

    # do not run the kayenta tests if kayenta not installed
    if "kayenta" not in PAYETTE_INSTALLED_MATERIALS:
        unreqkws.append("kayenta")
        if "kayenta" in reqkws:
            errors += 1
            warn(iam,"requested kayenta tests but kayenta model not installed")
            pass
        pass

    # do not run the piezo electric material's tests if not installed
    if "domain_switching_ceramic" not in PAYETTE_CONSTITUTIVE_MODELS:
        unreqkws.append("domain_switching_ceramic")
        if "domain_switching_ceramic" in reqkws:
            errors += 1
            warn(iam,("requested domain_switching_ceramic tests but "
                      "domain_switching_ceramic model not installed"))
            pass
        pass

    if "piezo_ceramic" not in PAYETTE_CONSTITUTIVE_MODELS:
        unreqkws.append("piezo_ceramic")
        if "piezo_ceramic" in reqkws:
            errors += 1
            warn(iam,("requested piezo_ceramic tests but "
                      "piezo_ceramic model not installed"))
            pass
        pass

    reqkws.sort()
    unreqkws.sort()

    found_tests = {}
    for speed_kw in speed_kws:
        found_tests[speed_kw] = {}
        continue

    py_modules = {}
    for dirname,dirs,files in os.walk(test_dir):

        if ".svn" in dirname or "__test_dir__.py" not in files:
            continue

        for fname in files:
            fpath = os.path.join(dirname,fname)
            fbase,fext = os.path.splitext(fname)

            # filter out all files we know cannot be test files
            if ( fext != ".py" or
                 fbase[0] == "." or
                 fbase == "__init__" or
                 fbase == "__test_dir__" or
                 fbase == "Payette_config" or
                 fbase == "template" ):
                continue

            py_mod = get_module_name(fpath)
            if py_mod in py_modules:
                errors += 1
                warn(iam,"removing duplicate python module {0} in tests"
                     .format(py_mod))
                del py_modules[py_mod]
            else:
                py_modules[py_mod] = fpath
                pass

            continue

        continue

    for py_mod, py_file in py_modules.items():

        # load module
        py_path = [os.path.dirname(py_file)]
        fp, pathname, description = imp.find_module(py_mod,py_path)
        py_module = imp.load_module(py_mod,fp,pathname,description)
        fp.close()

        # check if a payette test class is defined
        class_data = pyclbr.readmodule(py_mod,path=py_path)

        if not class_data:
            continue

        test, payette_test = False, False
        for name, data in sorted(class_data.items(), key=lambda x:x[1].lineno):
            class_name = data.name
            test = class_name == "Test"
            payette_test = "PayetteTest" in get_super_classes(name,data)
            if payette_test: break
            continue

        if not payette_test:
            continue

        if payette_test and not test:
            errors += 1
            warn(iam,"{0} test class name must be 'Test', got {1}"
                 .format(class_name,py_mod))
            continue

        include = False
        test = py_module.Test()
        check = test.checkSetup()

        if check != 0:
            errors += check
            warn(iam,"non conforming test module {0} found".format(py_mod))
            continue

        if not test.enabled:
            warn(iam,"disabled test: {0} encountered".format(py_mod))
            continue

        if spectests and test.name not in spectests:
            # test not in user requested tests
            continue

        if not reqkws and not unreqkws:
            # if user has not specified any kw or kw negations
            # append to conform
            include = True

        else:
            # user specified a kw or kw negation
            kws = test.keywords

            if [x for x in unreqkws if x in kws]:
                # negation takes precidence
                continue

            if not reqkws:
                # wasn't negated and no other kws
                include = True

            else:
                # specified kw, make sure that all kws match
                reqkeyl = [x for x in reqkws if x in kws]
                reqkeyl.sort()
                if reqkeyl == reqkws:
                    include = True
                else:
                    continue
                pass

            pass

        if include:

            speed = [x for x in speed_kws if x in test.keywords][0]

            found_tests[speed][py_mod] = py_file

            pass

        continue

    return errors, found_tests

def warn(iam,message):
    print("WARNING: {0} [reported by {1}]".format(message,iam))
    return

def error(iam,message):
    print("ERROR: {0} [reported by {1}]".format(message,iam))
    return

if __name__ == "__main__":
    errors, found_tests = findTests(["elastic","fast"],[],[])

    fast_tests = [ val for key,val in found_tests["fast"].items() ]
    medium_tests = [ val for key,val in found_tests["medium"].items() ]
    long_tests = [ val for key,val in found_tests["long"].items() ]
    print fast_tests
    print medium_tests
    print long_tests
    sys.exit("here at end")

