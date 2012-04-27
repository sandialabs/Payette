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

import Payette_config as pc
import Source.Payette_utils as pu
import Source.Materials.Payette_installed_materials as pim

speed_kws = ["fast", "medium", "long"]
type_kws = ["verification", "validation", "prototype", "regression"]


class TestLogger(object):
    def __init__(self, name, mode="w"):
        if name:
            self.file = open(name, mode)
        else:
            self.file = sys.stdout
        pass

    def __del__(self):
        if self.file != sys.stdout:
            self.file.close()
        pass

    def write(self, message):
        self.file.write(message + "\n")
        pass

    def warn(self, caller, message):
        self.write("WARNING: {0} [reported by {1}]".format(message, caller))
        pass

    def error(self, caller, message):
        self.write("ERROR: {0} [reported by {1}]".format(message, caller))
        pass


class PayetteTest(object):

    name = None
    tdir = None
    keywords = []
    owner = None
    date = None
    infile = None
    outfile = None
    rms_sets = None
    runcommand = None
    material = None
    baseline = []
    aux_files = []
    description = None
    enabled = False
    passcode = 0
    badincode = 1
    diffcode = 2
    failcode = 3
    failtoruncode = 4
    compare_method = None
    checked = False

    items_to_skip = []
    items_to_compare = []

    # tolerances
    difftol = 6.e-6
    failtol = 1.e-3

    def __init__(self, check=True):
        pass

    def get_keywords(self):
        return self.keywords

    def check_setup(self):

        iam = "{0}.checkSetup(self)".format(self.name)

        errors = 0

        if not self.enabled:
            return errors

        if not self.name:
            errors += 1
            pu.logwrn("no name given for test", caller=iam)
            pass

        if not self.tdir:
            errors += 1
            pu.logwrn("no test directory given for test", caller=iam)
            pass

        if not self.keywords:
            errors += 1
            pu.logwrn("no keywords given", caller=iam)
            pass

        if not isinstance(self.keywords, (list, tuple)):
            errors += 1
            pu.logwrn("keywords must be list, got {0}".format(self.keywords),
                      caller=iam)
            pass

        else:
            self.keywords = [x.lower() for x in self.keywords]
            lkw = 0
            tkw = 0
            for kw in self.keywords:
                if kw in speed_kws:
                    lkw += 1
                if kw in type_kws:
                    tkw += 1
                continue
            if not lkw:
                errors += 1
                msg = "keywords must specify one of {0}".format(", ".join(speed_kws))
                pu.logerr(msg, caller=iam)
            elif lkw > 1:
                errors += 1
                msg = ("keywords must specify only one of {0}"
                       .format(", ".join(speed_kws)))
                pu.logerr(msg, caller=iam)
                pass
            if not tkw:
                errors += 1
                msg = "keywords must specify one of {0}".format(", ".join(type_kws))
                pu.logerr(msg, caller=iam)
            elif tkw > 1:
                msg = ("keywords must specify only one of {0}"
                       .format(", ".join(type_kws)))
                pu.logwrn(msg, caller=iam)
                pass
            pass

        if self.owner is None:
            errors += 1
            pu.logwrn("no owner specified", caller=iam)
            pass

        if self.date is None:
            errors += 1
            pu.logwrn("no date given", caller=iam)
            pass

        if self.infile is not None and not os.path.isfile(self.infile):
            errors += 1
            pu.logwrn("infile {0} not found".format(self.infile), caller=iam)
            pass

        if self.baseline:
            if not isinstance(self.baseline, (list, tuple)):
                self.baseline = [self.baseline]

            for fff in self.baseline:
                if not os.path.isfile(fff):
                    errors += 1
                    pu.logwrn("baseline {0} not found".format(fff),
                              caller=iam)
                    continue

        if self.aux_files:
            if not isinstance(self.aux_files, (list, tuple)):
                self.aux_files = [self.aux_files]
            for fff in self.aux_files:
                if not os.path.isfile(fff):
                    errors += 1
                    pu.logwrn("auxilary file {0} not found".format(fff),
                              caller=iam)
                continue

        if self.compare_method is None:
            self.compare_method = self.compare_out_to_baseline_rms

        if self.description is None:
            errors += 1
            pu.logwrn("no description given", caller=iam)
            pass

        if not isinstance(self.items_to_skip, (list,tuple)):
            self.items_to_skip = [self.items_to_skip]


        if not isinstance(self.items_to_compare, (list,tuple)):
            self.items_to_compare = [self.items_to_compare]

        # check for previous errors
        if errors:
            pu.logerr("stopping due to previous errors", caller=iam)

        self.checkd = True

        return errors

    def runTest(self):

        """ run the test """

        perform_calcs = self.run_command(self.runcommand)

        if perform_calcs != 0:
            return self.failtoruncode

        compare = self.compare_method()

        return compare

    def run_command(self,*cmd,**kwargs):

        cmd, error = self.build_command(*cmd)
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
        with open(echof,"w") as fobj:
            run = subprocess.Popen(cmd, stdout=fobj, stderr=subprocess.STDOUT)
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

        elif exenam in pc.PC_EXES:
            exenam = pc.PC_EXES[exenam]
            found = True

        else:
            path = os.getenv("PATH").split(os.pathsep)
            path.insert(0, pc.PC_TOOLS)
            for p in path:
                exenam = os.path.join(p,exenam)
                if os.path.isfile(exenam):
                    found = True
                    break
                continue
            pass

        if not found:
            pu.error("executable {0} not found".format(exenam), caller=iam)
            return None, self.failcode

        cmd[0] = exenam

        return [x for x in cmd], self.passcode

    def compare_out_to_baseline_rms_general(self):
        """
            Compute the RMS difference between 2D data sets. The difference
            is defined by the sum of the shortest distances between each point
            in the simulated file to any given line segment in the gold file.

            OUTPUT
                0: passed
                1: bad input
                2: diffed
                3: failed
        """
        iam = "{0}.compare_out_to_baseline_rms_general(self)".format(self.name)
        errors = 0

        # open the log file
        log = TestLogger(self.name + ".diff","w")

        if "baselinef" not in globals():
            baselinef = self.baseline[0]
            pass

        if not os.path.isfile(baselinef):
            log.error(iam,"baseline file not found {0}".format(self.name))
            errors += 1
            pass

        if "outf" not in globals():
            outf = self.outfile
            pass

        if not os.path.isfile(outf):
            log.error(iam,"output file not found for {0}".format(self.name))
            errors += 1
            pass

        if errors:
            return self.badincode

        # Determine which columns will be analyzed
        if type(self.rms_sets) != list:
            log.error(iam,"No proper RMS sets to analyze for this benchmark.\n"+
            "Please set self.rms_sets in the input file. Here is an example:\n"+
            "self.rms_sets=[ [\"VAR1\", \"VAR2\"], "+
                            "[\"VAR3\", \"VAR4\"] ... ]\n")
            return self.badincode

        for rms_set in self.rms_sets:
            iserror = False
            if type(rms_set) != list:
                iserror = True
            elif len(rms_set) != 2:
                iserror = True
            elif not iserror and type(rms_set[0]) != str or type(rms_set[1]) != str:
                iserror = True

            if iserror:
                log.error(iam,"RMS set incorrectly defined.\n"+
                     "Expected list '[\"VAR1\",\"VAR2\"]', got '{0}'".\
                              format(repr(rms_set)))
                return self.badincode
        self.rms_sets = [ [x[0].lower(), x[1].lower()] for x in self.rms_sets]

        # read in header
        outheader = [x.lower() for x in self.get_header(outf)]
        goldheader = [x.lower() for x in self.get_header(baselinef)]

        for rms_set in self.rms_sets:
            if rms_set[0] not in outheader:
                errors += 1
                log.error(iam,"'{0}' not in {1}\nValid headers: {2}".\
                          format(rms_set[0], outf,repr(outheader)))
            if rms_set[1] not in outheader:
                errors += 1
                log.error(iam,"'{0}' not in {1}\nValid headers: {2}".\
                          format(rms_set[1], outf,repr(outheader)))
            if rms_set[0] not in goldheader:
                errors += 1
                log.error(iam,"'{0}' not in {1}\nValid headers: {2}".\
                          format(rms_set[0], baselinef,repr(goldheader)))
            if rms_set[1] not in goldheader:
                errors += 1
                log.error(iam,"'{0}' not in {1}\nValid headers: {2}".\
                          format(rms_set[1], baselinef,repr(goldheader)))
        if errors:
            return self.badincode

        # read in data
        out = pu.read_data(outf)
        gold = pu.read_data(baselinef)

        # check that the gold file has at least two points and that the
        # simulation file has at least one.
        if len(out[:,0]) <= 0:
            errors += 1
            log.error(iam,"Not enough points in {0} (must have at least one).".\
                      format(outf))
            log.write("\n{0:=^72s}".format(" FAIL "))
            pass

        if len(gold[:,0]) <= 1:
            errors += 1
            log.error(iam,"Not enough points in {0} (must have at least one).".\
                      format(baselinef))
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

        failed, diffed = False, False
        ftol, dtol = self.failtol, self.difftol
        for rms_set in self.rms_sets:
            gidx = goldheader.index(rms_set[0])
            oidx = outheader.index(rms_set[0])
            gjdx = goldheader.index(rms_set[1])
            ojdx = outheader.index(rms_set[1])

            rmsd, nrmsd = pu.compute_rms_closest_point_residual(
                                         gold[:,gidx], gold[:,gjdx],
                                         out[:,oidx], out[:,ojdx])

            # For good measure, write both the RMSD and normalized RMSD
            if nrmsd >= self.failtol:
                failed = True
                stat = "FAIL"

            elif nrmsd >= self.difftol:
                diffed = True
                stat = "DIFF"

            else:
                stat = "PASS"

            log.write("headers: {0}:{1} - {2}".format(rms_set[0],rms_set[1],stat))
            log.write("  Unscaled error: {0:.10e}".format(rmsd))
            log.write("    Scaled error: {0:.10e}".format(nrmsd))
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

    def compare_out_to_baseline_rms(self, baselinef=None, outf=None):
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
            baselinef = self.baseline[0]
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
            return self.badincode

        # read in header
        outheader = [x.lower() for x in self.get_header(outf)]
        goldheader = [x.lower() for x in self.get_header(baselinef)]

        if outheader[0] != "time":
            errors += 1
            log.error(iam,"time not first column of {0} for {1}".format(outf,
                                                                        self.name))
        if goldheader[0] != "time":
            errors += 1
            log.error(iam,"time not first column of {0} for {1}".format(baselinef,
                                                                        self.name))

        if errors:
            return self.badincode

        # read in data
        out = pu.read_data(outf)
        gold = pu.read_data(baselinef)

        # check that time is same (lengths must be the same)
        if len(gold[:,0]) == len(out[:,0]):
            rmsd, nrmsd = pu.compute_fast_rms(gold[:,0], out[:,0])

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

        # get items to skip
        # kayenta specific customization
        if "kayenta" in self.keywords:
            self.items_to_skip.extend(
                ["KAPPA","EOS1","EOS2","EOS3","EOS4",
                 "PLROOTJ2","SNDSP","ENRGY","RHO","TMPR"])

        to_skip = [x.lower() for x in self.items_to_skip]

        if not self.items_to_compare:
            to_compare = [ x for x in outheader if x in goldheader
                           and x not in to_skip ]

        else:
            to_compare = [x.lower() for x in self.items_to_compare]

        failed, diffed = False, False
        ftol, dtol = self.failtol, self.difftol
        for val in to_compare:
            gidx = goldheader.index(val)
            oidx = outheader.index(val)
            mxg = (np.amax(np.abs(gold[:, gidx])) if
                   np.amax(np.abs(gold[:, gidx])) != 0. else 1.)

            rmsd, nrmsd = pu.compute_rms(gold[:,0], gold[:,gidx],
                                         out[:,0], out[:,oidx])

            # For good measure, write both the RMSD and normalized RMSD
            if nrmsd >= self.failtol:
                failed = True
                stat = "FAIL"

            elif nrmsd >= self.difftol:
                diffed = True
                stat = "DIFF"

            else:
                stat = "PASS"

            log.write("{0}: {1}".format(val, stat))
            log.write("  Unscaled error: {0:.10e}".format(rmsd))
            log.write("    Scaled error: {0:.10e}".format(nrmsd))
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
        for ext in [".out", ".diff", ".log", ".prf", ".pyc", ".echo",
                    ".props", ".pyc", ".math1", ".math2"]:
            try: os.remove(self.name + ext)
            except: pass
            continue
        return

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
        out = pu.read_data(outf)

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

    def diff_files(self, gold=None, out=None):

        """ compare gold with out """

        iam = self.name + "diff_files(self,gold,out)"
        import difflib

        if gold is None:
            gold = self.baseline

        if out is None:
            out = self.outfile

        # open the log file
        log = TestLogger(self.name + ".diff","w")

        errors = 0
        if not isinstance(gold, (list, tuple)):
            gold = [gold]

        for goldf in gold:
            if not os.path.isfile(goldf):
                log.error(iam,"{0} not found".format(goldf))
                errors += 1
            continue

        if not isinstance(out, (list, tuple)):
            out = [out]

        for outf in out:
            if not os.path.isfile(outf):
                log.error(iam,"{0} not found".format(outf))
                errors += 1
            continue

        if len(gold) != len(out):
            errors += 1
            log.error(iam,"len(gold) != len(out)")
            pass

        if errors:
            del log
            return self.failcode

        diff = 0

        for goldf, outf in zip(gold, out):

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

def find_tests(reqkws, unreqkws, spectests, test_dirs=None):
    """ find the Payette tests

    Parameters
    ----------
    reqkws : list
       list of requested keywords
    unreqkws : list
       list of unrequested keywords
    spectests : list
       list of specific tests to find

    Returns
    -------
       found_tests : dict

    """

    import pyclbr

    iam = "find_tests"

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

    if test_dirs is not None:
        errors = 0
        for test_dir in test_dirs:
            if not os.path.isdir(test_dir):
                errors += 1
                pu.logerr("test directory {0} not found".format(test_dir))
            continue
        if errors:
            sys.exit("ERROR: stopping due to previous errors")
    else:
        test_dirs = [pc.PC_TESTS]

    # reqkws are user specified keywords
    errors = 0
    if reqkws:
        reqkws = [x.lower() for x in reqkws]
    if unreqkws:
        unreqkws = [x.lower() for x in unreqkws]
    if spectests:
        spectests = [x.lower() for x in spectests]

    reqkws.sort()
    unreqkws.sort()

    found_tests = {}
    for speed_kw in speed_kws:
        found_tests[speed_kw] = {}
        continue

    py_modules = {}
    for test_dir in test_dirs:
        for dirname, dirs, files in os.walk(test_dir):

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

                continue
            continue

        continue

    for py_mod, py_file in py_modules.items():

        # load module
        py_path = [os.path.dirname(py_file)]
        fp, pathname, description = imp.find_module(py_mod,py_path)
        py_module = imp.load_module(py_mod,fp,pathname,description)
        fp.close()

        # check if a payette test class is defined
        class_data = pyclbr.readmodule(py_mod, path=py_path)

        if not class_data:
            continue

        test, payette_test = False, False
        for name, data in sorted(class_data.items(), key=lambda x:x[1].lineno):
            class_name = data.name
            test = class_name == "Test"
            payette_test = "PayetteTest" in get_super_classes(name,data)
            if payette_test:
                break
            continue

        if not payette_test:
            continue

        if payette_test and not test:
            errors += 1
            warn(iam,"{0} test class name must be 'Test', got {1}"
                 .format(class_name,py_mod))
            continue

        include = False

        # instantiate the test object, but don't check it. we just want to
        # know if the test is requested, or not.
        test = py_module.Test(check=False)

        if not test.enabled:
            warn(iam,"disabled test: {0} encountered".format(py_mod))
            continue

        if not isinstance(test.keywords, (list, tuple)):
            test.keywords = [test.keywords]

        if spectests:
            if test.name not in spectests:
                # test not in user requested tests
                continue
            else:
                include = True

        elif not reqkws and not unreqkws:
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

        if not include:
            continue

        # we've gotten this far, so the test was requested, now check if the
        # material model used in the test is installed
        if test.material is not None:
            if test.material not in pim.PAYETTE_INSTALLED_MATERIALS:
                warn(iam, ("material model {0} required by {0} not installed"
                           .format(test.material, test.name)))
                continue

        speed = [x for x in speed_kws if x in test.keywords][0]
        found_tests[speed][py_mod] = py_file

        continue

    return errors, found_tests

if __name__ == "__main__":
    errors, found_tests = find_tests(["elastic","fast"],[],[])

    fast_tests = [ val for key,val in found_tests["fast"].items() ]
    medium_tests = [ val for key,val in found_tests["medium"].items() ]
    long_tests = [ val for key,val in found_tests["long"].items() ]
    print fast_tests
    print medium_tests
    print long_tests
    sys.exit("here at end")

