# Copyright (2011) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
# rights in this software.

# The MIT License

# Copyright (c) Sandia Corporation

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
import pickle
import textwrap
import shutil
import re
from inspect import getfile

if __name__ == "__main__":
    thisd = os.path.dirname(os.path.realpath(__file__))
    srcd = os.path.dirname(thisd)
    sys.path.append(srcd)

import Source.__config__ as cfg
import Source.Payette_utils as pu
import Source.Payette_model_index as pmi

# --- module level variables
CWD = os.getcwd()
SPEED_KWS = {"fast": 0, "medium": 1, "long": 2}
TYPE_KWS = ["verification", "validation", "prototype", "regression"]
SP = " " * 5
WIDTH_TERM = 80
WIDTH_INFO = 25
TEST_INFO = ".test.info"
RESDIR = os.path.join(CWD, "TestResults.{0}".format(cfg.OSTYPE))


class TestLogger(object):
    def __init__(self, name, mode="w"):
        self.name = str(name)
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

    def warn(self, message):
        caller = pu.who_is_calling()
        self.write("WARNING: {0} [reported by {1}]".format(message, caller))
        pass

    def error(self, message):
        caller = self.name + "." + pu.who_is_calling()
        self.write("ERROR: {0} [reported by {1}]".format(message, caller))
        pass


class PayetteTest(object):

    TOL = 1.e-6
    passcode = 0
    badincode = 1
    diffcode = 2
    failcode = 3
    failtoruncode = 4
    notruncode = 5

    def __init__(self, check=True):
        self.name = None
        self.tdir = None
        self._results_directory = None
        self.keywords = []
        self.owner = None
        self.date = None
        self.infile = None
        self.outfile = None
        self.rms_sets = None
        self.runcommand = None
        self.material = None
        self.baseline = []
        self.aux_files = []
        self.description = None
        self.enabled = False
        self.expect_constant = []
        self._completion_time = None
        self.status = "NOT RUN"
        self.old_status = None
        self.retcode = self.notruncode
        self.postprocess = False
        self.nbs = []

        self.checked = not check

        self.items_to_skip = []
        self.items_to_compare = []

        # tolerances
        self.difftol = 6. * self.TOL
        self.failtol = 1.e3 * self.TOL

        # architecture used to create test
        self.arch = "darwin"
        pass

    def completion_time(self, t=None, f=None):
        if t is None:
            if self._completion_time is not None and f is not None:
                return f.format(self._completion_time)
            return self._completion_time
        self._completion_time = t

    def compare_method(self):
        return self.compare_out_to_baseline_rms()

    def get_keywords(self):
        return self.keywords

    def check_setup(self):

        if not self.enabled:
            return 0

        if not self.name:
            pu.report_error("no name given for test")

        if not self.tdir:
            pu.report_error("no test directory given for test")

        if not self.keywords:
            pu.report_error("no keywords given")

        if not isinstance(self.keywords, (list, tuple)):
            pu.report_error(
                "keywords must be list, got {0}".format(self.keywords))

        else:
            self.keywords = [x.lower() for x in self.keywords]
            speed = [x for x in SPEED_KWS if x in self.keywords]
            typ = [x for x in TYPE_KWS if x in self.keywords]
            if not speed:
                pu.report_error("keywords must specify one of {0}"
                                .format(", ".join(SPEED_KWS)))
            elif len(speed) > 1:
                pu.report_error("keywords must specify only one of {0}"
                                .format(", ".join(SPEED_KWS)))
            if not typ:
                pu.report_error("keywords must specify one of {0}"
                                .format(", ".join(TYPE_KWS)))
            elif len(typ) > 1:
                pu.log_warning("keywords must specify only one of {0}"
                               .format(", ".join(TYPE_KWS)))
            self.speed = speed[0]
            self.type = typ[0]

        if self.owner is None:
            pu.report_error("no owner specified")

        if self.date is None:
            pu.report_error("no date given")

        if self.infile is not None and not os.path.isfile(self.infile):
            pu.report_error("infile {0} not found".format(self.infile))

        if self.baseline:
            if not isinstance(self.baseline, (list, tuple)):
                self.baseline = [self.baseline]

            for fff in self.baseline:
                if not os.path.isfile(fff):
                    pu.report_error("baseline file {0} not found".format(fff))
                    continue

        if self.aux_files:
            if not isinstance(self.aux_files, (list, tuple)):
                self.aux_files = [self.aux_files]
            for fff in self.aux_files:
                if not os.path.isfile(fff):
                    pu.report_error("auxilary file {0} not found".format(fff))
                continue

        if self.description is None:
            pu.report_error("no description given")

        if not isinstance(self.items_to_skip, (list, tuple)):
            self.items_to_skip = [self.items_to_skip]

        if not isinstance(self.items_to_compare, (list, tuple)):
            self.items_to_compare = [self.items_to_compare]

        if self.arch.lower() not in sys.platform.lower():
            # relax tolerances on architectures not used to create the gold
            # file
            self.failtol = 3.e6 * self.TOL

        self.test_file = os.path.realpath(getfile(self.__class__)).rstrip("c")
        self.test_file_dir = os.path.dirname(self.test_file)
        self.find_nbs()
        self.checked = True

        return pu.errors()

    def run_test(self):
        """ run the test """
        t0 = time.time()
        d = os.getcwd()
        os.chdir(self.results_directory())
        perform_calcs = self.run_command(self.runcommand)
        if perform_calcs != 0:
            retcode = self.failtoruncode
        else:
            retcode = self.compare_method()
        self.retcode = retcode
        self.status = self.get_status()
        tc = time.time()
        self.completion_time(tc - t0)
        os.chdir(d)
        return

    def runTest(self):
        return self.run_test()

    def run_command(self, *cmd, **kwargs):

        cmd, error = self.build_command(*cmd)
        if error:
            return error

        try:
            echof = kwargs["echof"]
        except KeyError:
            echof = self.name + ".echo"
#        if "payette" in cmd[0]:
#            # run directly and not through a subprocess
#            sys.stdout = open(echof,"w")
#            sys.stderr = sys.stdout
#            returncode = run_payette(len(cmd[1:]),cmd[1:])
#            sys.stdout.close()
#            sys.stdout = sys.__stdout__
#            sys.stderr = sys.__stderr__
#            return returncode
        with open(echof, "w") as fobj:
            run = subprocess.Popen(cmd, stdout=fobj, stderr=subprocess.STDOUT)
            run.wait()

        return run.returncode

    def build_command(self, cmd):

        if not isinstance(cmd, (list, tuple)):
            cmd = [cmd]
            pass

        exenam = cmd[0]
        found = False
        if os.path.isfile(exenam):
            found = True

        elif exenam in cfg.EXECUTABLES:
            exenam = cfg.EXECUTABLES["path"]
            found = True

        else:
            path = os.getenv("PATH").split(os.pathsep)
            path.insert(0, cfg.TOOLSET)
            for p in path:
                exenam = os.path.join(p, exenam)
                if os.path.isfile(exenam):
                    found = True
                    break
                continue
            pass

        if not found:
            pu.report_error("executable {0} not found".format(exenam))
            return None, self.failcode

        cmd[0] = exenam

        return [x for x in cmd], self.passcode

    def get_retcode(self, failed, diffed):
        if failed:
            return self.failcode
        if diffed:
            return self.diffcode
        return self.passcode

    def get_status(self):
        return {self.passcode: "PASS",
                self.failcode: "FAIL",
                self.diffcode: "DIFF",
                self.failtoruncode: "FAILED TO RUN",
                self.badincode: "BAD INPUT",
                self.notruncode: "NOT RUN"}[self.retcode]

    def switch_materials(self, files, switch):
        """switch materials"""
        pat = re.compile(r"\bconstitutive\s*model\s*{0}"
                         .format(switch[0]), re.I | re.M)
        repl = "constitutive model {0}".format(switch[1])
        for fname in files:
            lines = open(fname, "r").read()
            with open(fname, "w") as fobj:
                fobj.write(pat.sub(repl, lines))
            continue
        return

    def setup_test(self, *args):

        test_base_dir, postprocess, switch = args[:3]
        self.postprocess = postprocess

        # check for switched materials
        if switch is not None:
            switch = [x.lower() for x in switch.split(":")]

        # set up the results directory. If the test was taken from the
        # PC_TESTS directory, we wish to keep the same directory structure in
        # the TestResults.OSTYPE directory. e.g., if the test came from
        # PC_TESTS/Regression, we want to run the test in
        # TestResults.OSTYPE/Regression. But, if the test did not come from
        # the PC_TESTS directory (user specified additional directories to
        # look for tests), we try to keep the base directory name of that
        # test. e.g., if the user asked to look for tests in
        # /some/directory/to/special_tests, we run the tests in
        # TestResults.OSTYP/special_tests.
        if test_base_dir == self.test_file_dir:
            results_directory = (test_file_dir,)

        else:
            f = os.path.join(self.test_file_dir, "__test_dir__.py")
            py_module = pu.load_module(f)
            try:
                testbase = py_module.DIRECTORY
            except AttributeError:
                testbase = os.path.basename(self.test_file_dir)
            results_directory = (RESDIR, testbase, self.name)

        self.set_results_directory(*results_directory)
        d = self.results_directory()
        dd = self.test_file_dir
        if d != dd:
            # Create benchmark directory and copy the input and baseline files
            # into the new directory
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)

            self.copy_files_to_results_directory()

            if self.nbs:
                self.copy_mathematica_nbs()

            # check for switching
            try:
                if switch and self.material.lower() == switch[0]:
                    self.switch_materials(files=self.test_files, switch=switch)
            except AttributeError:
                pass

            # write out hidden test file with some relevant info
            self.test_info = os.path.join(d, TEST_INFO)

            with open(self.test_info, "w") as fobj:
                fobj.write("NAME={0}\n".format(self.name))
                fobj.write("PLATFORM={0}\n".format(cfg.OSTYPE))
                fobj.write("KEYWORDS={0}\n".format(", ".join(self.keywords)))
                fobj.write("FILE={0}\n".format(
                        os.path.basename(self.test_file[:-1])))

        # Let the user know which test is running
        pu.log_message(
            "{0:<{1}}".format(self.name, WIDTH_TERM - WIDTH_INFO) +
            "{0:>{1}s}".format("RUNNING", WIDTH_INFO), pre="", noisy=True)

        self.retdir = os.getcwd()
        os.chdir(self.results_directory())
        return

    def copy_files_to_results_directory(self):
        self.test_files = []
        # copy input file, if any
        d = self.results_directory()
        if self.infile:
            infile = os.path.join(d, os.path.basename(self.infile))
            shutil.copyfile(self.infile, infile)
            self.test_files.append(infile)

        # copy the python test file and make it executable
        test_py_file = os.path.join(d, os.path.basename(self.test_file))
        shutil.copyfile(self.test_file, test_py_file)
        self.test_files.append(test_py_file)
        os.chmod(test_py_file, 0o750)

        # symlink the baseline file
        if self.baseline:
            for base_f in self.baseline:
                source = base_f
                link_name = os.path.join(d, os.path.basename(base_f))
                if not os.path.isfile(source):
                    pu.file_not_found(source)
                    continue
                if os.path.isfile(link_name):
                    pu.report_error(
                        "symlink destination {0} exists".format(link_name))
                    continue
                os.symlink(source, link_name)
                continue

        # symlink and auxilary files
        if self.aux_files:
            for aux_f in self.aux_files:
                os.symlink(aux_f, os.path.join(d, os.path.basename(aux_f)))
                continue

        return

    def finish_test(self):
        # Print output at completion
        info_string = "{0} ({1}s)".format(
            self.status.upper(), self.completion_time(f="{0:6.02f}"))
        pu.log_message(
            "{0:<{1}}".format(self.name, WIDTH_TERM - WIDTH_INFO) +
            "{0:>{1}s}".format(info_string, WIDTH_INFO), pre="", noisy=True)

        if self.postprocess or self.retcode in (self.diffcode, self.failcode):
            pu.log_message(
                "{0:<{1}}".format(self.name, WIDTH_TERM - WIDTH_INFO) +
                "{0:>{1}s}".format("POSTPROCESSING", WIDTH_INFO),
                pre="", noisy=True)
            # This try-except block is here to catch errors while
            # postprocessing (like optimization benchmarks that diff).
            try:
                self.postprocess_test_result()
                pu.log_message(
                    "{0:<{1}}".format(self.name, WIDTH_TERM - WIDTH_INFO) +
                    "{0:>{1}s}".format("POSTPROCESSED", WIDTH_INFO),
                    pre="", noisy=True)
            except:
                pu.log_message(
                    "{0:<{1}}".format(self.name, WIDTH_TERM - WIDTH_INFO) +
                    "{0:>{1}s}".format("ERROR", WIDTH_INFO),
                    pre="", noisy=True)

        os.chdir(self.retdir)
        return

    def postprocess_test_result(self):
        import matplotlib.pyplot as plt
        aspect_ratio = 4.0/3.0
        try:
            ohead = pu.get_header(self.outfile)
            odat = pu.read_data(self.outfile)
        except IOError:
            pu.file_not_found(self.outfile, count=False)
            return
        except ValueError:
            # When loadtxt tries to convert to float and fails
            # do nothing.
            return
        try:
            ghead = pu.get_header(self.baseline[0])
            gdat = pu.read_data(self.baseline[0])
        except IOError:
            pu.file_not_found(self.baseline[0], count=False)
            return

        self.postprocess = os.path.splitext(self.outfile)[0] + ".html"
        plotd = os.path.splitext(self.outfile)[0] + ".post"

        # Make output directory for plots
        try: shutil.rmtree(plotd)
        except OSError: pass
        os.mkdir(plotd)

        xvar = "TIME"
        ox = odat[:, ohead.index(xvar)]
        gx = gdat[:, ghead.index(xvar)]

        plots = []
        for yvar in [x for x in sorted(ghead) if x not in self.items_to_skip]:

            if yvar == xvar:
                continue

            name = yvar + ".png"
            f = os.path.join(plotd, name)
            gy = gdat[:, ghead.index(yvar)]
            try:
                oy = odat[:, ohead.index(yvar)]
            except ValueError:
                continue

            plt.clf() # Clear the current figure
            plt.cla() # Clear the current axes
            plt.xlabel("TIME")
            plt.ylabel(yvar)

            plt.plot(gx, gy, ls="-", lw=4, c="orange", label="gold")
            plt.plot(ox, oy, ls="-", lw=2, c="green", label="out")
            plt.legend()
            plt.gcf().set_size_inches(aspect_ratio * 5, 5.)
            plt.savefig(f, dpi=100)
            plots.append(f)

        with open(self.postprocess, "w") as fobj:
            fobj.write("""\
    <html>
      <head>
        <title>{0}</title>
      </head>
      <body>
        <table>
          <tr>
    """.format(self.name))

            for i, plot in enumerate(plots):
                name = os.path.basename(plot)
                if i % 3 == 0 and i != 0:
                    fobj.write("      </tr>\n       <tr>\n")
                width = str(int(aspect_ratio * 300))
                height = str(int(300))
                fobj.write("""\
            <td>
              {0}
              <a href="{1}">
                <img src="{1}" width="{2}" height="{3}">
              </a>
            </td>
    """.format(name, plot, width, height))

            fobj.write("      </tr>\n    </table>\n  </body>\n</html>")
        return

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

        errors = 0
        # open the log file
        log = TestLogger(self.name + ".diff", "w")

        if "baselinef" not in globals():
            baselinef = self.baseline[0]
            pass

        if not os.path.isfile(baselinef):
            log.error("baseline file {0} not found for {1}"
                      .format(baselinef, self.name))
            errors += 1
            pass

        if "outf" not in globals():
            outf = self.outfile
            pass

        if not os.path.isfile(outf):
            log.error("output file {0} not found for {1}".format(outf, self.name))
            errors += 1
            pass

        if errors:
            return self.badincode

        # Determine which columns will be analyzed
        if type(self.rms_sets) != list:
            log.error("No proper RMS sets to analyze for this benchmark.\n" +
                      "Please set self.rms_sets in the input file. Here is an example:\n" +
                      "self.rms_sets=[ [\"VAR1\", \"VAR2\"], " +
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
                log.error("RMS set incorrectly defined.\n" +
                          "Expected list '[\"VAR1\",\"VAR2\"]', got '{0}'".
                          format(repr(rms_set)))
                return self.badincode
        self.rms_sets = [[x[0].lower(), x[1].lower()] for x in self.rms_sets]

        # read in header
        outheader = [x.lower() for x in self.get_header(outf)]
        goldheader = [x.lower() for x in self.get_header(baselinef)]

        for rms_set in self.rms_sets:
            if rms_set[0] not in outheader:
                errors += 1
                log.error("'{0}' not in {1}\nValid headers: {2}"
                          .format(rms_set[0], outf, repr(outheader)))
            if rms_set[1] not in outheader:
                errors += 1
                log.error("'{0}' not in {1}\nValid headers: {2}"
                          .format(rms_set[1], outf, repr(outheader)))
            if rms_set[0] not in goldheader:
                errors += 1
                log.error("'{0}' not in {1}\nValid headers: {2}"
                          .format(rms_set[0], baselinef, repr(goldheader)))
            if rms_set[1] not in goldheader:
                errors += 1
                log.error("'{0}' not in {1}\nValid headers: {2}"
                          .format(rms_set[1], baselinef, repr(goldheader)))
        if errors:
            return self.badincode

        # read in data
        out = pu.read_data(outf)
        gold = pu.read_data(baselinef)

        # check that the gold file has at least two points and that the
        # simulation file has at least one.
        if len(out[:, 0]) <= 0:
            errors += 1
            log.error("Not enough points in {0} (must have at least one)."
                      .format(outf))
            log.write("\n{0:=^72s}".format(" FAIL "))
            pass

        if len(gold[:, 0]) <= 1:
            errors += 1
            log.error("Not enough points in {0} (must have at least one)."
                      .format(baselinef))
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

        failed, diffed = [], []
        for rms_set in self.rms_sets:
            gidx = goldheader.index(rms_set[0])
            oidx = outheader.index(rms_set[0])
            gjdx = goldheader.index(rms_set[1])
            ojdx = outheader.index(rms_set[1])

            rmsd, nrmsd = pu.compute_rms_closest_point_residual(
                gold[:, gidx], gold[:, gjdx],
                out[:, oidx], out[:, ojdx])
            # Check to see if either of the error measures are less
            # than the tolerances.
            minerror = min(rmsd, nrmsd)
            if minerror >= self.failtol:
                failed.append("{0}:{1}".format(rms_set[0], rms_set[1]))
                stat = "FAIL"

            elif minerror >= self.difftol:
                diffed.append("{0}:{1}".format(rms_set[0], rms_set[1]))
                stat = "DIFF"

            else:
                stat = "PASS"

            # For good measure, write both the RMSD and normalized RMSD
            log.write(
                "headers: {0}:{1} - {2}".format(rms_set[0], rms_set[1], stat))
            log.write("  Unscaled error: {0:.10e}".format(rmsd))
            log.write("    Scaled error: {0:.10e}".format(nrmsd))
            continue

        if failed:
            msg = textwrap.fill(", ".join(failed), 72,
                                initial_indent=SP, subsequent_indent=SP)
            log.write("\nFAILED VARIABLES:\n{0}".format(msg))

        if diffed:
            msg = textwrap.fill(", ".join(diffed), 72,
                                initial_indent=SP, subsequent_indent=SP)
            log.write("\nDIFFED VARIABLES:\n{0}".format(msg))

        if failed:
            log.write("\n{0:=^72s}".format(" FAIL "))

        elif diffed:
            log.write("\n{0:=^72s}".format(" DIFF "))

        else:
            log.write("\n{0:=^72s}".format(" PASS "))

        del log
        return self.get_retcode(failed, diffed)

    def compare_out_to_baseline_rms(self, baselinef=None, outf=None):
        """
            Compare results from out file to those in baseline

            OUTPUT
                0: passed
                1: bad input
                2: diffed
                3: failed
        """
        errors = 0

        # open the log file
        log = TestLogger(self.name + ".diff", "w")

        if not baselinef:
            baselinef = self.baseline[0]
            pass

        if not os.path.isfile(baselinef):
            log.error("baseline file {0} not found for {1}"
                      .format(baselinef, self.name))
            errors += 1
            pass

        if not outf:
            outf = self.outfile
            pass

        if not os.path.isfile(outf):
            log.error("output file {0} not found for {1}"
                      .format(outf, self.name))
            errors += 1
            pass

        if errors:
            return self.badincode

        # read in header
        outheader = [x.lower() for x in self.get_header(outf)]
        goldheader = [x.lower() for x in self.get_header(baselinef)]

        if outheader[0] != "time":
            errors += 1
            log.error("time not first column of {0} for {1}"
                      .format(outf, self.name))
        if goldheader[0] != "time":
            errors += 1
            log.error("time not first column of {0} for {1}"
                      .format(baselinef, self.name))

        if errors:
            return self.badincode

        # read in data
        out = pu.read_data(outf)
        gold = pu.read_data(baselinef)

        # check that time is same (lengths must be the same)
        if len(gold[:, 0]) == len(out[:, 0]):
            rmsd, nrmsd = pu.compute_fast_rms(gold[:, 0], out[:, 0])

        else:
            rmsd, nrmsd = 1.0e99, 1.0e99

        if nrmsd > np.finfo(np.float).eps:
            errors += 1
            log.error("time step error between {0} and {1}"
                      .format(outf, baselinef))
            log.write("\n{0:=^72s}".format(" FAIL "))

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
                ["KAPPA", "EOS1", "EOS2", "EOS3", "EOS4",
                 "PLROOTJ2", "SNDSP", "ENRGY", "RHO", "TMPR"])

        to_compare_average = [x.lower() for x in self.expect_constant]
        to_skip = [x.lower() for x in self.items_to_skip]

        if not self.items_to_compare:
            to_compare = [x for x in outheader if x in goldheader
                          and x not in to_skip]

        else:
            to_compare = [x.lower() for x in self.items_to_compare]

        failed, diffed = [], []
        for val in to_compare:
            gidx = goldheader.index(val)
            oidx = outheader.index(val)

            if val in to_compare_average:
                # This is needed for comparing state variables that are
                # expected to be constant but might have noise that would
                # otherwise make it diff or fail (like noise in sig_n for
                # plane stress problems).
                val = val + " (expect const)"
                g_avg = sum(gold[:, gidx]) / len(gold[:, gidx])
                o_avg = sum(out[:, oidx]) / len(out[:, oidx])
                avg_diff = abs(g_avg - o_avg)
                g_stddev = sum([(x - g_avg) ** 2 for x in gold[:, gidx]])
                g_stddev = math.sqrt(g_stddev / len(gold[:, gidx]))

                tmptol = max(self.failtol, self.failtol * abs(g_avg))
                if avg_diff > 0.2 * g_stddev and avg_diff >= tmptol:
                    diffed.append(val)
                    stat = "FAIL"

                else:
                    stat = "PASS"

                log.write("{0}: {1}".format(val, stat))
                log.write("    Average diff: {0:.10e}".format(avg_diff))
                log.write("     Gold STDDEV: {0:.10e}".format(g_stddev))
                log.write("       tolerance: {0:.10e}".format(tmptol))
                continue

            else:
                rmsd, nrmsd = pu.compute_rms(gold[:, 0], gold[:, gidx],
                                             out[:, 0], out[:, oidx])

                # For good measure, write both the RMSD and normalized RMSD
                if nrmsd >= self.failtol:
                    failed.append(val)
                    stat = "FAIL"

                elif nrmsd >= self.difftol:
                    diffed.append(val)
                    stat = "DIFF"

                else:
                    stat = "PASS"

                log.write("{0}: {1}".format(val, stat))
                log.write("  Unscaled error: {0:.10e}".format(rmsd))
                log.write("    Scaled error: {0:.10e}".format(nrmsd))
                continue

        if failed:
            msg = textwrap.fill(", ".join(failed), 72,
                                initial_indent=SP, subsequent_indent=SP)
            log.write("\nFAILED VARIABLES:\n{0}".format(msg))

        if diffed:
            msg = textwrap.fill(", ".join(diffed), 72,
                                initial_indent=SP, subsequent_indent=SP)
            log.write("\nDIFFED VARIABLES:\n{0}".format(msg))

        if failed:
            log.write("\n{0:=^72s}".format(" FAIL "))

        elif diffed:
            log.write("\n{0:=^72s}".format(" DIFF "))

        else:
            log.write("\n{0:=^72s}".format(" PASS "))

        del log
        return self.get_retcode(failed, diffed)

    def get_header(self, f):
        """ get the header of f """
        return linecache.getline(f, 1).split()

    def clean_tracks(self):
        for ext in [".out", ".diff", ".log", ".prf", ".pyc", ".echo",
                    ".props", ".pyc", ".math1", ".math2"]:
            try:
                os.remove(self.name + ext)
            except:
                pass
            continue
        return

    def compare_constant_strain_at_failure(self, outf=None, epsfail=None):
        """ compare the constant strain at failure with expected """

        errors = 0

        # open the log file
        log = TestLogger(self.name + ".diff", "w")

        if outf:
            if not os.path.isfile(outf):
                errors += 1
                log.error("sent output file not found")
        else:
            outf = self.outfile
            pass

        if not outf:
            errors += 1
            log.error("not out file given")
            pass

        propf = self.name + ".props"
        if not os.path.isfile(propf):
            errors += 1
            log.error("{0} not found".format(propf))
            pass

        # get constant strain at failure
        if not epsfail:
            props = open(propf, "r").readlines()
            for prop in props:
                prop = [x.strip() for x in prop.split("=")]
                if prop[0].lower() == "fail2":
                    epsfail = float(prop[1])
                    break
                continue
            pass
        try:
            epsfail = float(epsfail)
        except TypeError:
            errors += 1
            log.error("epsfail must be float, got {0}".format(epsfail))
        except:
            errors += 1
            log.error("bad epsfail [{0}]".format(epsfail))

        # read in header
        outheader = [x.lower() for x in self.get_header(outf)]

        if outheader[0] != "time":
            errors += 1
            log.error("time not first column of {0}".format(outf))
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
        coher_idx = outheader.index("coher")
        accstrain_idx = outheader.index("accstrain")
        fail_idx = -1
        for idx, val in enumerate(out[:, coher_idx]):
            if val < 0.5:
                fail_idx = idx
                break
            continue

        if fail_idx == -1:
            log.error("COHER did not drop below 0.5.\n")
            log.error("Final value of COHER: {0}\n".format(out[-1, coher_idx]))
            return self.failcode

        # Perform an interpolation between COHER-ACCSTRAIN sets to find the
        # ACCSTRAIN when COHER=0.5 . Then compute the absolute and relative
        # errors.
        x0, y0 = out[fail_idx - 1, coher_idx], out[fail_idx - 1, accstrain_idx]
        x1, y1 = out[fail_idx, coher_idx], out[fail_idx, accstrain_idx]

        strain_f = y0 + (0.5 - x0) * (y1 - y0) / (x1 - x0)
        abs_err = abs(strain_f - epsfail)
        rel_err = abs_err / abs(max(strain_f, epsfail))

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

        if failed:
            log.write("\n{0:=^72s}".format(" FAIL "))

        elif diffed:
            log.write("\n{0:=^72s}".format(" DIFF "))

        else:
            log.write("\n{0:=^72s}".format(" PASS "))

        del log
        return self.get_retcode(failed, diffed)

    def runFromTerminal(self, argv):

        if "--cleanup" in argv or "-c" in argv:
            sys.exit(self.clean_tracks())

        self.setup_test(os.getcwd(), False, None)
        self.run_test()
        self.finish_test()
        return

    def diff_files(self, gold=None, out=None):

        """ compare gold with out """

        import difflib

        if gold is None:
            gold = self.baseline

        if out is None:
            out = self.outfile

        # open the log file
        log = TestLogger(self.name + ".diff", "w")

        errors = 0
        if not isinstance(gold, (list, tuple)):
            gold = [gold]

        for goldf in gold:
            if not os.path.isfile(goldf):
                log.error("{0} not found".format(goldf))
                errors += 1
            continue

        if not isinstance(out, (list, tuple)):
            out = [out]

        for outf in out:
            if not os.path.isfile(outf):
                log.error("{0} not found".format(outf))
                errors += 1
            continue

        if len(gold) != len(out):
            errors += 1
            log.error("len(gold) != len(out)")
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
                ddiff = difflib.ndiff(xout, xgold)
                diff += 1
                log.write("ERROR: {0} diffed from {1}:\n".format(bout, bgold))
                log.write("".join(ddiff))
            else:
                log.write("PASSED")
                pass
            continue

        del log
        if diff:
            return self.failcode

        return self.passcode

    def compare_opt_params(self, gold=None, out=None):
        """ compare gold with out """

        if gold is None:
            gold = self.baseline

        if out is None:
            out = self.outfile

        # open the log file
        log = TestLogger(self.name + ".diff", "w")

        errors = 0
        if not isinstance(gold, (list, tuple)):
            gold = [gold]

        for goldf in gold:
            if not os.path.isfile(goldf):
                log.error("{0} not found".format(goldf))
                errors += 1
            continue

        if not isinstance(out, (list, tuple)):
            out = [out]

        for outf in out:
            if not os.path.isfile(outf):
                log.error("{0} not found".format(outf))
                errors += 1
            continue

        if len(gold) != len(out):
            errors += 1
            log.error("len(gold) != len(out)")
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

            gold_params, out_params = {}, {}
            for line in xout[1:]:
                if not line.split() or line.strip()[0] in ("#", ):
                    continue
                nam, val = line.split("=")
                out_params[nam] = float(val)
            for line in xgold[1:]:
                if not line.split() or line.strip()[0] in ("#", ):
                    continue
                nam, val = line.split("=")
                gold_params[nam] = float(val)
            errors = []
            log.write("{0:12s}\t{1:12s}\t{2:12s}\t{3:12s}"
                      .format("PARAM", "GOLD", "OPT", "ERROR"))
            for gp in sorted(gold_params.keys()):
                gv = gold_params[gp]
                try:
                    ov = out_params[gp]
                except KeyError:
                    diff += 1
                    log.error("{0} not in output".format(param))
                    continue
                dnom = gv if abs(gv) > np.finfo(np.float).eps else 1.
                error = abs(ov - gv) / dnom
                errors.append(error)
                log.write("{0:12s}\t{1:12.6E}\t{2:12.6E}\t{3:12.6E}"
                          .format(gp, gv, ov, error))
            merror = max(errors)

        diffed, failed = False, False
        if merror >= self.failtol:
            log.write("\n{0:=^72s}".format(" FAIL "))
            failed = True
        elif merror >= self.difftol:
            log.write("\n{0:=^72s}".format(" DIFFED "))
            diffed = True
        else:
            log.write("\n{0:=^72s}".format(" PASSED "))
            passed = True
            pass

        if diff or diffed:
            return self.diffcode
        elif failed:
            return self.failcode

        return self.passcode

    def set_results_directory(self, *args):
        if not args:
            self._results_directory = os.getcwd()
            return
        self._results_directory = os.path.join(*args)
        return

    def results_directory(self):
        if self._results_directory is None:
            self._results_directory = os.getcwd()
        return self._results_directory

    def copy_mathematica_nbs(self):
        """copy the mathematica notebooks to the destination directory"""
        d, dd = os.path.split(self.results_directory())
        if dd != self.name:
            d = dd
        d = d + os.sep
        destnbs = [os.path.join(d, os.path.basename(x)) for x in self.nbs]
        if all(os.path.isfile(x) for x in destnbs):
            return
        for nb, destnb in zip(self.nbs, destnbs):
            try: os.remove(destnb)
            except OSError: pass
            if destnb.endswith(".m"):
                # don't copy the .m file, but write it, replacing rundir and
                # demodir with destdir
                with open(destnb, "w") as fobj:
                    for line in open(nb, "r").readlines():
                        if r"$DEMODIR" in line:
                            line = 'demodir="{0:s}"\n'.format(d)
                        elif r"$RUNDIR" in line:
                            line = 'rundir="{0:s}"\n'.format(d)
                        fobj.write(line)
                        continue
                continue

            # copy the notebook files
            shutil.copyfile(nb, destnb)
            continue
        return

    def find_nbs(self):
        """Find mathematica notebooks for this test"""
        d = self.test_file_dir
        def isnb(f):
            return {".nb": True, ".m": True}.get(os.path.splitext(f)[1], False)
        self.nbs = [os.path.join(d, x) for x in os.listdir(d) if isnb(x)]
        return


def find_tests(kws, skip, reqtests, test_dirs=None):
    """ find the Payette tests

    Parameters
    ----------
    kws : list
       list of requested keywords
    skip : list
       list of keywords to skip
    reqtests : list
       list of specific tests to find

    Returns
    -------
       found_tests : dict

    """

    if test_dirs is not None:
        errors = 0
        for test_dir in test_dirs:
            if not os.path.isdir(test_dir):
                pu.report_error(
                    "test directory {0} not found".format(test_dir))
            continue
        if pu.error_count():
            pu.report_and_raise_error("stopping due to previous errors")
    else:
        test_dirs = [cfg.TESTS]

    # get all potential test files
    py_files  = []
    for test_dir in test_dirs:
        for d, dirs, files in os.walk(test_dir):
            if ".svn" in d or "__test_dir__.py" not in files:
                continue
            py_files.extend([os.path.join(d, x) for x in files if ispyfile(x)])
            continue
        continue

    # weed out only those that are tests
    kws = [x.lower() for x in kws]
    skip = [x.lower() for x in skip]
    reqtests = [x.lower() for x in reqtests]
    tests = []
    for py_file in py_files:

        test = get_test(py_file)

        if test is None:
            continue

        if test in tests:
            pu.log_warning("{0}: duplicate test.  skipping".format(test.name))
            continue

        if not test.enabled:
            pu.log_warning("disabled test: {0} encountered".format(test.name))
            continue

        if reqtests:
            # only requested tests will be collected
            if test.name.lower() in reqtests:
                tests.append(test)
            continue

        tkws = [x.lower() for x in test.keywords]

        # check that all requested keywords match
        if kws and not all(x in tkws for x in kws):
            continue

        # check if any keywords to skip match
        if skip and any(x in tkws for x in skip):
            continue

        tests.append(test)
        continue

    model_index = pmi.ModelIndex()
    for test in tests:
        # check if the material model used in the test is installed
        if test.material is not None:
            if test.material not in model_index.constitutive_models():
                pu.log_warning(
                    "material model" + " '" + test.material + "' " +
                    "required by" + " '" + test.name + "' " +
                    "not installed, test will be skipped")
                tests.remove(test)

    # return tests sorted according to speed (fastest first)
    return sorted(tests, key=lambda x: SPEED_KWS[x.speed])


def get_test(py_file):
    if not py_file.endswith(".py"):
        return
    # get and load module
    try:
        py_module = pu.load_module(py_file)
    except ImportError:
        pu.log_warning("{0}: not importable".format(os.path.basename(py_file)))
        return
    # check if a payette test class is defined
    fnam = os.path.basename(py_file)
    try:
        cls = py_module.Test
    except AttributeError:
        pu.log_warning("{0}: Test class not found".format(fnam))
        return
    if PayetteTest not in cls.__bases__:
        pu.report_error("{0}: Test not a subclass of PayetteTest".format(fnam))
        return
    # instantiate and return the test object
    test = cls()
    return test


def ispyfile(fpath):
    fbase, fext = os.path.splitext(os.path.basename(fpath))
    # filter out all files we know cannot be test files
    if fext != ".py":
        return False
    if fbase[0] == ".":
        return False
    regex = r"(__init__|__test_dir__|config|template)"
    if re.search(regex, fbase):
        return False
    return True


if __name__ == "__main__":

    found_tests = find_tests(["elastic", "fast"], [], [])

    fast_tests = [x.name for x in found_tests if "fast" in x.keywords]
    medium_tests = [x.name for x in found_tests if "medium" in x.keywords]
    long_tests = [x.name for x in found_tests if "long" in x.keywords]
    print(fast_tests)
    print(medium_tests)
    print(long_tests)
    sys.exit()
