#!/usr/bin/env python

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

"""Main Payette testing file.
None of the functions in this file should be called directly, but only through
the executable script in $PC_ROOT/Toolset/testPayette

AUTHORS
Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov

"""

from __future__ import print_function
import sys
import imp
import os
import optparse
import time
import shutil
import platform
import multiprocessing as mp
from shutil import copyfile, rmtree
import datetime
import getpass

import Payette_config as pc
import Source.Payette_utils as pu
import Source.Payette_notify as pn
from Source.Payette_test import find_tests
import Toolset.postprocess as pp

# --- module level variables
CWD = os.getcwd()
TESTRESDIR = os.path.join(CWD, "TestResults.{0}".format(platform.system()))
RANTESTS = []
POSTPROCESS = False
FORCERERUN = False
WIDTH_TERM = 80
WIDTH_INFO = 25


def test_payette(argv):

    """Run the Payette benchmarks.

    Walk through and run the Payette test simulations, compare results
    against the accepted results.

    """

    global RANTESTS, POSTPROCESS, FORCERERUN

    # *************************************************************************
    # -- command line option parsing
    usage = "usage: testPayette [options]"
    parser = optparse.OptionParser(usage=usage, version="testPayette 1.0")
    parser.add_option(
        "-k",
        dest="KEYWORDS",
        action="append",
        default=[],
        help="keywords: [%default]")
    parser.add_option(
        "-K",
        dest="NOKEYWORDS",
        action="append",
        default=[],
        help="keyword negation: [%default]")
    parser.add_option(
        "-t",
        dest="SPECTESTS",
        action="append",
        default=[],
        help="specific tests to run, more than 1 collected: [%default]")
    parser.add_option(
        "-d",
        dest="BENCHDIRS",
        action="append",
        default=[],
        help=("Additional directories to scan for benchmarks, accumulated "
              "[default: %default]."))
    parser.add_option(
        "-i", "--index",
        dest="INDEX",
        action="store_true",
        default=False,
        help="Print benchmarks index [default: %default].")
    parser.add_option(
        "-F",
        dest="FORCERERUN",
        action="store_true",
        default=False,
        help="Force benchmarks to be run again [default: %default].")
    parser.add_option(
        "-j", "--nproc",
        dest="nproc",
        type=int,
        default=1,
        action="store")
    parser.add_option(
        "-b", "--buildpayette",
        dest="buildpayette",
        action="store_true",
        default=False,
        help="build payette [default: %default].")
    parser.add_option(
        "-p", "--postprocess",
        dest="POSTPROCESS",
        action="store_true",
        default=False,
        help="Generate plots for run tests [default: %default]")
    parser.add_option(
        "--notify",
        dest="NOTIFY",
        action="store_true",
        default=False,
        help="email test results to mailing list [default: %default].")
    parser.add_option(
        "-I",
        dest="IGNOREERROR",
        action="store_true",
        default=False,
        help="Ignore noncomforming tests [default: %default]")

    (opts, args) = parser.parse_args(argv)

    if args:
        sys.exit("ERROR: testPayette does not take any arguments")

    # number of processors
    nproc = min(mp.cpu_count(), opts.nproc)

    pu.logmes(pc.PC_INTRO)

    # pass user option to global variables
    FORCERERUN = opts.FORCERERUN
    POSTPROCESS = opts.POSTPROCESS

    # find tests
    errors = 0
    pu.loginf("Testing Payette")
    test_dirs = pc.PC_TESTS
    for dirnam in opts.BENCHDIRS:
        dirnam = os.path.expanduser(dirnam)
        if not os.path.isdir(dirnam):
            errors += 1
            pu.logerr("benchmark directory {0} not found".format(dirnam))
        elif pu.check_if_test_dir(dirnam):
            test_dirs.append(dirnam)
        else:
            errors += 1
            pu.logerr("__test_dir__.py not found in {0}".format(dirnam))
        continue
    if errors:
        sys.exit("ERROR: stopping due to previous errors")
    pu.loginf("Gathering Payette tests from {0}".format(", ".join(test_dirs)))
    errors, found_tests = find_tests(opts.KEYWORDS, opts.NOKEYWORDS,
                                     opts.SPECTESTS, test_dirs)

    # sort conforming tests from long to fast
    fast_tests = [val for key, val in found_tests["fast"].items()]
    medium_tests = [val for key, val in found_tests["medium"].items()]
    long_tests = [val for key, val in found_tests["long"].items()]
    conforming = long_tests + medium_tests + fast_tests

    # find mathematica notebooks
    mathnbs = {}
    for test_dir in test_dirs:
        for item in os.walk(test_dir):
            dirnam, files = item[0], item[2]
            root, base = os.path.split(dirnam)
            if base == "nb":
                # a notebook directory has been found, see if there are any
                # conforming tests that use it
                if [x for x in conforming if root in x]:
                    mathnbs[root.split(os.sep)[-1]] = [
                        os.path.join(dirnam, x) for x in files
                        if x.endswith(".nb") or x.endswith(".m")]
            continue
        continue

    if errors and not opts.IGNOREERROR:
        sys.exit("fix nonconforming benchmarks before continuing")

    pu.loginf("Found {0} Payette tests".format(len(conforming)), end="\n\n")

    if opts.INDEX:
        out = sys.stderr
        out.write("\n\nBENCHMARK INDEX\n\n")
        for key in found_tests:
            tests = found_tests[key]
            for py_mod, py_file in tests.items():
                # load module
                py_path = [os.path.dirname(py_file)]
                fobj, pathname, description = imp.find_module(py_mod, py_path)
                py_module = imp.load_module(py_mod, fobj, pathname, description)
                fobj.close()
                test = py_module.Test()
                if not test.checked:
                    test.check_setup()
                out.write(WIDTH_TERM * "=" + "\n")
                out.write("Name:  {0}\n".format(test.name))
                out.write("Owner: {0}\n\n".format(test.owner))
                out.write("Description:\n{0}".format(test.description))

                out.write("\nKeywords:\n")
                for key in test.keywords:
                    out.write("    {0}\n".format(key))
                continue
            continue

        return 0

    # start the timer
    runtimer = time.time()

    # Make a TestResults directory named "TestResults.{platform}"
    if opts.buildpayette:
        if os.path.isdir(TESTRESDIR):
            testresdir0 = "{0}_0".format(TESTRESDIR)
            shutil.move(TESTRESDIR, testresdir0)

    if not os.path.isdir(TESTRESDIR):
        os.mkdir(TESTRESDIR)

    old_results = {}
    summpy = os.path.join(TESTRESDIR, "summary.py")
    summhtml = os.path.splitext(summpy)[0] + ".html"
    if os.path.isfile(summpy):
        py_path = [os.path.dirname(summpy)]
        py_mod = pu.get_module_name(summpy)
        fobj, pathname, description = imp.find_module(py_mod, py_path)
        py_module = imp.load_module(py_mod, fobj, pathname, description)
        fobj.close()
        try:
            old_results = py_module.payette_test_results
        except AttributeError:
            copyfile(summpy,
                     os.path.join(TESTRESDIR, "summary_orig.py"))
        try:
            os.remove("{0}c".format(summpy))
        except OSError:
            pass

    # check type of old_results, if not dict start over with empty dict
    if not isinstance(old_results, dict):
        old_results = {}

    # Put all run tests in a flattened list for checking if test has
    # been run
    if old_results:
        RANTESTS = [x for y in old_results.values() for x in y]

    # reset the test results
    test_statuses = ["pass", "diff", "fail", "notrun",
                     "bad input", "failed to run"]
    test_res = {}
    for i in test_statuses:
        test_res[i] = {}
        continue

    pu.logmes("=" * WIDTH_TERM)
    pu.logmes("Running {0} benchmarks:".format(len(conforming)))
    pu.logmes("=" * WIDTH_TERM)

    # run the tests on multiple processors using the multiprocessor map ONLY f
    # nprocs > 1. For debug purposes, when nprocs=1, run without using the
    # multiprocessor map because it makes debugging worse than it should be.
    if nproc == 1:
        all_results = [run_payette_test(test) for test in conforming]

    else:
        pool = mp.Pool(processes=nproc)
        all_results = pool.map(run_payette_test, conforming)
        pool.close()
        pool.join()

    ttot = time.time() - runtimer
    pu.logmes("=" * WIDTH_TERM)

    # copy the mathematica notebooks to the output directory
    for mtldir, mathnb in mathnbs.items():
        for item in mathnb:
            fbase = os.path.basename(item)
            fold = os.path.join(TESTRESDIR, mtldir, fbase)

            try:
                os.remove(fold)
            except OSError:
                pass

            if item.endswith(".m"):
                # don't copy the .m file, but write it, replacing rundir and
                # demodir with TESTRESDIR
                with open(fold, "w") as fobj:
                    for line in open(item, "r").readlines():
                        demodir = os.path.join(TESTRESDIR, mtldir) + os.sep
                        rundir = os.path.join(TESTRESDIR, mtldir) + os.sep
                        if r"$DEMODIR" in line:
                            line = 'demodir="{0:s}"\n'.format(demodir)
                        elif r"$RUNDIR" in line:
                            line = 'rundir="{0:s}"\n'.format(rundir)
                        fobj.write(line)
                        continue

                continue

            else:
                # copy the notebook files
                shutil.copyfile(item, fold)

            continue

        continue

    # all_results is a large list of the summary of every test.
    # Go through it and use the information to construct the test_res
    # dictionary of the form:
    #    test_res["notrun"] = ...
    #    test_res["pass"] = ...
    #    test_res["diff"] = ...
    #    test_res["failed"] = ...
    for item in all_results:
        for name, summary in item.items():
            status = summary["status"]
            if status not in test_statuses:
                msg = ("return code {0} from {1} not recognized"
                       .format(status, name))
                pu.logwrn(msg)
                continue
            tcompletion = summary["completion time"]
            benchdir = summary["benchmark directory"]
            keywords = summary["keywords"]
            test_res[status][name] = {"benchmark directory": benchdir,
                                      "completion time": tcompletion,
                                      "keywords": keywords}
            continue
        continue

    nnotrun = len(test_res["notrun"])
    nfail = len(test_res["fail"])
    npass = len(test_res["pass"])
    ndiff = len(test_res["diff"])
    nfailtorun = len(test_res["failed to run"])
    txtsummary = (
        "SUMMARY\n" +
        "{0} benchmarks took {1:.2f}s.\n".format(len(conforming), ttot) +
        "{0} benchmarks passed\n".format(npass) +
        "{0} benchmarks diffed\n".format(ndiff) +
        "{0} benchmarks failed\n".format(nfail) +
        "{0} benchmarks failed to run\n".format(nfailtorun) +
        "{0} benchmarks not run\n".format(nnotrun) +
        "For a summary of which benchmarks passed, diffed, failed, " +
        "or not run, see\n{0}".format(summhtml))

    # Make a long summary including the names of what passed and
    # what didn't as well as system information.
    str_date = datetime.datetime.today().strftime("%A, %d. %B %Y %I:%M%p")

    longtxtsummary = (
         "=" * WIDTH_TERM + "\nLONG SUMMARY\n" +
         "{0} benchmarks took {1:.2f}s.\n".format(len(conforming), ttot) +
         "{0:^{1}}\n".format("{0:-^30}".format(" system information "),
                             WIDTH_TERM) +
         "   Date complete:    {0:<}\n".format(str_date) +
         "   Username:         {0:<}\n".format(getpass.getuser()) +
         "   Hostname:         {0:<}\n".format(os.uname()[1]) +
         "   Platform:         {0:<}\n".format(sys.platform) +
         "   Python Version:   {0:<}\n".format(pc.PC_PYVER))

    # List each category (diff, fail, notrun, and pass) and the tests
    test_result_statuses = test_res.keys()
    test_result_statuses.sort()
    for stat in test_result_statuses:
        names = test_res[stat].keys()
        header = "{0:-^30}".format(" " +
                                   stat +
                                   " ({0}) ".format(len(names)))
        longtxtsummary += "{0:^{1}}\n".format(header, WIDTH_TERM)
        if len(names) == 0:
#            longtxtsummary += "None\n"
            continue
        elif stat == "notrun":
#            longtxtsummary += "None\n"
            continue
        for name in names:
            try:
                tcmpl = ("{0:.2f}s."
                         .format(test_res[stat][name]["completion time"]))
            except ValueError:
                tcmpl = str(test_res[stat][name]["completion time"])

            longtxtsummary += "  {0:>8}   {1}\n".format(tcmpl, name)
            continue
        continue
    longtxtsummary += "=" * WIDTH_TERM + "\n"
    # longtxtsummary is finished at this point

    # This sends an email to everyone on the mailing list.
    if opts.NOTIFY:
        pu.logmes("Sending results to mailing list.")
        pn.notify("Payette Benchmarks", longtxtsummary)

    pu.logmes(longtxtsummary)
    pu.logmes("=" * WIDTH_TERM)
    pu.logmes(txtsummary)
    pu.logmes("=" * WIDTH_TERM)

    # write out the results to the summary file
    write_py_summary(summpy, test_res)
    write_html_summary(summhtml, test_res)

    # cleanup our tracks
    for test_dir in test_dirs:
        for item in os.walk(test_dir):
            dirnam, files = item[0], item[2]
            for name in files:
                fbase, fext = os.path.splitext(name)
                delext = [".so", ".pyo", ".pyc", ".log", ".out", ".prf"]
                if fext in delext:
                    os.remove(os.path.join(dirnam, name))
                continue
            continue
        continue

    if opts.buildpayette:
        shutil.rmtree(TESTRESDIR)
        try:
            shutil.move(testresdir0, TESTRESDIR)
        except:
            pass

    return 0


def run_payette_test(py_file):

    """ run the payette test in py_file """

    py_path = [os.path.dirname(py_file)]
    py_mod = pu.get_module_name(py_file)
    fobj, pathname, description = imp.find_module(py_mod, py_path)
    py_module = imp.load_module(py_mod, fobj, pathname, description)
    fobj.close()

    test = py_module.Test()

    # directory where test will be run. If the test was taken from the
    # PC_TESTS directory, we wish to keep the same directory structure in the
    # TestResults.OSTYPE directory. e.g., if the test came from
    # PC_TESTS/Regression, we want to run the test in
    # TestResults.OSTYPE/Regression. But, if the test did not come from the
    # PC_TESTS directory (user specified additional directories to look for
    # tests), we try to keep the base directory name of that test. e.g., if
    # the user asked to look for tests in /some/directory/to/special_tests, we
    # run the tests in TestResults.OSTYP/special_tests.
    if pc.PC_TESTS[0] in py_file:
        testbase = os.path.dirname(py_file).split(pc.PC_TESTS[0] + os.sep)[1]
    else:
        testbase = os.path.split(os.path.dirname(py_file))[1]
    benchdir = os.path.join(TESTRESDIR, testbase, test.name)

    # check if benchmark has been run
    ran = [x for x in RANTESTS if x == test.name]

    if not FORCERERUN and ran and os.path.isdir(benchdir):
        pu.logmes("{0}".format(test.name) +
                  " " * (50 - len(test.name)) +
                  "{0:>10s}".format("notrun\n") +
                  "Test already ran. " +
                  "Use -F option to force a rerun")
        result = {test.name: {"status": "notrun",
                              "keywords": test.keywords,
                              "completion time": "NA",
                              "benchmark directory": benchdir}}
        return result

    # Let the user know which test is running
    pu.logmes("{0:<{1}}".format(test.name, WIDTH_TERM - WIDTH_INFO) +
              "{0:>{1}s}".format("RUNNING", WIDTH_INFO))

    # Create benchmark directory and copy the input and baseline files into the
    # new directory
    if os.path.isdir(benchdir):
        rmtree(benchdir)
    os.makedirs(benchdir)

    # copy input file, if any
    if test.infile:
        copyfile(test.infile,
                 os.path.join(benchdir, os.path.basename(test.infile)))

    # copy the python test file and make it executable
    copyfile(py_file,
             os.path.join(benchdir, os.path.basename(py_file)))
    os.chmod(os.path.join(benchdir, os.path.basename(py_file)), 0o750)

    # symlink the baseline file
    if test.baseline:
        for base_f in test.baseline:
            source = base_f
            link_name = os.path.join(benchdir, os.path.basename(base_f))
            if not os.path.isfile(source):
                pu.logerr("cannot symlink to non-existant file\n"+
                          "os.symlink({0},{1})".format(repr(source),repr(link_name)))
            if os.path.isfile(link_name):
                pu.logerr("cannot create symlink when link destination exists\n"+
                          "os.symlink({0},{1})".format(repr(source),repr(link_name)))
            os.symlink(
                base_f,
                os.path.join(benchdir, os.path.basename(base_f)))
            continue

    # symlink and auxilary files
    if test.aux_files:
        for aux_f in test.aux_files:
            os.symlink(
                aux_f,
                os.path.join(benchdir, os.path.basename(aux_f)))
            continue

    # move to the new directory and run the test
    os.chdir(benchdir)
    starttime = time.time()

    retcode = test.runTest()

    if POSTPROCESS and os.path.isfile(test.outfile):
        pp.postprocess(test.outfile, verbosity=0)

    retcode = ("bad input" if retcode == test.badincode else
               "pass" if retcode == test.passcode else
               "diff" if retcode == test.diffcode else
               "fail" if retcode == test.failcode else
               "failed to run" if retcode == test.failtoruncode else
               "unkown")

    # Print output at completion
    tcompletion = time.time() - starttime
    info_string = "{0} ({1:6.02f}s)".format(retcode.upper(), tcompletion)
    pu.logmes("{0:<{1}}".format(test.name, WIDTH_TERM - WIDTH_INFO) +
              "{0:>{1}s}".format(info_string, WIDTH_INFO))

    # return to the directory we came from
    os.chdir(CWD)
    result = {test.name: {"status": retcode,
                          "keywords": test.keywords,
                          "completion time": tcompletion,
                          "benchmark directory": benchdir}}
    return result


def write_py_summary(fname, results):

    """ write summary of the results dictionary to python file """

    # the results dictionary is of the form
    # results = { {status: {name: { "benchmark directory":benchdir,
    #                               "completion time":tcompletion,
    #                               "keywords":keywords } } } }
    with open(fname, "w") as fobj:
        fobj.write("payette_test_results = {}\n")
        for stat in results:
            # key is one of diff, pass, fail, notrun
            # names are the names of the tests
            fobj.write("payette_test_results['{0}']=[".format(stat))
            fobj.write(",".join(["'{0}'".format(x)
                                 for x in results[stat].keys()]))
            fobj.write("]\n")
            continue

    return


def write_html_summary(fname, results):

    """ write summary of the results dictionary to html file """

    # the results dictionary is of the form
    # results = { {status: {name: { "benchmark directory":benchdir,
    #                               "completion time":tcompletion,
    #                               "keywords":keywords } } } }
    resd = os.path.dirname(fname)
    npass, nfail, ndiff, nskip = [len(results[x]) for x in
                               ["pass", "fail", "diff", "notrun"]]
    with open(fname, "w") as fobj:
        # write header
        fobj.write("<html>\n<head>\n<title>Test Results</title>\n</head>\n")
        fobj.write("<body>\n<h1>Summary</h1>\n")
        fobj.write("<ul>\n")
        fobj.write("<li> Directory: {0} </li>\n".format(resd))
        fobj.write("<li> Options: {0} </li>\n".format(" ".join(sys.argv[1:])))
        fobj.write(
            "<li> {0:d} pass, {1:d} diff, {2:d} fail, {3:d} notrun </li>\n"
            .format(npass, ndiff, nfail, nskip))
        fobj.write("</ul>\n")

        # write out details test that fail, diff, pass, notrun
        for stat in results.keys():
            fobj.write("<h1>Tests that showed '{0}'</h1>\n<ul>\n".format(stat))
            for test in results[stat]:
                tresd = results[stat][test]["benchmark directory"]
                fobj.write("<li>{0}  {1}</li>\n".format(test, tresd))
                fobj.write("<ul>\n")
                fobj.write("<li>Files: \n")
                files = os.listdir(tresd)
                for fnam in files:
                    fpath = os.path.join(tresd, fnam)
                    fobj.write("<a href='{0}' type='text/plain'>{1}</a> \n"
                               .format(fpath, fnam))
                    continue
                keywords = "  ".join(results[stat][test]["keywords"])

                try:
                    tcompletion = (
                        "{0:.2f}s."
                        .format(results[stat][test]["completion time"]))
                except ValueError:
                    tcompletion = str(results[stat][test]["completion time"])

                status = "Exit {0} {1}".format(stat, tcompletion)
                for myfile in files:
                    if myfile.endswith(".out.html") and POSTPROCESS:
                        fobj.write("<li><a href='{0}'>PostProcessing</a>\n"
                                   .format(os.path.join(tresd, myfile)))
                fobj.write("<li>Keywords: {0}\n".format(keywords))
                fobj.write("<li>Status: {0}\n".format(status))
                fobj.write("</ul>\n")
                continue
            fobj.write("</ul>\n")
            continue

    return


if __name__ == "__main__":

    ARGV = sys.argv[1:]
    if "--profile" in ARGV:
        PROFILE = True
        ARGV.remove("--profile")
        import cProfile
    else:
        PROFILE = False

    if PROFILE:
        CMD = "test_payette(ARGV)"
        PROF = "payette.prof"
        cProfile.runctx(CMD, globals(), locals(), PROF)
        TEST = 0
    else:
        TEST = test_payette(ARGV)

    sys.exit(TEST)

