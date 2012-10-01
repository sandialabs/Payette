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

"""Main Payette testing file.
None of the functions in this file should be called directly, but only through
the executable script in $PC_ROOT/Toolset/testPayette

AUTHORS
Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov

"""

import sys
import imp
import os
import optparse
import time
import shutil
import multiprocessing as mp
from shutil import copyfile, rmtree
from pickle import dump, load
import datetime
import getpass
import re

import Source.__config__ as cfg
import Source.Payette_utils as pu
import Source.Payette_notify as pn
from Source.Payette_test import find_tests
import Toolset.postprocess as pp
from Source.Payette_container import PayetteError as PayetteError

# --- module level variables
CWD = os.getcwd()
WIDTH_TERM = 80
WIDTH_INFO = 25
TEST_INFO = ".test.info"
RESDIR = os.path.join(CWD, "TestResults.{0}".format(cfg.OSTYPE))


def main(argv, print_help=False):
    """Parse user command line arguments and send them to test_payette """
    # *************************************************************************
    # -- command line option parsing
    usage = "usage: testPayette [options]"
    parser = optparse.OptionParser(usage=usage, version="testPayette 1.0")
    parser.add_option(
        "--builtin",
        dest="BUILTIN",
        action="store_true",
        default=False,
        help=": Run only 'builtin' tests [default: %default]")
    parser.add_option(
        "-k",
        dest="KEYWORDS",
        action="append",
        default=[],
        help="filter in tests that match keyword (accumlated): [default: %default]")
    parser.add_option(
        "-K",
        dest="NOKEYWORDS",
        action="append",
        default=[],
        help="filter out tests matching keyword (accumulated): [default: %default]")
    parser.add_option(
        "-t",
        dest="SPECTESTS",
        action="append",
        default=[],
        help="specific tests to run (accumulated) [default: %default]")
    parser.add_option(
        "-d",
        dest="BENCHDIRS",
        action="append",
        default=[],
        help=("Additional directories to scan for benchmarks, accumulated "
              "[default: %default]."))
    parser.add_option(
        "-D",
        dest="TESTRESDIR",
        action="store",
        default=RESDIR,
        help="Directory to run tests [default: %default].")
    parser.add_option(
        "-F",
        dest="FORCERERUN",
        action="store_true",
        default=False,
        help="Force benchmarks to be run again [default: %default].")
    parser.add_option(
        "-j", "--nproc",
        dest="NPROC",
        type=int,
        default=1,
        action="store")
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
    parser.add_option(
        "-u",
        dest="TESTFILE",
        action="store_true",
        default=False,
        help=("Use previously generated test file, -F is implied "
              "[default: %default]"))
    parser.add_option(
        "-S",
        dest="SWITCH",
        action="store",
        default=None,
        help=("Switch material A for B [usage -S'A:B'] [default: %default]"))
    parser.add_option(
        "-b", "--rebaseline",
        dest="REBASELINE",
        action="store_true",
        default=False,
        help="Rebaseline a test [default: %default]")

    # indexing
    parser.add_option(
        "-i", "--index",
        dest="INDEX",
        action="store_true",
        default=False,
        help="Print benchmarks index [default: %default].")
    parser.add_option(
        "--name",
        dest="INDEX_NAME",
        action="store_true",
        default=False,
        help="Print benchmark names, -i is implied [default: %default].")
    if print_help:
        parser.print_help()
        return
    (opts, args) = parser.parse_args(argv)
    sys.exit(test_payette(
            opts.TESTRESDIR, args, builtin=opts.BUILTIN, keywords=opts.KEYWORDS,
            nokeywords=opts.NOKEYWORDS, spectests=opts.SPECTESTS,
            benchdirs=opts.BENCHDIRS, forcererun=opts.FORCERERUN,
            nproc=opts.NPROC, postprocess=opts.POSTPROCESS,
            notify=opts.NOTIFY, ignoreerror=opts.IGNOREERROR,
            testfile=opts.TESTFILE, switch=opts.SWITCH,
            rebaseline=opts.REBASELINE, index=opts.INDEX,
            index_name=opts.INDEX_NAME))


def test_payette(testresdir, args, builtin=False, keywords=[], nokeywords=[],
                 spectests=[], benchdirs=[], forcererun=False, nproc=1,
                 postprocess=False, notify=False, ignoreerror=False,
                 testfile=False, switch=None, rebaseline=False, index=False,
                 index_name=False):
    """Run the Payette benchmarks.

    Walk through and run the Payette test simulations, compare results
    against the accepted results.

    Parameters
    ----------
    testresdir : str
        path to directory where tests are to be run
    args : list
        list of args
    builtin : bool {False}
        run only the builtin tests
    keywords : list {[]}
        run tests with these keywords
    nokeywords : list {[]}
        do not run tests with these keywords
    spectests : list {[]}
        run these specific tests by name
    benchdirs : list {[]}
        additional directories to search for tests
    forcererun : bool {False}
        force tests to be run that were already run
    nproc : int {1}
        number of simultaneous jobs to run
    postprocess : bool {False}
        post process results
    notify : bool {False}
        send notification email of results
    ignoreerror : bool {False}
        ignore errors and continue running tests
    testfile : bool {False}
        do not look for tests but use test previously logged
    switch : str {None}
        switch materials
    rebaseline : bool {False}
        rebaseline test
    index : bool {False}
    index_name : bool {False}
        write index [_name] and exit

    """
    if switch is not None:
        pu.log_warning("switching materials is an untested feature")

    # number of processors
    nproc = min(mp.cpu_count(), nproc)

    pu.log_message(cfg.INTRO, pre="", noisy=True)

    if rebaseline:
        sys.exit(rebaseline_tests(args))

    # find tests
    pu.log_message("Testing Payette", noisy=True)

    # if we are running testPayette in a test result directory, just run that
    # test
    if os.path.isfile(os.path.join(CWD, TEST_INFO)):
        lines = open(os.path.join(CWD, TEST_INFO)).readlines()
        for line in lines:
            line = line.split("=")
            if line[0] == "FILE":
                pyfile = line[1]
                break
        else:
            pu.report_error("FILE not found in {0}".format(test_info))
        _run_the_test(pyfile, postprocess=postprocess)
        return

    # if the user passed in python test files, just run those tests
    pyargs = [x for x in args if x.endswith(".py")]
    if pyargs:
        for pyarg in pyargs:
            if not os.path.isfile(pyarg):
                pu.report_error("{0} not found".format(pyarg))
            _run_the_test(pyarg, postprocess=postprocess)
            continue
        return

    test_dirs = cfg.TESTS
    for dirnam in benchdirs:
        dirnam = os.path.expanduser(dirnam)
        if not os.path.isdir(dirnam):
            pu.report_error("benchmark directory {0} not found".format(dirnam))
        elif "__test_dir__.py" not in os.listdir(dirnam):
            pu.report_error("__test_dir__.py not found in {0}".format(dirnam))
        else:
            test_dirs.append(dirnam)
        continue
    if pu.error_count():
        pu.report_and_raise_error("stopping due to previous errors")

    if builtin:
        keywords = ["builtin"]

    t_start = time.time()
    conforming = None
    if testfile:
        try:
            pu.log_message("Using Payette tests from\n{0}"
                           .format(" " * 6 + cfg.PREV_TESTS), noisy=True)
            conforming = load(open(cfg.PREV_TESTS, "r"))
            errors = 0
        except IOError:
            pu.log_warning(
                "test file {0} not imported".format(cfg.PREV_TESTS))

    if conforming is None:
        pu.log_message("Gathering Payette tests from\n{0}"
                       .format("\n".join([" " * 6 + x for x in test_dirs])),
                       noisy=True)
        errors, found_tests = find_tests(keywords, nokeywords,
                                         spectests, test_dirs)

        # sort conforming tests from long to fast
        fast_tests = [val for key, val in found_tests["fast"].items()]
        medium_tests = [val for key, val in found_tests["medium"].items()]
        long_tests = [val for key, val in found_tests["long"].items()]
        conforming = long_tests + medium_tests + fast_tests
        dump(conforming, open(cfg.PREV_TESTS, "w"))

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
                    __test_dir__ = _get_test_module(
                        os.path.join(root, "__test_dir__.py"))
                    try:
                        base_nb_dir = __test_dir__.DIRECTORY
                    except AttributeError:
                        base_nb_dir = root.split(os.sep)[-1]
                    mathnbs[base_nb_dir] = [
                        os.path.join(dirnam, x) for x in files
                        if x.endswith(".nb") or x.endswith(".m")]
            continue
        continue
    t_find = time.time() - t_start

    if errors and not ignoreerror:
        pu.report_and_raise_error(
            "fix nonconforming benchmarks before continuing")

    pu.log_message("Found {0} Payette tests in {1:.2f}s."
                   .format(len(conforming), t_find),
                   noisy=True)

    if index_name:
        index = True
    if index:
        out = sys.stderr
        out.write("\nBENCHMARK INDEX\n\n")
        for key in found_tests:
            tests = found_tests[key]
            for py_mod, py_file in tests.items():
                # load module
                py_path = [os.path.dirname(py_file)]
                fobj, pathname, description = imp.find_module(py_mod, py_path)
                py_module = imp.load_module(py_mod, fobj, pathname, description)
                fobj.close()
                test = py_module.Test()

                # write out the information
                pre = WIDTH_TERM * "=" + "\n" if not index_name else ""
                out.write("{0}Name:  {1}\n".format(pre, test.name))

                if index_name:
                    # Only write out name
                    continue
                out.write("Owner: {0}\n\n".format(test.owner))
                out.write("Description:\n{0}".format(test.description))
                out.write("\nKeywords:\n")
                for key in test.keywords:
                    out.write("    {0}\n".format(key))
                continue
            continue

        return 0

    if not conforming:
        pu.report_and_raise_error("No tests found")

    # start the timer
    t_start = time.time()

    # Make a TestResults directory named "TestResults.{platform}"
    if not os.path.isdir(testresdir):
        os.mkdir(testresdir)

    summhtml = os.path.join(testresdir, "summary.html")
    respkl = os.path.join(testresdir, "Previous_Results.pkl")
    try:
        old_results = load(open(respkl, "r"))
    except IOError:
        old_results = {}

    # initialize the test results dictionary
    test_statuses = ["pass", "diff", "fail", "notrun",
                     "bad input", "failed to run"]
    test_res = {}
    for test_status in test_statuses:
        test_res[test_status] = {}
        continue

    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)
    pu.log_message("Running {0} benchmarks:".format(len(conforming)),
                   pre="", noisy=True)
    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)

    # run the tests on multiple processors using the multiprocessor map ONLY f
    # nprocs > 1. For debug purposes, when nprocs=1, run without using the
    # multiprocessor map because it makes debugging worse than it should be.
    topts = (switch, testresdir, forcererun, testfile, postprocess,)
    test_inp = ((test, topts, old_results) for test in conforming)
    if nproc == 1:
        all_results = [_run_test(job) for job in test_inp]

    else:
        pool = mp.Pool(processes=nproc)
        all_results = pool.map(_run_test, test_inp)
        pool.close()
        pool.join()

    t_run = time.time() - t_start
    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)

    # copy the mathematica notebooks to the output directory
    _copy_mathematica_nbs(mathnbs, testresdir)

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
                pu.log_warning(msg)
                continue
            tcompletion = summary["completion time"]
            benchdir = summary["benchmark directory"]
            keywords = summary["keywords"]
            old = summary.get("old status")
            test_res[status][name] = {"benchmark directory": benchdir,
                                      "completion time": tcompletion,
                                      "keywords": keywords,
                                      "old status": old}
            continue
        continue

    nnotrun = len(test_res["notrun"])
    nfail = len(test_res["fail"])
    npass = len(test_res["pass"])
    ndiff = len(test_res["diff"])
    nfailtorun = len(test_res["failed to run"])
    txtsummary = (
        "SUMMARY\n" +
        "{0} benchmarks took {1:.2f}s. total\n".format(len(conforming),
                                                       t_find + t_run) +
        "{0} benchmarks took {1:.2f}s. to run\n".format(len(conforming), t_run) +
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
         "{0} benchmarks took {1:.2f}s.\n".format(len(conforming), t_run) +
         "{0:^{1}}\n".format("{0:-^30}".format(" system information "),
                             WIDTH_TERM) +
         "   Date complete:    {0:<}\n".format(str_date) +
         "   Username:         {0:<}\n".format(getpass.getuser()) +
         "   Hostname:         {0:<}\n".format(os.uname()[1]) +
         "   Platform:         {0:<}\n".format(cfg.OSTYPE) +
         "   Python Version:   {0:<}\n".format(cfg.PYVER))

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
            continue
        longtxtlist = []
        for name in names:
            try:
                tcmpl = ("{0:.2f}s."
                         .format(test_res[stat][name]["completion time"]))
            except ValueError:
                tcmpl = str(test_res[stat][name]["completion time"])

            longtxtlist.append([tcmpl, name])

        longtxtlist = sorted(longtxtlist, key=lambda x: x[1])
        for pair in longtxtlist:
            longtxtsummary += "  {0:>8}   {1}\n".format(pair[0], pair[1])
            continue
        continue
    longtxtsummary += "=" * WIDTH_TERM + "\n"
    # longtxtsummary is finished at this point

    # This sends an email to everyone on the mailing list.
    if notify:
        pu.log_message("Sending results to mailing list.", pre="", noisy=True)
        pn.notify("Payette Benchmarks", longtxtsummary)

    pu.log_message(longtxtsummary, pre="", noisy=True)
    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)
    pu.log_message(txtsummary, pre="", noisy=True)
    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)

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

    for dirnam, dirs, files in os.walk(testresdir):
        for name in files:
            fbase, fext = os.path.splitext(name)
            delext = [".so", ".pyo", ".pyc"]
            if fext in delext:
                os.remove(os.path.join(dirnam, name))
                continue
            continue
        continue

    # dump results and write html summary
    dump(test_res, open(respkl, "w"))
    write_html_summary(summhtml, test_res)

    if ndiff and nfail:
        retval = -1
    elif ndiff:
        retval = 1
    elif nfail:
        retval = 2
    elif nfailtorun:
        retval = 3
    else:
        retval = 0

    return retval


def _run_test(args):

    """ run the payette test in py_file """

    # pass args to local variables
    py_file, opts, old_results = args
    py_dir = os.path.dirname(py_file)
    switch, testresdir, forcererun, testfile, postprocess = opts

    # check for switched materials
    if switch is not None:
        switch = [x.lower() for x in switch.split(":")]

    py_module = _get_test_module(py_file)

    # we instantiate the test here to copy files to destination directory
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
    __test_dir__ = _get_test_module(os.path.join(py_dir, "__test_dir__.py"))
    try:
        testbase = __test_dir__.DIRECTORY
    except AttributeError:
        testbase = os.path.split(os.path.dirname(py_file))[1]

    benchdir = os.path.join(testresdir, testbase, test.name)

    # check if benchmark has been run
    ran = [(y, x) for y in old_results for x in old_results[y] if x == test.name]

    if not (forcererun or testfile) and ran and os.path.isdir(benchdir):
        pu.log_message(
            "{0}".format(test.name) +
            " " * (50 - len(test.name)) +
            "{0:>10s}".format("notrun\n") +
            "Test already ran. " +
            "Use -F option to force a rerun", pre="", noisy=True)
        result = {test.name: old_results[ran[0][0]][ran[0][1]]}
        if result[test.name].get("old status") is None:
            result[test.name]["old status"] = ran[0][0]
        result[test.name]["status"] = "notrun"
        return result

    # Let the user know which test is running
    pu.log_message(
        "{0:<{1}}".format(test.name, WIDTH_TERM - WIDTH_INFO) +
        "{0:>{1}s}".format("RUNNING", WIDTH_INFO), pre="", noisy=True)

    # Create benchmark directory and copy the input and baseline files into the
    # new directory
    if os.path.isdir(benchdir):
        rmtree(benchdir)
    os.makedirs(benchdir)

    # copy input file, if any
    test_files = []
    if test.infile:
        infile = os.path.join(benchdir, os.path.basename(test.infile))
        copyfile(test.infile, infile)
        test_files.append(infile)

    # copy the python test file and make it executable
    test_py_file = os.path.join(benchdir, os.path.basename(py_file))
    copyfile(py_file, test_py_file)
    test_files.append(test_py_file)
    os.chmod(test_py_file, 0o750)

    # symlink the baseline file
    if test.baseline:
        for base_f in test.baseline:
            source = base_f
            link_name = os.path.join(benchdir, os.path.basename(base_f))
            if not os.path.isfile(source):
                pu.report_error(
                    "cannot symlink to non-existant file\n" +
                    "os.symlink({0},{1})".format(repr(source),repr(link_name)))
            if os.path.isfile(link_name):
                pu.report_error(
                    "cannot create symlink when link destination exists\n"+
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

    # check for switching
    if (test.material is not None and
        switch is not None and
        test.material.lower() == switch[0]):
            # switch the material
            _switch_materials(files=test_files, switch=switch)

    # write out hidden test file with some relevant info
    with open(os.path.join(benchdir, TEST_INFO), "w") as fobj:
        fobj.write("NAME={0}\n".format(test.name))
        fobj.write("PLATFORM={0}\n".format(cfg.OSTYPE))
        fobj.write("KEYWORDS={0}\n".format(", ".join(test.keywords)))
        fobj.write("FILE={0}\n".format(os.path.basename(py_file)))

    # delete the current test instance, and instantiate new from the files
    # just copied. Move to the new directory and run the test
    del py_module, test
    os.chdir(benchdir)
    result = _run_the_test(test_py_file, postprocess=postprocess)
    os.chdir(CWD)

    return result


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

                if stat == "notrun":
                    oldstat = results[stat][test].get("old status")
                    status = "notrun (old status: {0})".format(oldstat)
                else:
                    status = stat

                for myfile in files:
                    if myfile.endswith(".out.html") and postprocess:
                        fobj.write("<li><a href='{0}'>PostProcessing</a>\n"
                                   .format(os.path.join(tresd, myfile)))
                fobj.write("<li>Keywords: {0}\n".format(keywords))
                fobj.write("<li>Status: {0}\n".format(status))
                fobj.write("<li>Completion time: {0}\n".format(tcompletion))
                fobj.write("</ul>\n")
                continue
            fobj.write("</ul>\n")
            continue

    return


def _copy_mathematica_nbs(mathnbs, destdir):
    """copy the mathematica notebooks to the destination directory"""

    for mtldir, mathnb in mathnbs.items():
        for item in mathnb:
            fbase = os.path.basename(item)
            fold = os.path.join(destdir, mtldir, fbase)

            try:
                os.remove(fold)
            except OSError:
                pass

            if item.endswith(".m"):
                # don't copy the .m file, but write it, replacing rundir and
                # demodir with destdir
                with open(fold, "w") as fobj:
                    for line in open(item, "r").readlines():
                        demodir = os.path.join(destdir, mtldir) + os.sep
                        rundir = os.path.join(destdir, mtldir) + os.sep
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


def _switch_materials(files, switch):
    """switch materials"""
    #pat, repl = re.compile(switch[0], re.I|re.M), switch[1].lower()
    #for fname in files:
    #    lines = open(fname, "r").read()
    #    with open(fname, "w") as fobj:
    #        fobj.write(pat.sub(repl, lines))
    #    continue
    pat, repl = switch[0].lower(), switch[1].lower()
    for fname in files:
        lines = open(fname, "r").readlines()
        with open(fname, "w") as fobj:
            for line in lines:
                test_line = " ".join(line.lower().split())
                if "constitutive model" in test_line and pat in test_line:
                    line = "constitutive model {0}\n".format(repl)
                fobj.write(line)
                continue
        continue
    return


def _get_test_module(py_file):
    py_path = [os.path.dirname(py_file)]
    py_mod = pu.get_module_name(py_file)
    fobj, pathname, description = imp.find_module(py_mod, py_path)
    py_module = imp.load_module(py_mod, fobj, pathname, description)
    fobj.close()
    return py_module


def rebaseline_tests(args):
    if not args:
        args = list(set([os.path.splitext(os.path.basename(x))[0]
                         for x in os.listdir(os.getcwd())]))
        if len(args) > 1:
            print args
            sys.stdout.write("Could not determine which test to rebaseline\n")
            return

    for arg in args:
        fpath = os.path.realpath(os.path.expanduser(arg))
        fnam, fext = os.path.splitext(fpath)
        old = os.path.realpath(fnam + ".gold")
        new = fnam + ".out"
        errors = 0
        for f in (old, new):
            if not os.path.isfile(f):
                errors += 1
                sys.stderr.write("ERROR: {0} not found\n".format(f))
            continue
        if errors:
            sys.stdout.write("ERROR: Test not rebaselined\n")
            continue
        sys.stdout.write("Rebaseling {0}\n".format(os.path.basename(fnam)))
        shutil.move(new, old)
        sys.stdout.write("{0} rebaselined\n".format(os.path.basename(fnam)))
        continue
    sys.stdout.write("Rebaselining complete\n")
    return


def _run_the_test(the_test, postprocess=False):
    """Run the test in the_test

    """
    py_module = _get_test_module(the_test)
    test = py_module.Test()
    starttime = time.time()
    try:
        retcode = test.runTest()
    except PayetteError as error:
        retcode = 66
        pu.log_warning(error.message)

    retcode = ("bad input" if retcode == test.badincode else
               "pass" if retcode == test.passcode else
               "diff" if retcode == test.diffcode else
               "fail" if retcode == test.failcode else
               "failed to run" if retcode == test.failtoruncode else
               "unkown")

    # Print output at completion
    tcompletion = time.time() - starttime
    info_string = "{0} ({1:6.02f}s)".format(retcode.upper(), tcompletion)
    pu.log_message(
        "{0:<{1}}".format(test.name, WIDTH_TERM - WIDTH_INFO) +
        "{0:>{1}s}".format(info_string, WIDTH_INFO), pre="", noisy=True)

    # return to the directory we came from
    result = {test.name: {"status": retcode,
                          "keywords": test.keywords,
                          "completion time": tcompletion,
                          "benchmark directory": os.getcwd()}}

    if postprocess and os.path.isfile(test.outfile):
        pp.postprocess(test.outfile, verbosity=0)

    return result


if __name__ == "__main__":

    if not os.path.isfile(cfg.MTLDB):
        pu.report_and_raise_error(
            "buildPayette must be executed before tests can be run")

    ARGV = sys.argv[1:]
    if "--profile" in ARGV:
        PROFILE = True
        ARGV.remove("--profile")
        import cProfile
    else:
        PROFILE = False

    if PROFILE:
        CMD = "main(ARGV)"
        PROF = "payette.prof"
        cProfile.runctx(CMD, globals(), locals(), PROF)
        TEST = 0
    else:
        TEST = main(ARGV)

    sys.exit(TEST)
