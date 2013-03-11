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
import time
import shutil
import multiprocessing as mp
from pickle import dump, load
import datetime
import getpass
import re
from inspect import getfile
import optparse

import Source.__config__ as cfg
import Source.Payette_utils as pu
import Source.Payette_notify as pn
from Source.Payette_test import (find_tests, WIDTH_TERM, WIDTH_INFO,
                                 TEST_INFO, RESDIR, CWD, get_test)
from Source.Payette_utils import PayetteError as PayetteError
from Source.Payette_utils import PassThroughOptionParser


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
        help="Run only 'builtin' tests [default: %default]")
    parser.add_option(
        "-c",
        action="store_true",
        default=False,
        help="Run tests found in current directory [default: %default]")
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
              "[default: %default]"))
    parser.add_option(
        "-D",
        dest="TESTBASEDIR",
        action="store",
        default=RESDIR,
        help="Directory to run tests [default: %default]")
    parser.add_option(
        "-F",
        dest="FORCERERUN",
        action="store_true",
        default=False,
        help="Force benchmarks to be run again [default: %default]")
    parser.add_option(
        "-j", "--nproc",
        dest="NPROC",
        default="1",
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
        help="email test results to mailing list [default: %default]")
    parser.add_option(
        "-I",
        dest="IGNOREERROR",
        action="store_true",
        default=False,
        help="Ignore noncomforming tests [default: %default]")
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
    parser.add_option(
        "-w",
        dest="WIPE",
        action="store_true",
        default=False,
        help="Wipe results directory first [default: %default]")

    # indexing
    parser.add_option(
        "-i", "--index",
        dest="INDEX",
        action="store_true",
        default=False,
        help="Print benchmarks index [default: %default]")
    parser.add_option(
        "--name",
        dest="INDEX_NAME",
        action="store_true",
        default=False,
        help="Print benchmark names, -i is implied [default: %default]")
    if print_help:
        parser.print_help()
        return
    (opts, args) = parser.parse_args(argv)
    nproc = int(opts.NPROC) if opts.NPROC.upper() != "M" else mp.cpu_count()
    sys.exit(test_payette(
        args, testbase=opts.TESTBASEDIR, builtin=opts.BUILTIN,
        keywords=opts.KEYWORDS, nokeywords=opts.NOKEYWORDS,
        spectests=opts.SPECTESTS, benchdirs=opts.BENCHDIRS,
        forcererun=opts.FORCERERUN, nproc=nproc, postprocess=opts.POSTPROCESS,
        notify=opts.NOTIFY, ignoreerror=opts.IGNOREERROR, switch=opts.SWITCH,
        rebaseline=opts.REBASELINE, index=opts.INDEX,
        index_name=opts.INDEX_NAME, wipe=opts.WIPE, runcwd=opts.c))


def test_payette(args, testbase=None, builtin=False, keywords=[], nokeywords=[],
                 spectests=[], benchdirs=[], forcererun=False, nproc=1,
                 postprocess=False, notify=False, ignoreerror=False,
                 switch=None, rebaseline=False, index=False,
                 index_name=False, wipe=False, runcwd=False):
    """Run the Payette benchmarks.

    Walk through and run the Payette test simulations, compare results
    against the accepted results.

    Parameters
    ----------
    testbase : str
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
    switch : str {None}
        switch materials
    rebaseline : bool {False}
        rebaseline test
    index : bool {False}
    index_name : bool {False}
        write index [_name] and exit
    wipe : bool {False}
        Wipe test directory first

    """
    if switch is not None:
        pu.log_warning("switching materials is an untested feature")

    # number of processors
    nproc = min(mp.cpu_count(), nproc)

    pu.log_message(cfg.INTRO, pre="", noisy=True)

    if rebaseline:
        sys.exit(rebaseline_tests(args))

    # find tests
    pu.log_message("Finding Payette tests", noisy=True)

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
            pu.report_and_raise_error(
                "FILE not found in {0}".format(TEST_INFO))
        args.append(pyfile)

    # if the user passed in python test files, just run those tests
    if runcwd:
        args.extend([x for x in os.listdir(CWD) if x.endswith(".py")])

    t_start = time.time()
    tests = []
    no_write = False
    pyargs = [x for x in args if x.endswith(".py")]
    if pyargs:
        dd = []
        for f in pyargs:
            if not os.path.isfile(f):
                pu.file_not_found(f, count=False)
                continue
            test = get_test(f)
            if test is None:
                continue
            test.set_results_directory(os.getcwd())
            d = os.path.dirname(test.test_file)
            if d != os.getcwd():
                pu.report_error("Run testPayette from {0}"
                                .format(os.path.dirname(test.test_file)))
                continue
            if dd and d not in dd:
                pu.report_error("Can only run files from same dir")
            dd.append(d)
            tests.append(test)
            continue
        if not tests:
            sys.exit("No tests found")
        testbase = os.getcwd()
        wipe = False
        no_write = True

    if not tests:
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
        if pu.errors():
            pu.report_and_raise_error("stopping due to previous errors")

        if builtin:
            keywords.append("builtin")

        pu.log_message(
            "Gathering Payette tests from\n{0}".format(
                "\n".join([" " * 6 + x for x in test_dirs])), noisy=True)
        tests = find_tests(keywords, nokeywords, spectests, test_dirs)

    if ignoreerror:
        pu.reset_error_and_warnings()

    if pu.errors():
        pu.report_and_raise_error(
            "fix nonconforming benchmarks before continuing")

    t_find = time.time() - t_start

    pu.log_message("Found {0} Payette tests in {1:.2f}s."
                   .format(len(tests), t_find), noisy=True)

    if index_name:
        index = True

    if index:
        out = sys.stderr
        out.write("\nBENCHMARK INDEX\n\n")
        for test in tests:
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
        return

    if not tests:
        pu.report_and_raise_error("No tests found")

    # start the timer
    t_start = time.time()

    # wipe directory if requested
    if wipe and testbase != os.getcwd():
        try:
            shutil.rmtree(RESDIR)
        except OSError:
            pass
    if not os.path.isdir(testbase):
        os.makedirs(testbase)


    summhtml = os.path.join(testbase, "summary.html")
    respkl = os.path.join(testbase, "Previous_Results.pkl")
    try:
        prev_tests = load(open(respkl, "r"))
    except IOError:
        prev_tests = []

    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)
    pu.log_message("Running {0} benchmarks:".format(len(tests)),
                   pre="", noisy=True)
    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)

    # run the tests on multiple processors using the multiprocessor map ONLY f
    # nprocs > 1. For debug purposes, when nprocs=1, run without using the
    # multiprocessor map because it makes debugging worse than it should be.
    topts = (testbase, postprocess, switch,)
    test_inp = ((test, topts, prev_tests, forcererun) for test in tests)
    if nproc == 1:
        all_results = [_run_test(job) for job in test_inp]

    else:
        pool = mp.Pool(processes=nproc)
        all_results = pool.map(_run_test, test_inp)
        pool.close()
        pool.join()

    t_run = time.time() - t_start
    pu.log_message("=" * WIDTH_TERM, pre="", noisy=True)

    if no_write:
        return

    statuses = {}
    for result in all_results:
        try:
            statuses[result.status] += 1
        except KeyError:
            statuses[result.status] = 1
    txtsummary = """\
SUMMARY
{0} benchmarks took {1:.2f}s. total
{0} benchmarks took {2:.2f}s. to run
""".format(len(tests), t_find+t_run, t_run)
    for status, n in statuses.items():
        txtsummary += "\n{0} benchmarks showed '{1}'".format(n, status)
    txtsummary += """
For a summary of which benchmarks passed, diffed, failed, or not run, see
{0}""".format(summhtml)

    # Make a long summary including the names of what passed and
    # what didn't as well as system information.
    str_date = datetime.datetime.today().strftime("%A, %d. %B %Y %I:%M%p")

    longtxtsummary = (
        "=" * WIDTH_TERM + "\nLONG SUMMARY\n" +
        "{0} benchmarks took {1:.2f}s.\n".format(len(tests), t_run) +
        "{0:^{1}}\n".format("{0:-^30}".format(" system information "),
                            WIDTH_TERM) +
        "   Date complete:    {0:<}\n".format(str_date) +
        "   Username:         {0:<}\n".format(getpass.getuser()) +
        "   Hostname:         {0:<}\n".format(os.uname()[1]) +
        "   Platform:         {0:<}\n".format(cfg.OSTYPE) +
        "   Python Version:   {0:<}\n".format(cfg.PYVER))

    for status, n in statuses.items():
        _tests = [x for x in all_results if x.status == status]
        header = "{0:-^30}".format(" " +
                                   status +
                                   " ({0}) ".format(n))
        longtxtsummary += "{0:^{1}}\n".format(header, WIDTH_TERM)
        if len(_tests) == 0:
            continue
        longtxtlist = []
        for test in _tests:
            tcmpl = test.completion_time(f="{0:.2f}s")
            longtxtlist.append([tcmpl, test.name])
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

    for dirnam, dirs, files in os.walk(testbase):
        for name in files:
            fbase, fext = os.path.splitext(name)
            delext = [".so", ".pyo", ".pyc"]
            if fext in delext:
                os.remove(os.path.join(dirnam, name))
                continue
            continue
        continue

    # dump results and write html summary
    dump_results(respkl, all_results)
    write_html_summary(summhtml, all_results)

    return


def _run_test(args):
    """ run the payette test """

    # pass args to local variables
    test, setup_args, prev_tests, forcererun = args

    # check if benchmark has been run
    ran = [x for x in prev_tests if x.name == test.name]
    if not forcererun and ran:
        prev_test = ran[0]
        pu.log_message("""\
{0}{1}{2:>10s}
Test already ran.  Use -F option to force a rerun
""".format( test.name, " " * (50 - len(test.name)), test.status),
                       pre="", noisy=True)
        if test.old_status is None:
            test.old_status = prev_test.status
        return test

    test.setup_test(*setup_args)

    try:
        test.run_test()
    except PayetteError as error:
        test.retcode = test.failtoruncode
        test.status = test.get_status()
        pu.log_warning(error.message)
    test.finish_test()
    return test


def dump_results(fpath, all_results):
    dump(all_results, open(fpath, "w"))
    return
    res = {}
    for x in all_results:
        d = {}
        for k, v in x.__dict__.items():
            try:
                if v.__self__ is not None:
                    continue
            except AttributeError:
                pass
            d[k] = v
        res[x.name] = d
    dump(res, open(fpath, "w"))
    return


def write_html_summary(fname, results):
    """ write summary of the results dictionary to html file """
    resd = os.path.dirname(fname)

    # get different status types
    statuses = {}
    for result in results:
        try:
            statuses[result.status].append(result)
        except KeyError:
            statuses[result.status] = [result]
    statstr = " ".join("{0:3d} {1}".format(len(v), k) for k, v in statuses.items())
    rdir = os.path.dirname(fname)
    with open(fname, "w") as fobj:
        # write header
        fobj.write("<html>\n<head>\n<title>Test Results</title>\n</head>\n")
        fobj.write("<body>\n<h1>Summary</h1>\n")
        fobj.write("<ul>\n")
        fobj.write("<li> Directory: {0} </li>\n".format(resd))
        fobj.write("<li> Options: {0} </li>\n".format(" ".join(sys.argv[1:])))
        fobj.write("<li> {0} </li>\n".format(statstr))
        fobj.write("</ul>\n")

        # write out details test that fail, diff, pass, notrun
        for status, _tests in statuses.items():
            fobj.write("<h1>Tests that showed '{0}'</h1>\n<ul>\n".format(status))
            for test in _tests:
                tresd = test.results_directory()
                fobj.write("<li>{0}</li>\n".format(test.name))
                fobj.write("<ul>\n")
                fobj.write("<li>Files: \n")
                files = os.listdir(tresd)
                for fnam in files:
                    fpath = os.path.join(tresd, fnam).replace(rdir, ".")
                    fobj.write("<a href='{0}' type='text/plain'>{1}</a> \n"
                               .format(fpath, fnam))
                    continue
                keywords = "  ".join(test.keywords)
                html_link = test.postprocess
                tcompletion = test.completion_time(f="{0:.2f}s")

                if html_link:
                    if not os.path.isfile(html_link):
                        html_link = os.path.join(tresd, html_link)
                    if os.path.isfile(html_link):
                        fobj.write("<li><a href='{0}'>PostProcessing</a>\n"
                                   .format(os.path.join(html_link)))
                fobj.write("<li>Keywords: {0}\n".format(keywords))

                if status == "NOT RUN":
                    status = "NOT RUN (old status: {0})".format(test.old_status)
                fobj.write("<li>Status: {0}\n".format(status))
                fobj.write("<li>Completion time: {0}\n".format(tcompletion))
                fobj.write("</ul>\n")
                continue
            fobj.write("</ul>\n")
            continue

    return


def rebaseline_tests(args):
    if not args:
        args = list(set([os.path.splitext(os.path.basename(x))[0]
                         for x in os.listdir(os.getcwd())]))
        try:
            args.remove(".test")
        except:
            pass
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
                pu.file_not_found(f)
            continue
        if pu.errors():
            pu.report_error("ERROR: Test not rebaselined\n")
            pu.reset_error_and_warnings()
            continue
        sys.stdout.write("Rebaselining {0}\n".format(os.path.basename(fnam)))
        open(old, "w").write(open(new, "r").read())
        sys.stdout.write("{0} rebaselined\n".format(os.path.basename(fnam)))
        continue
    sys.stdout.write("Rebaselining complete\n")
    return


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
