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

"""Main Payette building file.
None of the functions in this file should be called directly, but only through
the executable script PAYETTE_ROOT/Toolset/buildPayette

AUTHORS
Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov

"""

from __future__ import print_function
import sys
import imp
import os
import optparse
import subprocess as sbp
import multiprocessing as mp
import pyclbr

import Payette_config as pc
import Source.Payette_utils as pu
from Source.Payette_utils import BuildError as BuildError

# --- module level constants
SPACE = "      "  # spacing used for logs to console
COMPILER_INFO = {}
MATERIALS = {}
VERBOSE = True
BUILD_ERRORS = 0

def build_payette(argv):

    """ create/build: material library files """

    global COMPILER_INFO, MATERIALS

    # *************************************************************************
    # -- command line option parsing
    usage = ("usage: buildPayette  [options]")
    parser = optparse.OptionParser(usage=usage, version="buildPayette 1.0")
    parser.add_option(
        "-x", "--no-build-libs",
        dest="nobuildlibs",
        action="store_true",
        default=False,
        help="do not build material libraries: [default: %default]")
    parser.add_option(
        "-m",
        dest="mtllib",
        action="append",
        default=[],
        help="specify material libraries to build: [default: ['all']]")
    parser.add_option(
        "-t", "--test",
        dest="TEST",
        action="store_true",
        default=False,
        help="run testPayette executable: [default: %default]")
    parser.add_option(
        "-o",
        dest="OPTIONS",
        action="append",
        default=[],
        help=("Options to build (accumulated) [default: %default] "
              "[choices: [electromech, special]]"))
    parser.add_option(
        "-F",
        dest="FORCEREBUILD",
        action="store_true",
        default=False,
        help="Rebuild Payette_materials.py [default:%default]")

    # the following options are shortcuts for building specific materials
    parser.add_option(
        "--dsc",
        dest="DSC",
        action="store_true",
        default=False,
        help="Build domain_switching_ceramic [default: %default]")
    parser.add_option(
        "--kmm",
        dest="KMM",
        action="store_true",
        default=False,
        help="Build kayenta [default: %default]")
    parser.add_option(
        "--lpc",
        dest="LPC",
        action="store_true",
        default=False,
        help="Build pieze_ceramic [default: %default]")
    parser.add_option(
        "--summary",
        dest="SUMMARY",
        action="store_true",
        default=False,
        help="write summary to screen [default: %default]")
    parser.add_option(
        "-j",
        dest="NPROC",
        action="store",
        type=int,
        default=1,
        help="number of simultaneous jobs [default: %default]")
    parser.add_option(
        "-a",
        dest="BUILDALL",
        action="store_true",
        default=False,
        help="build all materials (including non default) [default: %default")
    parser.add_option(
        "-d",
        dest="MTLDIRS",
        action="append",
        default=[],
        help=("Additional directories to scan for materials, accumulated "
              "[default: %default]."))
    (opts, args) = parser.parse_args(argv)

    if len(args) > 0:
        parser.print_help()
        parser.error("buildPayette does not require arguments, only options")

    if opts.SUMMARY:
        write_summary_to_screen()
        sys.exit(0)

    pu.logmes(pc.PC_INTRO)

    # determine if we build all materials, or just a selection
    if opts.DSC:
        opts.mtllib.append("domain_switching_ceramic")

    if opts.KMM:
        opts.mtllib.append("kayenta")

    if opts.LPC:
        opts.mtllib.append("piezo_ceramic")

    if opts.FORCEREBUILD:
        try:
            os.remove(pc.PC_MTLS_FILE)
        except OSError:
            pass

    # clean up options:
    options = []
    for option in opts.OPTIONS:
        if "elec" in option:
            options.append("electromechanical")
        elif "special" in options:
            options.append("special")
        else:
            options.append(option)
        continue

    if opts.BUILDALL:
        options.append("buildall")

    # intro message
    pu.loginf("Building Payette\n")

    # prepare compiler options
    if pc.PC_FCOMPILER:
        f2pyopts = ["--fcompiler={0}".format(pc.PC_FCOMPILER)]
    else:
        f2pyopts = ["--f77exec={0}".format(pc.PC_F77EXEC),
                    "--f90exec={0}".format(pc.PC_F90EXEC)]
    if pc.PC_F2PYDBG:
        f2pyopts.append("--debug")

    # compiler options to send to the build scripts
    COMPILER_INFO = {"f2py": {"compiler": pc.PC_F2PY,
                              "options": f2pyopts}}

    # check material directories
    errors = 0
    mtl_dirs = pc.PC_MTLS
    for dirnam in opts.MTLDIRS:
        dirnam = os.path.expanduser(dirnam)
        if not os.path.isdir(dirnam):
            errors += 1
            pu.logerr("material directory {0} not found".format(dirnam))
        else:
            mtl_dirs.append(dirnam)
        continue

    if errors:
        sys.exit("ERROR: stopping due to previous errors")

    if not opts.nobuildlibs:

        # get names of materials from Source/Materials
        errors = 0
        pu.loginf("finding Payette materials from\n{0}"
                  .format("\n".join([SPACE + x for x in mtl_dirs])))
        MATERIALS = get_payette_mtls(mtl_dirs, opts.mtllib, options)
        pu.loginf("Payette materials found\n")
        non_existent = MATERIALS.get("non existent", False)
        if non_existent:
            errors += 1
            del MATERIALS["non existent"]

        # build the requested material libraries
        nproc = min(mp.cpu_count(), opts.NPROC)
        errors += build_payette_mtls(nproc)
        # material libraries built, now write the
        # Source/Materials/Payette_installed_materials.py file containing all
        # materials
        write_payette_materials(MATERIALS)

    else:
        errors = 0

    # if the user wanted only to build certain libraries, return when built
    if opts.mtllib:
        return errors

    # check that runPayette works
    test_error = test_run_payette(opts.TEST)

    if errors and test_error == 0:
        # test_run_payette completed without error, but there were previous
        # build errors, meaning that some libraries did not build, but
        # Payette still built.
        errors = 55

    elif not errors and test_error != 0:
        # test_run_payette completed with error, but everything built fine,
        # something is wrong...
        errors = 75

    errors += test_error

    return errors


def test_run_payette(test):

    """ test that runPayette executes properly for [-h] """

    pu.begmes("INFO: testing that runPayette [-h] executes normally", pre="")
    cmd = [pc.PC_RUNPAYETTE, "-h"]
    runcheck = sbp.Popen(cmd, stdout=sbp.PIPE, stderr=sbp.STDOUT)
    runcheck.wait()
    if runcheck.returncode != 0:
        sbp_msg = runcheck.communicate()[0]
        if type(sbp_msg) == bytes:
            sbp_msg = sbp_msg.decode("ascii")
        msg = [x for x in sbp_msg.split("\n") if x]
        message = ("the following error was trapped from runPayette [-h]:\n"
                   "{0}".format("=" * 25 +
                                " Start Error\n" +
                                sbp_msg + "\n" + "=" * 25 +
                                " End Error\n"))
        build_fail(message)

        pu.logmes("<<< IF >>> no other build errors were encountered, "
                  "please let the Payette developers know so a fix "
                  "can be found")
        return 1
    else:
        pu.endmes("runPayette [-h] executed normally\n")

    if not test:
        return 0

    pu.begmes("INFO: testing that testPayette [-k elastic -K kayenta] "
              "executes normally")
    cmd = [os.path.join(pc.PC_TOOLS, "testPayette"), "-k", "elastic",
           "-K", "kayenta", "-F", "buildPayette"]
    runcheck = sbp.Popen(cmd, stdout=sbp.PIPE, stderr=sbp.STDOUT)
    runcheck.wait()
    if runcheck.returncode != 0:
        sbp_msg = runcheck.communicate()[0]
        if type(sbp_msg) == bytes:
            sbp_msg = sbp_msg.decode("ascii")
        msg = [x for x in sbp_msg.split("\n") if x]
        message = ("the following error was trapped from "
                   "testPayette [-k elastic -K kayenta]:\n{0}".format(msg[-1]))
        build_fail(message)
        message = ("please let the Payette developers know so a "
                   "fix can be found")
        pu.logmes(message)
        return 1
    else:
        pu.endmes("testPayette [-k elastic -K kayenta] executed normally\n")

    return 0


def write_payette_materials(payette_materials):

    """ Write the Source/Materials/Payette_materials.py file containing a
    dictionary of installed models and model attributes

    """

    pu.loginf("writing {0}".format("PAYETTE_ROOT" +
                                   pc.PC_MTLS_FILE.split(pc.PC_ROOT)[1]))

    lines = []
    if os.path.isfile(pc.PC_MTLS_FILE):
        lines.extend(open(pc.PC_MTLS_FILE, "r").readlines())

    # get list of previously installed materials
    installed_materials = []
    for line in lines:
        if "PAYETTE_INSTALLED_MATERIALS" in line:
            installed_materials = eval(line.strip().split("=")[1])
            break
        continue

    # remove failed materials from installed materials
    for material in [x for x in payette_materials
                     if payette_materials[x]["build requested"]
                     and payette_materials[x]["build failed"]]:
        try:
            installed_materials.remove(material)
        except ValueError:
            pass
        continue

    # add built materials to installed materials
    for material in [x for x in payette_materials
                     if payette_materials[x]["build requested"]
                     and payette_materials[x]["build succeeded"]]:
        if material not in installed_materials:
            installed_materials.append(material)

        continue

    # check that all installed materials are in all_materials
    all_materials = [payette_materials[x]["name"] for x in payette_materials]
    for material in installed_materials:
        if material not in all_materials:
            pu.logwrn("installed material {0} not in payette_materials"
                      .format(material))

        continue
    installed_materials = [x for x in installed_materials
                           if x in all_materials]

    # write the PC_MTLS_FILE file
    with open(pc.PC_MTLS_FILE, "w") as ftmp:
        intro = ("""
# *********************************************************************** #
#                                                                         #
# This file was generated automatically by the Payette. It contains       #
# important directives for materials detected and built by Payette.       #
#                                                                         #
# This file is intended to be imported by Payette so that material        #
# classes# can be instantiated from built materials.                      #
#                                                                         #
# DO NOT EDIT THIS FILE. This entire file is regenerated automatically    #
# each time Payette is built. Any changes you make to this file will be   #
# overwritten.                                                            #
#                                                                         #
# If changes are needed, please contact the Payette developers so that    #
# changes can be made to the build scripts.                               #
# *********************************************************************** #
""")
        ftmp.write(intro)
        pu.begmes("writing successfully installed materials", pre=SPACE)
        ftmp.write("PAYETTE_INSTALLED_MATERIALS={0}\n"
                   .format(installed_materials))
        pu.endmes("successfully installed materials written")

        ftmp.write("PAYETTE_CONSTITUTIVE_MODELS={}\n")

        pu.begmes("writing constitutive model declarations", pre=SPACE)
        for material in installed_materials:
            payette_material = payette_materials[material]
            py_module = payette_material["module"]
            name = payette_material["name"]
            aliases = payette_material["aliases"]
            class_name = payette_material["class name"]
            ftmp.write("from {0} import {1}\n".format(py_module, class_name))
            ftmp.write('PAYETTE_CONSTITUTIVE_MODELS["{0}"]='
                       '{{"class name":{1},'
                       '"aliases":{2}}}\n'.format(name, class_name, aliases))
            continue

        pu.endmes("constitutive model declarations written")

    pu.loginf("{0} written\n".format("PAYETTE_ROOT" +
                                     pc.PC_MTLS_FILE.split(pc.PC_ROOT)[1]))
    return


def build_payette_mtls(nproc=1):

    """ build the library files for each material.  most materials are
    are implemented as fortran files and need to be compiled with
    f2py.

    """

    global VERBOSE, MATERIALS

    pu.loginf("building Payette material libraries")

    # now build the materials
    requested_builds = [x for x in MATERIALS
                        if MATERIALS[x]["build requested"]]
    if not requested_builds:
        pu.logmes("no material libraries to build", pre=SPACE)

    # build the libraries
    nproc = min(nproc, len(requested_builds))
    if nproc > 1:
        VERBOSE = False
        pool = mp.Pool(processes=nproc)
        build_results = pool.map(_build_lib, requested_builds)
        pool.close()
        pool.join()

    else:
        build_results = [_build_lib(material) for material in requested_builds]

    # reconstruct materials from build_results
    for item in build_results:
        MATERIALS[item[0]] = item[1]

    pu.loginf("Payette material libraries built\n")

    failed_materials = [MATERIALS[x]["libname"]
                        for x in MATERIALS
                        if MATERIALS[x]["build requested"]
                        and MATERIALS[x]["build failed"]]
    built_materials = [MATERIALS[x]["libname"]
                       for x in MATERIALS
                       if MATERIALS[x]["build requested"]
                       and MATERIALS[x]["build succeeded"]]

    if failed_materials:
        errors = 55
        pu.logwrn("The following materials WERE NOT built:\n{0}\n"
                  .format("\n".join([" " * 10 + x for x in failed_materials])))

    if built_materials:
        pu.loginf("The following materials WERE built:\n{0}\n"
                  .format("\n".join([" " * 10 + x for x in built_materials])))

    # remove cruft
    for ftmp in [x for x in os.listdir(pc.PC_TOOLS)
              if x.split(".")[-1] in ["so", "o"]]:
        os.remove(ftmp)
        continue

    return BUILD_ERRORS


def _build_lib(material):

    """ build the material library for payette_material """

    global MATERIALS, BUILD_ERRORS

    # get attributes
    name = MATERIALS[material]["name"]
    libname = MATERIALS[material]["libname"]
    build_script = MATERIALS[material]["build script"]
    parse_err = MATERIALS[material]["parse error"]
    if VERBOSE:
        pu.begmes("building {0}".format(libname), pre=SPACE)

    if parse_err:
        MATERIALS[material]["build failed"] = True
        if VERBOSE:
            pu.endmes("{0} skipped due to previous errors".format(libname))

    elif build_script is None:
        MATERIALS[material]["build succeeded"] = True
        if VERBOSE:
            pu.endmes("{0} built ".format(libname))

    else:
        # import build script
        py_mod, py_path = pu.get_module_name_and_path(build_script)
        fobj, pathname, description = imp.find_module(py_mod, py_path)
        build = imp.load_module(py_mod, fobj, pathname, description)
        fobj.close()

        try:
            build = build.Build(name, libname, COMPILER_INFO)
            build_error = build.build_extension_module()

        except BuildError as error:
            build_error = error.errno

        if build_error:
            BUILD_ERRORS += 1
            MATERIALS[material]["build failed"] = True
            if build_error == 5 or build_error == 10 or build_error == 40:
                pass
            elif build_error == 66:
                pu.logwrn("{0}: missing attribute: build".format(build_script))
            else:
                msg = ("failed to build {0} extension module. see {1}"
                       .format(libname, "build.echo"))
                pu.logwrn(msg, pre="\t\t")

        else:
            MATERIALS[material]["build succeeded"] = True
            if VERBOSE:
                pu.endmes("{0} built ".format(libname))

        # remove bite compiled files
        try:
            os.remove(build_script + "c")
        except OSError:
            pass

    return [material,MATERIALS[material]] # end of _build_lib


def get_payette_mtls(mtl_dirs, requested_libs=None, options=None):

    """Read python files in Source/Materials and determine which are interface
    files for material models. If they are, add them to the payette_materials
    dictionary, along with their attributes

    """

    if requested_libs is None:
        requested_libs = []

    if options is None:
        options = []

    # determine if we want to build only select libraries
    build_select = bool(requested_libs)
    buildall = "buildall" in options

    def get_super_classes(data):

        """ return the super class name from data """

        super_class_names = []
        for super_class in data.super:
            if super_class == "object":
                continue
            if isinstance(super_class, basestring):
                super_class_names.append(super_class)
            else:
                super_class_names.append(super_class.name)

            continue
        return super_class_names

    payette_materials = {}
    py_files = []
    for dirnam in mtl_dirs:
        py_files.extend([os.path.join(dirnam, x)
                         for x in os.listdir(dirnam) if x.endswith(".py")])

    # go through list of python files in
    for py_file in py_files:

        parse_err = False

        py_mod, py_path = pu.get_module_name_and_path(py_file)
        fobj, pathname, description = imp.find_module(py_mod, py_path)
        py_module = imp.load_module(py_mod, fobj, pathname, description)
        fobj.close()

        attributes = getattr(py_module, "attributes", None)
        if attributes is None or not isinstance(attributes, dict):
            continue

        # check if this is a payette material
        payette_material = attributes.get("payette material")
        if payette_material is None or not payette_material:
            continue

        # check if a constitutive model class is defined
        class_data = pyclbr.readmodule(py_mod, path=py_path)

        parent = class_data.get("Parent")
        if parent is not None:
            proto = parent.name
        else:
            proto = "ConstitutiveModelPrototype"

        for name, data in class_data.items():
            class_name = data.name
            constitutive_model = proto in get_super_classes(data)
            if constitutive_model:
                break
            continue

        if not constitutive_model:
            del py_module
            continue

        # file is an interface file check attributes, define defaults
        name = attributes.get("name")
        if name is None:
            pu.logerr("No name attribute given in {0}, skipping"
                      .format(py_file))
            continue

        name = name.replace(" ", "_").lower()
        libname = attributes.get("libname", name + pc.PC_EXT_MOD_FEXT)

        # material type
        material_type = attributes.get("material type")
        if material_type is None:
            pu.logerr("No material type attribute given in {0}, skipping"
                      .format(py_file))
            continue

        electromtl = bool([x for x in material_type if "electro" in x])
        specialmtl = bool([x for x in material_type if "special" in x])

        if electromtl and "electromechanical" in options:
            requested_libs.append(name)

        if specialmtl and "special" in options:
            requested_libs.append(name)

        # default material?
        default = attributes.get("default material", False)
        if default and not build_select:
            requested_libs.append(name)

        # get aliases, they need to be a list of aliases
        aliases = attributes.get("aliases", [])
        if not isinstance(aliases, (list, tuple)):
            aliases = [aliases]
        aliases = [x.replace(" ", "_").lower() for x in aliases]

        # fortran model set up
        fortran_source = attributes.get("fortran source", False)
        build_script = attributes.get("build script")
        depends = attributes.get("depends")

        # all fortran models must give a build script
        if fortran_source and not build_script:
            parse_err = True
            pu.logerr("No build script given for fortran source in {0} for {1}"
                      .format(py_file, libname), pre=SPACE)

        # unless it is not needed...
        elif build_script == "Not_Needed":
            build_script = None

        # and the build script must exist.
        elif build_script is not None:
            if not os.path.isfile(build_script):
                parse_err = True
                pu.logerr("build script {0} not found".format(build_script))

        # collect all parts
        mtl_dict = {
            "name": name,
            "libname": libname,
            "fortran source": fortran_source,
            "build script": build_script,
            "aliases": aliases,
            "material type": material_type,
            "module": py_mod,
            "file": py_file,
            "class name": class_name,
            "depends": depends,
            "parse error": parse_err,
            "build requested": False,  # by default, do not build the material
            "build succeeded": False,
            "build failed": False,
            }

        payette_materials[name] = mtl_dict
        # payette_materials[py_mod] = mtl_dict
        del py_module

        if buildall and name not in requested_libs:
            requested_libs.append(name)

        # decide if it should be built or not
        if name not in requested_libs:
            continue

        # by this point, we have filtered out the materials we do not want to
        # build, so request that it be built
        payette_materials[name]["build requested"] = True

        continue

    dependent_materials = [x for x in payette_materials
                           if payette_materials[x]["depends"]]
    if dependent_materials:
        for material in dependent_materials:
            depends_on = payette_materials[material]["depends"]
            # user has requested to build a material that depends on another.
            # make sure that the other material exists
            if depends_on not in payette_materials:
                raise BuildError("{0} depends on {1} which was not found"
                                 .format(material, depends_on), 25)

            # if material was requested to be built, make sure the material it
            # depends on is also built
            if payette_materials[material]["build requested"]:
                payette_materials[depends_on]["build requested"] = True

            continue

    # the user may have requested to build a material that does not exist, let
    # them know
    all_names = [payette_materials[x]["name"] for x in payette_materials]
    non_existent = []
    for name in requested_libs:
        if name not in all_names:
            non_existent.append(name)
            payette_materials["non existent"] = True
        continue

    if non_existent:
        pu.logwrn("requested material[s] {0} not found"
                  .format(", ".join(non_existent)), pre=SPACE)

    return payette_materials


def build_fail(msg):

    """ warn that the build failed """

    msg = msg.split("\n")
    err = "BUILD FAILED"
    sss = r"*" * int((80 - len(err)) / 2)
    pu.logmes("\n\n{0} {1} {2}\n".format(sss, err, sss))
    for line in msg:
        pu.logmes("BUILD FAIL: {0}".format(line))
    pu.logmes("\n\n")
    return


def write_summary_to_screen():

    """ write summary of entire Payette project to the screen """

    def num_code_lines(fpath):

        """ return the number of lines of code in fpath """

        nlines = 0
        if os.path.splitext(fpath)[1] not in code_exts:
            return nlines
        for line in open(fpath, "r").readlines():
            line = line.strip().split()
            if not line or line[0] == "#":
                continue
            nlines += 1
            continue
        return nlines

    all_dirs, all_files = [], []
    code_exts = [".py", ".pyf", "", ".F", ".C", ".f"]
    all_exts = code_exts + [".inp", ".tex", ".pdf"]
    for dirnam, dirs, files in os.walk(pc.PC_ROOT):
        if ".git" in dirnam:
            continue
        all_dirs.extend([os.path.join(dirnam, d) for d in dirs])
        all_files.extend([os.path.join(dirnam, ftmp) for ftmp in files
                          if not os.path.islink(os.path.join(dirnam, ftmp))
                          and os.path.splitext(ftmp)[1] in all_exts])
        continue
    num_lines = sum([num_code_lines(ftmp) for ftmp in all_files])
    num_dirs = len(all_dirs)
    num_files = len(all_files)
    num_infiles = len([x for x in all_files if x.endswith(".inp")])
    num_pyfiles = len([x for x in all_files
                       if x.endswith(".py") or x.endswith(".pyf")])
    pu.logmes(pc.PC_INTRO)
    pu.logmes("Summary of Project:")
    pu.logmes("\tNumber of files in project:         {0:d}"
              .format(num_files))
    pu.logmes("\tNumber of directories in project:   {0:d}"
              .format(num_dirs))
    pu.logmes("\tNumber of input files in project:   {0:d}"
              .format(num_infiles))
    pu.logmes("\tNumber of python files in project:  {0:d}"
              .format(num_pyfiles))
    pu.logmes("\tNumber of lines of code in project: {0:d}"
              .format(num_lines))
    return


if __name__ == "__main__":

    BUILD = build_payette(sys.argv[1:])

    ERROR, WARN = 0, 0
    if BUILD == 0:
        pu.loginf("buildPayette succeeded\n")

    elif BUILD == 55:
        WARN += 1
        pu.logwrn("buildPayette failed to build one or "
                  "more material libraries\n")

    elif BUILD == 75:
        ERROR += 1
        pu.logerr("buildPayette failed due to an unknown error\n")

    else:
        pu.logerr("buildPayette failed\n")
        ERROR += 1

    if not ERROR and not WARN:
        pu.logmes("Enjoy Payette!")

    elif WARN:
        pu.logmes("You've been warned, tread lightly!")

    else:
        pu.logmes("Better luck next time!")

    sys.exit(BUILD)

