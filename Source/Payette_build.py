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
import pickle

import Payette_config as pc
import Source.Payette_utils as pu
from Source.Payette_utils import BuildError as BuildError

# --- module level constants
SPACE = "      "  # spacing used for logs to console
COMPILER_INFO = {}
MATERIALS = {}
VERBOSE = True
BUILD_ERRORS = 0

# python 3 compatibility
try:
    unicode
except NameError:
    basestring = unicode = str


def build_payette(argv):

    """ create/build: material library files """

    global COMPILER_INFO, MATERIALS, VERBOSE

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
    parser.add_option(
        "-v",
        dest="VERBOSITY",
        action="store",
        default=1,
        type=int,
        help="Verbosity [default: %default]")

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

    if not opts.VERBOSITY:
        VERBOSE = False

    if VERBOSE:
        pu.log_message(pc.PC_INTRO, pre="")

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
    pu.log_message("Building Payette\n")

    # prepare compiler options
    if pc.PC_FCOMPILER:
        f2pyopts = ["--fcompiler={0}".format(pc.PC_FCOMPILER)]
    else:
        f2pyopts = ["--f77exec={0}".format(pc.PC_F77EXEC),
                    "--f90exec={0}".format(pc.PC_F90EXEC)]
    if pc.PC_F2PYDBG:
        f2pyopts.append("--debug")

    # compiler options to send to the fortran build scripts
    COMPILER_INFO = {"f2py": {"compiler": pc.PC_F2PY[0],
                              "options": f2pyopts}}

    # check material directories
    mtl_dirs = pc.PC_MTLDIRS
    for dirnam in opts.MTLDIRS:
        dirnam = os.path.expanduser(dirnam)
        if not os.path.isdir(dirnam):
            pu.report_error("material directory {0} not found".format(dirnam))
        else:
            mtl_dirs.append(dirnam)
        continue

    if pu.error_count():
        pu.report_and_raise_error("stopping due to previous errors",
                                  tracebacklimit=0)

    if not opts.nobuildlibs:

        # get names of materials from Source/Materials
        errors = 0
        search_dirs = []
        for dirnam in mtl_dirs:
            if not any(x in dirnam for x in search_dirs):
                search_dirs.append(dirnam)
        pu.log_message(
            "finding Payette material model interface files from:\n{0}"
            .format("\n".join([SPACE + x.replace(os.path.expanduser("~"), "~")
                               for x in search_dirs])))
        MATERIALS = get_payette_mtls(mtl_dirs, opts.mtllib, options)
        pu.log_message(
            "the following materials were found:\n{0}"
            .format(SPACE + ", ".join(['"' + x + '"' for x in MATERIALS])),
            beg="\n")
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

        pu.log_message("<<< IF >>> no other build errors were encountered, "
                       "please let the Payette developers know so a fix "
                       "can be found", pre="")
        return 1
    else:
        pu.log_message("runPayette [-h] executed normally", pre="")

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
        if VERBOSE:
            pu.log_message(message, pre="")
        return 1
    else:
        pu.log_message(
            "testPayette [-k elastic -K kayenta] executed normally", pre="")

    return 0


def write_payette_materials(payette_materials):

    """ Write the Source/Materials/Payette_materials.py file containing a
    dictionary of installed models and model attributes

    """

    # get list of previously installed materials
    try:
        constitutive_models = pickle.load(open(pc.PC_MTLS_FILE, "rb"))
    except IOError:
        constitutive_models = {}
    installed_materials = constitutive_models.keys()

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
    for material in [x for x in installed_materials]:
        if material not in all_materials:
            pu.log_warning("installed material {0} not in payette_materials"
                           .format(material))
            installed_materials.remove(material)
        continue

    # write the PC_MTLS_FILE file
    constitutive_models = {}
    for key, val in payette_materials.items():
        if key in installed_materials:
            constitutive_models[key] = val
        continue

    pu.log_message(
        "writing constitutive model information to: {0}"
        .format("PAYETTE_ROOT" + pc.PC_MTLS_FILE.split(pc.PC_ROOT)[1]),
        beg="\n")
    with open(pc.PC_MTLS_FILE, "wb") as fobj:
        pickle.dump(constitutive_models, fobj)
    pu.log_message("constitutive model information written")
    return


def build_payette_mtls(nproc=1):

    """ build the library files for each material.  most materials are
    are implemented as fortran files and need to be compiled with
    f2py.

    """

    global VERBOSE

    # now build the materials
    requested_builds = [x for x in MATERIALS
                        if MATERIALS[x]["build requested"]]
    if not requested_builds:
        if VERBOSE:
            pu.log_warning("no material libraries to build")

    if VERBOSE:
        pu.log_message(
            "the following materials were requested to be built:\n{0}"
            .format(SPACE + ", ".join(['"' + x + '"' for x in requested_builds])),
            beg="\n")

        pu.log_message("building the requested material libraries", beg="\n")

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

    if VERBOSE:
        pu.log_message("finished building the requested material libraries")

    failed_materials = [MATERIALS[x]["libname"]
                        for x in MATERIALS
                        if MATERIALS[x]["build requested"]
                        and MATERIALS[x]["build failed"]]
    built_materials = [MATERIALS[x]["libname"]
                       for x in MATERIALS
                       if MATERIALS[x]["build requested"]
                       and MATERIALS[x]["build succeeded"]]

    failed_materials=["a fake material"]
    if failed_materials:
        # errors = 55
        pu.log_warning(
            "the following materials WERE NOT built:\n{0}"
            .format(SPACE + ", ".join(['"' + x + '"' for x in failed_materials])),
            beg="\n")

    if built_materials:
        pu.log_message(
            "the following materials WERE built:\n{0}"
            .format(SPACE + ", ".join(['"' + x + '"' for x in built_materials])),
            beg="\n")

    # remove cruft
    for ftmp in [x for x in os.listdir(pc.PC_TOOLS)
              if x.split(".")[-1] in ["so", "o"]]:
        os.remove(ftmp)
        continue

    return BUILD_ERRORS


def _build_lib(material):

    """ build the material library for payette_material """

    global BUILD_ERRORS

    # get attributes
    name = MATERIALS[material]["name"]
    libname = MATERIALS[material]["libname"]
    fort_build_script = MATERIALS[material]["fortran build script"]
    parse_err = MATERIALS[material]["parse error"]
    if VERBOSE:
        pu.log_message("building {0}".format(libname), pre=SPACE, end="...   ")

    if parse_err:
        MATERIALS[material]["build failed"] = True
        if VERBOSE:
            pu.log_warning("{0} skipped due to previous errors".format(libname),
                           beg=SPACE)

    elif fort_build_script is None:
        MATERIALS[material]["build succeeded"] = True
        if VERBOSE:
            pu.log_message("{0} built ".format(libname), pre="")

    else:
        # import fortran build script
        py_mod, py_path = pu.get_module_name_and_path(fort_build_script)
        fobj, pathname, description = imp.find_module(py_mod, py_path)
        build = imp.load_module(py_mod, fobj, pathname, description)
        fobj.close()

#        try:
        build = build.Build(name, libname, COMPILER_INFO)
        build_error = build.build_extension_module()

#        except BuildError as error:
#            build_error = error.errno

        if build_error:
            BUILD_ERRORS += 1
            MATERIALS[material]["build failed"] = True
            if build_error == 5 or build_error == 10 or build_error == 40:
                pass
            elif build_error == 66:
                pu.log_warning("{0}: missing attribute: build"
                               .format(fort_build_script), beg="\n"+SPACE)
            elif build_error == 20:
                pu.log_warning("{0} extension module built, but not importable"
                               .format(libname), beg="\n"+SPACE)
            else:
                msg = ("failed to build {0} extension module. see {1}"
                       .format(libname, "build.echo"))
                pu.log_warning(msg, beg="\n"+SPACE)

        else:
            MATERIALS[material]["build succeeded"] = True
            if VERBOSE:
                pu.log_message("{0} built ".format(libname), pre="")

        # remove bite compiled files
        try:
            os.remove(fort_build_script + "c")
        except OSError:
            pass

    return [material, MATERIALS[material]] # end of _build_lib


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
    for mtl_dir in mtl_dirs:
        for dirnam, dirs, files in os.walk(mtl_dir):
            py_files.extend([os.path.join(dirnam, x)
                             for x in files
                             if x.endswith(".py")
                             and "__init__.py" not in x
                             and "Build_" not in x
                             ])
            continue
        continue

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
            pu.log_warning("No name attribute given in {0}, skipping"
                           .format(py_file))
            continue

        name = name.replace(" ", "_").lower()
        libname = attributes.get("libname", name + pc.PC_EXT_MOD_FEXT)

        # material type
        material_type = attributes.get("material type")
        if material_type is None:
            pu.log_warning("No material type attribute given in {0}, skipping"
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

        # models can be in one or more languages
        model_code_types = attributes.get("code types")
        if model_code_types is None:
            pu.log_warning("Attribute 'code types' not found in {0}.attributes"
                           .format(py_module))
        elif not isinstance(model_code_types, tuple):
            model_code_types = (model_code_types, )

        # fortran model set up
        if model_code_types is not None:
            fortran_source = "fortran" in model_code_types
        else:
            # old way to be depricated
            fortran_source = attributes.get("fortran source", False)
            if fortran_source:
                pu.log_warning(
                    "Using depricated 'fortran source' in {0}.attributes"
                    .format(py_mod))

        fort_build_script = attributes.get("build script")
        if fort_build_script is not None:
            pu.log_warning(
                "Using depricated 'build script' in {0}.attributes"
                .format(py_mod))
        else:
            fort_build_script = attributes.get("fortran build script")
        depends = attributes.get("depends")

        # all fortran models must give a fortran build script
        if fortran_source and fort_build_script is None:
            parse_err = True
            pu.log_warning(
                "No fortran build script given for fortran source in "
                "{0} for {1}".format(py_file, libname), pre=SPACE)

        # unless it is not needed...
        elif fort_build_script == "Not_Needed":
            fort_build_script = None

        # and the fortran build script must exist.
        elif fort_build_script is not None:
            if not os.path.isfile(fort_build_script):
                parse_err = True
                pu.log_warning("fortran build script {0} not found"
                               .format(fort_build_script))

        # collect all parts
        #print("="*70)
        #print(attributes)
        #print("="*70)
        mtl_dict = {
            "name": name,
            "libname": libname,
            "fortran source": fortran_source,
            "fortran build script": fort_build_script,
            "aliases": aliases,
            "material type": material_type,
            "material database": attributes.get('material database'),
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
        pu.log_warning("requested material[s] {0} not found"
                       .format(", ".join(non_existent)))

    return payette_materials


def build_fail(msg):

    """ warn that the build failed """

    msg = msg.split("\n")
    err = "BUILD FAILED"
    sss = r"*" * int((80 - len(err)) / 2)
    if VERBOSE:
        pu.log_message("\n\n{0} {1} {2}\n".format(sss, err, sss), pre="")
        for line in msg:
            pu.log_message("BUILD FAIL: {0}".format(line), pre="")
        pu.log_message("\n\n", pre="")
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
    code_exts = [".py", ".pyf", "", ".F", ".C", ".f", ".f90"]
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
    if VERBOSE:
        pu.log_message(pc.PC_INTRO, pre="")
        pu.log_message("Summary of Project:", pre="")
        pu.log_message("\tNumber of files in project:         {0:d}"
                       .format(num_files), pre="")
        pu.log_message("\tNumber of directories in project:   {0:d}"
                       .format(num_dirs), pre="")
        pu.log_message("\tNumber of input files in project:   {0:d}"
                       .format(num_infiles), pre="")
        pu.log_message("\tNumber of python files in project:  {0:d}"
                       .format(num_pyfiles), pre="")
        pu.log_message("\tNumber of lines of code in project: {0:d}"
                       .format(num_lines), pre="")
    return


if __name__ == "__main__":

    BUILD = build_payette(sys.argv[1:])

    ERROR, WARN = 0, 0
    if BUILD == 0:
        pu.log_message("buildPayette succeeded", pre="\n", end="\n\n")

    elif BUILD == 55:
        WARN += 1
        pu.log_warning("buildPayette failed to build one or "
                       "more material libraries\n")

    elif BUILD == 75:
        ERROR += 1
        pu.report_error("buildPayette failed due to an unknown error\n")

    else:
        pu.report_error("buildPayette failed\n")
        ERROR += 1

    if not ERROR and not WARN:
        pu.log_message("Enjoy Payette!", pre="")

    elif WARN:
        pu.log_message("You've been warned, tread lightly!", pre="")

    else:
        pu.log_message("Better luck next time!", pre="")

    sys.exit(BUILD)

