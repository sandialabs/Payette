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

"""Main Payette building file.
None of the functions in this file should be called directly, but only through
the executable script PAYETTE_ROOT/Toolset/buildPayette

AUTHORS
Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
M. Scot Swan, Sandia National Laboratories, mswan@sandia.gov

"""
import os
import sys
import imp
import subprocess as sbp
import multiprocessing as mp
import pyclbr
from textwrap import fill as textfill
try:
    import cPickle as pickle
except ImportError:
    import pickle

import Source.__config__ as cfg
import Source.Payette_utils as pu
import Source.__runopts__ as ro
import Source.Payette_xml_parser as px
import Source.Payette_model_index as pmi
from Source.Payette_xml_parser import XMLParserError as XMLParserError
from Payette_utils import PayetteError as PayetteError
from Source.Payette_utils import PassThroughOptionParser

# --- module level constants
SPACE = "      "  # spacing used for logs to console

# python 3 compatibility
try:
    unicode
except NameError:
    basestring = unicode = str


def build_payette(argv):

    """ create/build: material library files """

    # *************************************************************************
    # -- command line option parsing
    usage = ("usage: buildPayette  [options]")
    parser = PassThroughOptionParser(usage=usage, version="buildPayette 1.0")
    parser.add_option(
        "-m",
        dest="MTLLIB",
        action="append",
        default=[],
        help="specify material libraries to build: [default: ['all']]")
    parser.add_option(
        "-w",
        dest="WIPE",
        action="store_true",
        default=False,
        help="Wipe before building Payette_materials.py [default:%default]")
    parser.add_option(
        "-v",
        dest="VERBOSITY",
        action="store",
        default=ro.VERBOSITY,
        type=int,
        help="Verbosity [default: %default]")

    # the following options are shortcuts for building specific materials
    parser.add_option(
        "--dsf",
        dest="DSF",
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
        "-j",
        dest="NPROC",
        action="store",
        type=int,
        default=1,
        help="number of simultaneous jobs [default: %default]")
    parser.add_option(
        "--dbg",
        dest="DBG",
        action="store_true",
        default=False,
        help="Compile (f2py) with debug information [default: %default]")
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
    parser.add_option(
        "-A",
        dest="AUXMTL",
        action="store",
        default=None,
        help=("Augment existing materials by building in place the "
              "materials in the passed directory [default: %default]."))

    (opts, args) = parser.parse_args(argv)

#    if len(args) > 0:
#       parser.print_help()
#       parser.error("buildPayette does not require arguments, only options")

    ro.set_global_option("VERBOSITY", opts.VERBOSITY, default=True)

    pu.log_message(cfg.INTRO, pre="")

    # determine if we build all materials, or just a selection
    requested_materials = opts.MTLLIB
    if opts.DSF:
        requested_materials.append("domain_switching_ferroelectric")
    if opts.KMM:
        requested_materials.append("kayenta")
    if opts.LPC:
        requested_materials.append("piezo_ceramic")

    # force a rebuild by wiping the existing installed materials file
    if opts.WIPE:
        pmi.remove_index_file()

    # directories to search for materials
    search_directories = []

    if opts.AUXMTL is not None:
        # user can specify AUXMTL as either
        # -A/path/to/auxdir
        # in which case we build the materials in place in /path/to/auxdir, or
        # -A/path/to/auxdir:mtldb_name
        # in which case we build the materials in DOTPAYETTE
        auxmtl = opts.AUXMTL.split(os.pathsep)
        mtldir = auxmtl[0]
        if not os.path.isdir(mtldir):
            parser.error("{0} not found".format(mtldir))
        try:
            payette_mtls_file = os.path.join(cfg.DOTPAYETTE, auxmtl[1])
            if not payette_mtls_file.endswith(".db"):
                payette_mtls_file += ".db"
        except IndexError:
            payette_mtls_file = os.path.join(mtldir, cfg.AUXDB)

        payette_libdir = os.path.dirname(payette_mtls_file)
        search_directories.extend(walk_mtldirs(mtldir))

    else:
        payette_mtls_file = cfg.MTLDB
        payette_libdir = cfg.LIBDIR
        search_directories.extend(cfg.MTLDIRS)

    search_directories = list(set(search_directories))

    # prepare compiler options
    f2pyopts = ["--f{0}exec={1}".format(x, cfg.F2PY["fexe"]) for x in (
        "77", "90")]
    if opts.DBG:
        f2pyopts.append("--debug")

    # compiler options to send to the fortran build scripts
    compiler_info = {"f2py": {"options": f2pyopts}, }

    # intro message
    pu.log_message("Building Payette materials")

    # instantiate the BuildPayette object
    build = BuildPayette(
        search_directories=search_directories,
        requested_materials=requested_materials,
        compiler_info=compiler_info,
        libdir=payette_libdir,
        verbosity=opts.VERBOSITY)

    build.collect_all_materials()

    nproc = min(mp.cpu_count(), opts.NPROC)
    build.build_libraries(nproc=nproc)

    build.write_installed_materials_file(payette_mtls_file)

    return build.errors


class BuildPayette(object):

    def __init__(self, search_directories=None, requested_materials=None,
                 compiler_info=None, libdir=None, verbosity=1):

        # verify each search directory exists
        if not search_directories:
            pu.report_and_raise_error("No search directories given.")

        for directory in search_directories:
            if not os.path.isdir(directory):
                pu.report_error("search directory {0} not found"
                                .format(directory))
            continue
        if pu.error_count():
            pu.report_and_raise_error("Stopping due to previous errors.")
        self.search_directories = search_directories

        # verify that materials were requested
        if requested_materials is None:
            pu.report_and_raise_error("No materials requested to be built.")
        self.requested_materials = requested_materials

        # compiler info needed to build
        if compiler_info is None:
            compiler_info = {}
        self.compiler_info = compiler_info
        self.errors = 0

        if libdir is None:
            libdir = cfg.LIBRARY
        self.libdir = libdir
        self.verbosity = verbosity

        pass

    def collect_all_materials(self):
        """Look through search directories for Payette materials"""

        # tell the users which directories we'll be looking in
        search_dirs = []
        for dirnam in self.search_directories:
            if not any(x in dirnam for x in search_dirs):
                search_dirs.append(dirnam)
            continue
        if self.verbosity > 1:
            pu.log_message(
                "finding Payette material model interface files from:\n{0}"
                .format("\n".join([SPACE + x.replace(os.path.expanduser("~"), "~")
                                   for x in search_dirs])))

        self._get_payette_mtls()

        pu.log_message(
            "the following materials will be built:\n{0}"
            .format(textfill(", ".join([x for x in self.materials_to_build]),
                             initial_indent=SPACE,
                             subsequent_indent=SPACE)),
            beg="\n")
        return

    def _get_payette_mtls(self):
        """Read python files in Source/Materials and determine which are
        interface files for material models. If they are, add them to the
        payette_materials dictionary, along with their attributes

        """

        # determine if we want to build only select libraries
        all_mtls = []
        self.materials_to_build = {}
        control_files = []
        for directory in self.search_directories:
            control_files.extend(
                [os.path.join(directory, x) for x in os.listdir(directory)
                 if x.endswith("_control.xml")])
            continue

        # go through control files and get only those that have a Payette
        # specification
        for control_file in control_files:
            xml_lib = px.XMLParser(control_file)
            build_info = None
            try:
                build_info = xml_lib.get_payette_build_info()
            except XMLParserError as error:
                pu.log_warning(error.message)

            if build_info is None:
                continue

            name, aliases, material_type, interface, source_types = build_info
            all_mtls.append(name)
            if name in self.materials_to_build:
                pu.log_warning(
                    "Duplicate material name {0}, skipping".format(name))
                continue
            libname = name + cfg.EXT_MOD_FEXT

            if (self.requested_materials and
                name.lower() not in [x.lower() for x in self.requested_materials]):
                continue

            # check for interface file
            interface_file = [
                x for x in interface
                if os.path.basename(x).startswith("Payette_")
                and os.path.basename(x).endswith(".py")]
            if not interface_file:
                pu.log_warning(
                    "Skipping material '{0}' because no interface file was found"
                    .format(name))
                continue
            interface_file = interface_file[0]

            # check if the material model was derived from the constitutive
            # model base class as required by Payette
            py_mod, py_path = pu.get_module_name_and_path(interface_file)
            class_data = pyclbr.readmodule(py_mod, path=py_path)
            base_classes = ("ConstitutiveModelPrototype", )
            for item, data in class_data.items():
                model_class_name = data.name
                if any(x in pu.get_super_classes(data) for x in base_classes):
                    break
                continue
            else:
                pu.log_warning(
                    "Skipping material '{0}'".format(name) +
                    " because {1} not derived from any of {2}"
                    .format(class_name, ", ".join(base_classes)))
                continue

            # check if a parameterization file exists and check validity
            parameterization_file = [
                x for x in interface
                if os.path.basename(x).endswith("_parameterize.py")]
            if not parameterization_file:
                parameterization_file = None
                param_class_name = None
            else:
                parameterization_file = parameterization_file[0]
                py_mod, py_path = pu.get_module_name_and_path(
                    parameterization_file)
                class_data = pyclbr.readmodule(py_mod, path=py_path)
                base_classes = ("Parameterize", )
                for item, data in class_data.items():
                    param_class_name = data.name
                    if any(x in pu.get_super_classes(data)
                           for x in base_classes):
                        break
                    continue
                else:
                    pu.log_warning(
                        "Ignoring parameterization file '{0}'".format(name) +
                        " because {1} not derived from any of {2}"
                        .format(class_name, ", ".join(base_classes)))
                    parameterization_file = None
                    param_class_name = None

            # get the build script
            try:
                build_script = [
                    x for x in interface
                    if os.path.basename(x).startswith("Build_")][0]
            except IndexError:
                build_script = None

            # all fortran models must give a fortran build script
            fortran_source = "fortran" in source_types
            if fortran_source and build_script is None:
                pu.log_warning(
                    "Skipping material '{0}' because no build script was found"
                    .format(name))
                continue

            self.materials_to_build[name] = {
                "name": name,
                "libname": libname,
                "libdir": self.libdir,
                "build script": build_script,
                "aliases": aliases,
                "interface file": interface_file,
                "control file": control_file,
                "class name": model_class_name,
                "parameterization file": parameterization_file,
                "parameterization class": param_class_name,
            }

            continue

        # the user may have requested to build a material that does not exist,
        # let them know
        non_existent = [x for x in self.requested_materials
                        if x not in self.materials_to_build]
        if non_existent:
            pu.log_warning("Requested material[s] {0} not found. "
                           .format(", ".join(non_existent)) +
                           "Valid materials are:\n{0}"
                           .format(textfill(", ".join([x for x in all_mtls]),
                                            initial_indent=SPACE,
                                            subsequent_indent=SPACE)))
        return

    def build_libraries(self, nproc=1):

        """ build the library files for each material. most materials are are
        implemented as fortran files and need to be compiled with f2py.

        """
        pu.log_message("Building the requested material libraries", beg="\n")

        # build the libraries
        nproc = min(nproc, len(self.materials_to_build))
        requested_builds = [(key, val, self.compiler_info, self.libdir)
                            for key, val in self.materials_to_build.items()]
        if nproc > 1:
            ro.set_global_option("VERBOSITY", False)
            pool = mp.Pool(processes=nproc)
            build_results = pool.map(_build_lib, requested_builds)
            pool.close()
            pool.join()
            ro.set_global_option("VERBOSITY", True)

        else:
            build_results = [_build_lib(material) for material in
                             requested_builds]

        pu.log_message("finished building the requested material libraries")

        # determine which built and which failed
        built, failed = [], []
        for idx, failed_to_build in enumerate(build_results):
            material = requested_builds[idx][0]
            if failed_to_build:
                failed.append(material)
                self.materials_to_build[material]["built"] = False
            else:
                built.append(material)
                self.materials_to_build[material]["built"] = True
            continue

        if failed:
            pu.log_warning(
                "the following materials WERE NOT built:\n{0}"
                .format(textfill(", ".join([x for x in failed]),
                                 initial_indent=SPACE,
                                 subsequent_indent=SPACE)),
                beg="\n", caller="anonymous")

        if built:
            pu.log_message(
                "the following materials WERE built:\n{0}"
                .format(textfill(", ".join([x for x in built]),
                                 initial_indent=SPACE,
                                 subsequent_indent=SPACE)),
                beg="\n")

        if failed:
            if built:
                self.errors = -1
            else:
                self.errors = 1

        # remove cruft
        for ftmp in [x for x in
                     os.listdir(cfg.TOOLSET) if x.endswith(("so", "o"))]:
            os.remove(ftmp)
            continue

        return

    def write_installed_materials_file(self, index_file):
        """ Write the Source/payette_materials.db file containing a
        dictionary of installed models and model attributes

        """

        # get list of previously installed materials
        model_index = pmi.ModelIndex(index_file)

        # remove materials that failed to build from constitutive models, and
        # add materials that were built to constitutive models, if not already
        # in.
        for material, info in self.materials_to_build.items():
            if not info["built"]:
                model_index.remove_model(material)
            else:
                model_index.store(
                    material, info["libname"], info["libdir"],
                    info["class name"], info["interface file"],
                    info["control file"], info["aliases"],
                    info["parameterization file"],
                    info["parameterization class"])
            continue
        model_index.dump()
        return


def _build_lib(args):

    """ build the material library for payette_material """

    material, material_data, compiler_info, libdir = args

    # get attributes
    libname = material_data["libname"]
    build_script = material_data["build script"]
    pu.log_message("building {0}".format(libname), pre=SPACE, end="...   ")

    if build_script is None:
        pu.log_message("{0} built ".format(libname), pre="")
        return 0

    # import build script
    py_mod, py_path = pu.get_module_name_and_path(build_script)
    fobj, pathname, description = imp.find_module(py_mod, py_path)
    build_module = imp.load_module(py_mod, fobj, pathname, description)
    fobj.close()

    try:
        build = build_module.Build(material, libname, libdir, compiler_info)
        build_error = build.build_extension_module()

    except PayetteError as error:
        pu.log_message(error.message)
        build_error = error.errno

    if build_error:
        if build_error == 5 or build_error == 10 or build_error == 40:
            pass
        elif build_error == 66:
            pu.log_warning(
                "build script {0} missing 'build' attribute"
                .format(os.path.basename(build_script)),
                beg="\n" + SPACE)
        elif build_error == 20:
            pu.log_warning(
                "{0} extension module built, but not importable"
                .format(libname), beg="\n" + SPACE)
        else:
            pu.log_warning(
                "failed to build {0} extension module. see {1}"
                .format(libname, "build.echo"),
                beg="\n" + SPACE)

    else:
        pu.log_message("{0} built ".format(libname), pre="")

    # remove bite compiled files
    try:
        os.remove(build_script + "c")
    except OSError:
        pass

    return build_error


def walk_mtldirs(path):
    """Walk through mtldirs and find directories that have Payette control
    files

    """
    if not isinstance(path, (list, tuple)):
        path = [path]
    mtldirs = []
    for mtldir in path:
        mtldirs.extend([dirnam for dirnam, dirs, files in os.walk(mtldir)
                        if (".svn" not in dirnam and
                            ".git" not in dirnam and
                            any("_control.xml" in y for y in files))])
        continue
    return list(set(mtldirs))


if __name__ == "__main__":

    BUILD = build_payette(sys.argv[1:])

    WARN, ERROR = 0, 0
    if BUILD == 0:
        sys.stderr.write("\nINFO: buildPayette succeeded\n")

    elif BUILD < 0:
        WARN += 1
        sys.stderr.write("\nWARNING: buildPayette failed to build one or "
                         "more material libraries\n")

    elif BUILD > 0:
        ERROR += 1
        sys.stderr.write("\nERROR: buildPayette failed\n")

    if not ERROR and not WARN:
        sys.stderr.write("\nEnjoy Payette!\n")

    elif WARN:
        sys.stderr.write("\nYou've been warned, tread lightly!\n")

    else:
        sys.stderr.write("\nBetter luck next time!\n")

    sys.exit(BUILD)
