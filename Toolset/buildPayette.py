#!/usr/bin/env python
"""
NAME
   buildPayette

PURPOSE
   Builds the Payette source. Although Payette is written mainly in python (and
   therefore does not need to be built/compiled), there is still some setup
   required. For example, many of the material models are written in Fortran
   and must be compiled by f2py.  For materials that require this step, a build
   script gives all of the directives for building with f2py.  This script
   moves throught the Source/Materials directory looking for material files and
   associated build scripts and builds the material libraries.  Additonal setup
   is also performed.
"""

from __future__ import print_function
import os,sys
import imp
import optparse
import subprocess as sbp
from linecache import getline
from distutils import sysconfig


# spacing used for logs to console
sp = "      "

build_payette = os.path.realpath(__file__)
this_d = os.path.dirname(build_payette)

class BuildError(Exception):
    def __init__(self, message, errno):
        # errno:
        # 1: bad input files
        # 2: f2py failed
        #  5 = environmental variable not found (not error)
        # 10 = source files/directories not found
        # 35 = Extension module not imported
        # 40 = Bad/no sigfile
        # 66 = No build attribute
        self.message = message
        self.errno = errno
        logwrn(message)
        pass

    def __repr__(self):
        return self.__name__

    def __str__(self):
        return repr(self.errno)


def buildPayette(argc,argv):
    global createx, Payette_Toolset
    """
    NAME
       buildPayette

    PURPOSE
       create/build: runPayette and testPayette scripts, material library files

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    # *************************************************************************
    # -- command line option parsing
    usage = ("usage: python %prog [options]\nmust be executed from "
             "{0}".format(os.path.dirname(build_payette)))
    parser = optparse.OptionParser(usage = usage, version = "%prog 1.0")
    parser.add_option("-x","--no-build-libs",
                      dest = "nobuildlibs",
                      action = "store_true",
                      default = False,
                      help = "do not build material libraries: [default: %default]")
    parser.add_option("-m",
                      dest = "mtllib",
                      action = "append",
                      default = [],
                      help = "material libraries to build: [default: %default]")
    parser.add_option("-t","--test",
                      dest = "TEST",
                      action = "store_true",
                      default = False,
                      help = "run testPayette executable: [default: %default]")
    parser.add_option("-o",
                      dest = "OPTIONS",
                      action = "append",
                      default = [],
                      help = ("Options to build (accumulated) [default: %default] "
                              "[choices: [electromech, special]]"))
    parser.add_option("-F",
                      dest = "FORCEREBUILD",
                      action = "store_true",
                      default = False,
                      help = ("Rebuild Payette_materials.py [default:%default]"))
    parser.add_option("--fcompiler",
                      dest = "FCOMPILER",
                      action = "store",
                      type = "choice",
                      choices = (None,"gnu95"),
                      default = None,
                      help = ("Specify Fortran compiler type by vendor "
                              "[default: %default]"))
    parser.add_option("--f77exec",
                      dest = "F77EXEC",
                      action = "store",
                      default = "gfortran",
                      help = "Specify the path F77 to compiler [default: %default]")
    parser.add_option("--f90exec",
                      dest = "F90EXEC",
                      action = "store",
                      default = "gfortran",
                      help = "Specify the path F90 to compiler [default: %default]")
    parser.add_option("-W",
                      dest = "WRITEONLY",
                      action = "store_true",
                      default = False,
                      help = ("Write configuration files and exit "
                              "[default: %default]"))
    parser.add_option("--clean",
                      dest = "CLEAN",
                      action = "store_true",
                      default = False,
                      help = ("Remove Payette configuraton files and extension "
                              "library files and exit [default: %default]"))

    # the following parser options are shortcuts for building specific materials
    parser.add_option("--dsc", dest = "DSC", action = "store_true", default = False,
                      help = "Build domain_switching_ceramic [default: %default]")
    parser.add_option("--kmm", dest = "KMM", action = "store_true", default = False,
                      help = "Build kayenta [default: %default]")
    parser.add_option("--lpc", dest = "LPC", action = "store_true", default = False,
                      help = "Build pieze_ceramic [default: %default]")
    parser.add_option("--summary",dest="SUMMARY",action="store_true",default=False,
                      help="write summary to screen [default: %default]")

    (opts,args) = parser.parse_args(argv)
    if len(args) > 0:
        parser.print_help()
        parser.error("buildPayette does not require arguments, only options")
        pass

    if opts.SUMMARY:
        write_summary_to_screen()
        sys.exit(0)

    # determine if we build all materials, or just a selection
    if opts.DSC: opts.mtllib.append("domain_switching_ceramic")
    if opts.KMM: opts.mtllib.append("kayenta")
    if opts.LPC: opts.mtllib.append("piezo_ceramic")
    buildselect = any(opts.mtllib)
    libstobuild = ["all"] if not buildselect else opts.mtllib

    if opts.FORCEREBUILD:
        try: os.remove(Payette_Materials_File)
        except: pass
        pass

    # clean up options:
    options = []
    for option in opts.OPTIONS:
        if "elec" in option: options.append("electromechanical")
        elif "special" in options: options.append("special")
        else: options.append(option)
        continue

    if opts.CLEAN:
        loginf("Cleaning Payette... ",end="")
        cleanPayette()
        endmes("Payette cleaned\n")
        sys.exit()


    # intro message
    loginf("Building Payette: An Object Oriented Material Model Driver\n",
           pre="\n")

    # configure Payette
    loginf("configuring Payette environment")
    configure_payette_environment(opts)

    try: import Payette_config
    except ImportError:
        if not os.path.isfile(os.path.dirname(build_payette),"Payette_config.py"):
            logerr("Payette_config.py not written")
            sys.exit(1)
        else: raise

    # write our own f2py script
    Payette_f2py = write_f2py()

    loginf("Payette environment configured\n")

    if opts.WRITEONLY: sys.exit(0)

    # prepare compiler options
    if Payette_fcompiler:
        f2pyopts = ["--fcompiler={0}".format(Payette_fcompiler)]
    else:
        f2pyopts = ["--f77exec={0}".format(Payette_f77exec),
                    "--f90exec={0}".format(Payette_f90exec)]

    # compiler options to send to the build scripts
    compiler_info = {"f2py":{"compiler":Payette_f2py,
                             "options":f2pyopts}}

    # if the user desires to build only a single material library, make sure that
    # the python interpreter used to build that library is the same as what was
    # used to initially build Payette
    createx = True
    if buildselect:
        if (os.path.isfile(Payette_runPayette) and
            os.path.isfile(Payette_testPayette) and
            os.path.isfile(Payette_rebuildPayette)):
            if Payette_pyint != getline(Payette_runPayette,2).split(" ")[0]:
                print("ERROR: attempting to build material library with "
                      "different interpreter than Payette was previously built")
                return 1
            else: createx = False
            pass
        pass

    if createx:
        # remove the executables built by Payette, if the build is successful
        # successful, they will be recreated
        for key,val in Payette_Built_Executables.items():
            try: os.remove(val)
            except: pass
            continue

        # create the runPayette, testPayette, and rebuildPayette executables
        createPayetteExecutables()
        pass

    if not opts.nobuildlibs:

        # get names of materials from Source/Materials
        loginf("finding Payette materials")
        payette_materials = getPayetteMaterials(libstobuild,options)
        loginf("Payette materials found\n")

        # build the requested material libraries
        errors, payette_materials = buildPayetteMaterials(payette_materials,
                                                          compiler_info)

        # material libraries built, now write the
        # Source/Materials/Payette_Materials file containing all materials
        writePayetteMaterials(payette_materials)

    else:
        errors = 0
        pass

    # if the user wanted only to build certain libraries, return when built
    if opts.mtllib: return errors

    # check that runPayette works
    test_error = testRunPayette(opts.TEST)

    if errors and test_error == 0:
        # testRunPayette completed without error, but there were previous build
        # errors, meaning that some libraries did not build, but Payette still
        # built.
        errors = 55
    elif not errors and test_error != 0:
        # testRunPayette completed with error, but everything built fine,
        # something is wrong...
        errors = 75
    else: pass

    return errors


def createPayetteExecutables():
    """
    NAME
       createPayetteExecutables

    PURPOSE
       create the runPayette and testPayette executable scripts

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    loginf("writing executable scripts")

    for item in [[Payette_runPayette,"run"],[Payette_testPayette,"test"]]:
        script,typ = item
        begmes("writing %s"%(os.path.basename(script)),pre=sp)
        execmd ="#!/bin/sh -f\n%s %s %s $* 2>&1\n"%(Payette_pyint,Payette_Payette,typ)
        with open(script,"w") as f: f.write(execmd)
        os.chmod(script,0o750)
        endmes("%s script written"%os.path.basename(script))
        continue

    # Make the rebuildPayette script
    begmes("writing {0}".format(os.path.basename(Payette_rebuildPayette)),pre=sp)
    execmd =("#!/bin/sh -f\ncd {0}\n{1} {2} {3}\n"
             .format(Payette_Toolset,Payette_pyint,
                     build_payette," ".join(sys.argv[1:])))
    with open(Payette_rebuildPayette,"w") as f: f.write(execmd)
    os.chmod(Payette_rebuildPayette,0o750)
    endmes("%s script written"%os.path.basename(Payette_rebuildPayette))

    loginf("executable scripts written\n")
    return 0


def testRunPayette(test):
    """
    NAME
       testRunPayette

    PURPOSE
       test that runPayette executes properly for [-h]
       test that testPayette executes properly for [-k elastic -K kayenta]

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    begmes("INFO: testing that runPayette [-h] executes normally",pre="")
    cmd = [Payette_runPayette,"-h"]
    runcheck = sbp.Popen(cmd,stdout=sbp.PIPE,stderr=sbp.STDOUT)
    runcheck.wait()
    if runcheck.returncode != 0:
        sbp_msg = runcheck.communicate()[0]
        if type(sbp_msg) == bytes:
            sbp_msg = sbp_msg.decode("ascii")
        msg = [x for x in sbp_msg.split("\n") if x]
        message = ("the following error was trapped from runPayette [-h]:\n"
                   "%s"%("="*25+" Start Error\n"+sbp_msg+"\n"+"="*25+" End Error\n"))
        buildfail(message)

        print("<<< IF >>> no other build errors were encountered,")
        if "compatible f2py" in msg[-1].lower():
            print("this error might be a result of attempting to use "
                  "the python bundled with sage  it is not known "
                  "at this time why the python libraries compiled \n"
                  "sage\"s f2py are not being imported properly.  please "
                  "build Payette with\npython2.{6,7} or python3.? until a "
                  "fix is found\n")
        else: print ("please let the Payette developers know so a fix "
                     "can be found")
        loginf("removing {runPayette,testPayette} and exiting")
        for key,val in Payette_Built_Executables.items():
            try: os.remove(val)
            except: pass
            continue
        return 1
        pass
    else:
        endmes("runPayette [-h] executed normally\n")
        pass

    if not test: return 0

    begmes("INFO: testing that testPayette [-k elastic -K kayenta] "
             "executes normally")
    cmd = [os.path.join(tools_d,"testPayette"),"-k","elastic","-K","kayenta","-F",
           "buildPayette"]
    runcheck = sbp.Popen(cmd,stdout=sbp.PIPE,stderr=sbp.STDOUT)
    runcheck.wait()
    if runcheck.returncode != 0:
        sbp_msg = runcheck.communicate()[0]
        if type(sbp_msg) == bytes:
            sbp_msg = sbp_msg.decode("ascii")
        msg = [x for x in sbp_msg.split("\n") if x]
        message = ("the following error was trapped from "
                   "testPayette [-k elastic -K kayenta]:\n%s"%(msg[-1]))
        buildfail(message)
        message = ("please let the Payette developers know so a fix can be found\n"
                   "removing {runPayette,testPayette} and exiting")
        print(message)
        for name, exe in Payette_Built_Executables.items():
            try: os.remove(exe)
            except: pass
            continue
        return 1
        pass
    else:
        endmes("testPayette [-k elastic -K kayenta] executed normally\n")
        pass
    return 0


def writePayetteMaterials(payette_materials):
    """
        Write the Source/Materials/Payette_materials.py file containing a
        dictionary of installed models and model attributes
    """

    loginf("writing {0}"
           .format("$PAYETTE_HOME" + Payette_Materials_File.split(Payette_Home)[1]))

    try: lines = open(Payette_Materials_File,"r").readlines()
    except: lines = []

    # get list of previously installed materials
    installed_materials = []
    for line in lines:
        if "Payette_Installed_Materials" in line:
            installed_materials = eval(line.strip().split("=")[1])
            break
        continue

    # remove failed materials from installed materials
    for material in [ x for x in payette_materials
                      if payette_materials[x]["build requested"]
                      and payette_materials[x]["build failed"] ]:
        try: installed_materials.remove(material)
        except: pass
        continue

    # add built materials to installed materials
    for material in [ x for x in payette_materials
                      if payette_materials[x]["build requested"]
                      and payette_materials[x]["build succeeded"] ]:
        if material not in installed_materials:
            installed_materials.append(material)
            pass
        continue

    # check that all installed materials are in all_materials
    all_materials = [payette_materials[x]["name"] for x in payette_materials]
    for material in installed_materials:
        if material not in all_materials:
            logwrn("installed material {0} not in payette_materials"
                   .format(material))
            pass
        continue
    installed_materials = [ x for x in installed_materials if x in all_materials]

    # write the Payette_Materials_File file
    with open(Payette_Materials_File,"w") as f:
        intro = (
"""# ****************************************************************************** #
#                                                                                #
# This file was generated automatically by the Payette. It contains important    #
# directives for materials detected and built by Payette.                        #
#                                                                                #
# This file is intended to be imported by Payette so that material classes can   #
# be instantiated from built materials.                                          #
#                                                                                #
# DO NOT EDIT THIS FILE. This entire file is regenerated automatically each time #
# Payette is built. Any changes you make to this file will be overwritten.       #
#                                                                                #
# If changes are needed, please contact the Payette developers so that changes   #
# can be made to the build scripts.                                              #
# ****************************************************************************** #
""")
        f.write(intro)
        begmes("writing successfully installed materials",pre=sp)
        f.write("Payette_Installed_Materials={0}\n".format(installed_materials))
        endmes("successfully installed materials written")

        f.write("Payette_Constitutive_Models={}\n")

        begmes("writing constitutive model declarations",pre=sp)
        for material in installed_materials:
            payette_material = payette_materials[material]
            py_module = payette_material["module"]
            name = payette_material["name"]
            aliases = payette_material["aliases"]
            class_name = payette_material["class name"]
            py_path = "Source.Materials." + py_module
            f.write("from {0} import {1}\n".format(py_path, class_name))
            f.write('Payette_Constitutive_Models["{0}"]='
                    '{{"class name":{1},'
                    '"aliases":{2}}}\n'.format(name,class_name,aliases))
            continue
        pass
        endmes("constitutive model declarations written")

    loginf("{0} written\n"
           .format("$PAYETTE_HOME" + Payette_Materials_File.split(Payette_Home)[1]))
    return


def buildPayetteMaterials(payette_materials,compiler_info):
    """
    NAME
       buildPayetteMaterials

    PURPOSE
       build the library files for each material.  most materials are
       are implemented as fortran files and need to be compiled with
       f2py.

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    loginf("building Payette material libraries")

    # now build the materials
    errors = 0

    requested_builds = [ x for x in payette_materials
                         if payette_materials[x]["build requested"]]
    if not requested_builds: logmes("no material libraries to build",pre=sp)

    for material in requested_builds:

        # get attributes
        name = payette_materials[material]["name"]
        libname = payette_materials[material]["libname"]
        build_script = payette_materials[material]["build script"]
        parse_error = payette_materials[material]["parse error"]
        begmes("building %s" % (libname),pre=sp)

        if parse_error:
            payette_materials[material]["build failed"] = True
            endmes("{0} skipped due to previous errors".format(libname))
            continue

        if not build_script:
            payette_materials[material]["build succeeded"] = True
            endmes("{0} built ".format(libname))
            continue

        # import build script
        py_mod, py_path = get_module_name_and_path(build_script)
        fp, pathname, description = imp.find_module(py_mod,py_path)
        build = imp.load_module(py_mod,fp,pathname,description)
        fp.close()

        try:
            build = build.Build(name,libname,compiler_info)
            build_error = build.build_extension_module()

#        except AttributeError:
#            build_error = 66

        except BuildError as error:
            build_error = error.errno
            pass

        except Exception as error:
            build_error = error.errno
            if build_error not in [1,2,5,10,35,40,66]:
                # raise what ever error came through
                raise
            else:
                if hasattr(error,"message"):
                    logwrn(error.message,pre=sp)
                else: raise
                pass
            pass

        if build_error:
            errors += 1
            payette_materials[material]["build failed"] = True
            if build_error == 5 or build_error == 10 or build_error == 40:
                pass
            elif build_error == 66:
                logwrn("{0}: missing attribute: build".format(build_script))
            else:
                msg = ("failed to build {0} extension module. see {1}"
                       .format(libname,"build.echo"))
                logwrn(msg,pre="\t\t")
                pass
        else:
            endmes("%s built " % (libname))
            payette_materials[material]["build succeeded"] = True
            pass

        # remove bite compiled files
        try: os.remove(build_script + "c")
        except: pass

        continue

    loginf("Payette material libraries built\n")
    failed_materials = [payette_materials[x]["libname"] for x in payette_materials
                        if payette_materials[x]["build requested"]
                        and payette_materials[x]["build failed"]]
    built_materials = [payette_materials[x]["libname"] for x in payette_materials
                       if payette_materials[x]["build requested"]
                       and payette_materials[x]["build succeeded"]]

    if failed_materials:
        errors = 55
        logwrn("The following material libraries WERE NOT built: {0}\n"
               .format(", ".join(failed_materials)))
        pass

    if built_materials:
        loginf("The following material libraries WERE built: {0}\n"
               .format(", ".join(built_materials)))
        pass

    # remove cruft
    for f in [x for x in os.listdir(Payette_Toolset)
              if x.split(".")[-1] in ["so","o"]]:
        os.remove(f)
        continue

    return errors, payette_materials


def getPayetteMaterials(requested_libs=["all"],options=[]):

    """
        Read python files in Source/Materials and determine which are interface
        files for material models.  If they are, add them to the payette_materials
        dictionary, along with their attributes
    """

    import pyclbr

    buildselect = False if requested_libs[0] == "all" else True

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

    payette_materials, material_depends = {}, {}
    py_files = [os.path.join(Payette_Materials,x)
                for x in os.listdir(Payette_Materials) if x.endswith(".py")]

    # go through list of python files in
    for py_file in py_files:
        verb = "qsfail" in py_file

        parse_error = False

        py_mod, py_path = get_module_name_and_path(py_file)
        fp, pathname, description = imp.find_module(py_mod,py_path)
        py_module = imp.load_module(py_mod,fp,pathname,description)
        fp.close()

        attributes = getattr(py_module,"attributes",None)
        if not attributes: continue

        # check if this is a payette material
        try: payette_material = attributes["payette material"]
        except: continue
        if not payette_material: continue

        # check if a constitutive model class is defined
        class_data = pyclbr.readmodule(py_mod,path=[os.path.dirname(py_file)])

        try: proto = class_data["Parent"].name
        except: proto = "ConstitutiveModelPrototype"

        for name, data in class_data.items():
            class_name = data.name
            constitutive_model = proto in get_super_classes(name,data)
            if constitutive_model: break
            continue

        if not constitutive_model:
            del py_module
            continue

        # file is an interface file check attributes, define defaults
        try: name = attributes["name"]
        except: logerr("No name attribute given in {0}".format(py_file))
        name = name.replace(" ","_").lower()

        try: libname = attributes["libname"]
        except: libname = name + Payette_Extension_Module_Fext

        # material type
        try: material_type = attributes["material type"]
        except: logerr("No material type attribute given in {0}".format(py_file))
        electromtl = bool([x for x in material_type if "electro" in x])
        specialmtl = bool([x for x in material_type if "special" in x])

        # get aliases, they need to be a list of aliases
        try: aliases = attributes["aliases"]
        except: aliases = []
        if isinstance(aliases,str): aliases = [aliases]
        aliases = [ x.replace(" ","_").lower() for x in aliases]

        # fortran model set up
        try: fortran_source = attributes["fortran source"]
        except: fortran_source = False
        try: build_script = attributes["build script"]
        except: build_script = None
        try: depends = attributes["depends"]
        except: depends = None

        # all fortran models must give a build script
        if fortran_source and not build_script:
            parse_error = True
            logerr("No build script given for fortran source in {0} for {1}"
                   .format(py_file,libname),pre=sp)

        # unless it is not needed...
        elif build_script == "Not_Needed":
            build_script = None

        # and the build script must exist.
        elif build_script != None:
            if not os.path.isfile(build_script):
                parse_error = True
                logerr("build script {0} not found".format(build_script))
                pass
            pass

        # collect all parts
        mtl_dict = {
            "name":name,
            "libname":libname,
            "fortran source":fortran_source,
            "build script":build_script,
            "aliases":aliases,
            "material type":material_type,
            "module":py_mod,
            "file":py_file,
            "class name":class_name,
            "depends":depends,
            "parse error":parse_error,
            "build requested":False, # by default, do not build the material
            "build succeeded":False,
            "build failed":False
            }

        payette_materials[name] = mtl_dict
        # payette_materials[py_mod] = mtl_dict
        del py_module

        # if building select libraries, check if this is it
        if buildselect and name not in requested_libs: continue

        elif not buildselect:
            # only build electromechanical models if requested
            if electromtl and "electromechanical" not in options: continue
            elif specialmtl and "special" not in options: continue
            else: pass

        else: pass

        # by this point, we have filtered out the materials we do not want to
        # build, so request that it be built
        payette_materials[name]["build requested"] = True

        continue

    dependent_materials = [ x for x in payette_materials
                            if payette_materials[x]["depends"] ]
    if dependent_materials:
        for material in dependent_materials:
            depends_on = payette_materials[material]["depends"]
            # user has requested to build a material that depends on another.
            # make sure that the other material exists
            if depends_on not in payette_materials:
                raise BuildError("{0} depends on {1} which was not found"
                                 .format(material,depends_on),25)

            # if material was requested to be built, make sure the material it
            # depends on is also built
            if payette_materials[material]["build requested"]:
                payette_materials[depends_on]["build requested"] = True
                pass

            continue
        pass

    if buildselect:
        # the user may have requested to build a material that does not exist,
        # let them know
        all_names = [payette_materials[x]["name"] for x in payette_materials]
        non_existent = []
        for name in requested_libs:
            if name not in all_names: non_existent.append(name)
            continue
        if non_existent:
            logwrn("requested material[s] {0} not found"
                   .format(", ".join(non_existent)),pre=sp)
            pass
        pass

    return payette_materials


def buildfail(msg):
    msg = msg.split("\n")
    err = "BUILD FAILED"
    s = r"*"*int((80 - len(err))/2)
    print("\n\n%s %s %s\n"%(s,err,s))
    for line in msg: print("BUILD FAIL: %s"%(line))
    print("\n\n")
    return


def get_module_name_and_path(py_file):
    return os.path.splitext(os.path.basename(py_file))[0],[os.path.dirname(py_file)]


def begmes(msg,pre="",end="  "):
    print("{0}{1}...".format(pre,msg),end=end)
    return


def endmes(msg,pre="",end="\n"):
    print("{0}{1}".format(pre,msg),end=end)
    return


def loginf(msg,pre="",end="\n"):
    print("{0}INFO: {1}".format(pre,msg),end=end)
    return

def logmes(msg,pre="",end="\n"):
    print("{0}{1}".format(pre,msg),end=end)
    return


def logwrn(msg,pre="",end="\n"):
    print("{0}WARNING: {1}".format(pre,msg),end=end)
    return


def logerr(msg,pre="",end="\n"):
    print("{0}ERROR: {1}".format(pre,msg),end=end)
    return


def get_fortran_compiler(f77,f90):
    """
        get absolute path to fortran compilers
    """
    f77exec, f90exec = None, None
    if os.path.isfile(f77): f77exec = f77
    if os.path.isfile(f90): f90exec = f90

    if f77exec and f90exec: return f77exec, f90exec

    try: path = os.getenv("PATH").split(os.pathsep)
    except: path = []

    for dirname in path:
        if not f77exec and os.path.isfile(os.path.join(dirname,f77)):
            f77exec = os.path.join(dirname,f77)
        if not f90exec and os.path.isfile(os.path.join(dirname,f90)):
            f90exec = os.path.join(dirname,f90)
            pass
        if f77exec and f90exec: return f77exec, f90exec
        continue
    if f90exec: return f90exec, f90exec

    logerr("fortran compiler not found")
    sys.exit(1)


def write_f2py():
    # write out f2py. we write out our own to ensure that we use the right python
    # interpreter. I just copied this verbatim from my installation of f2py,
    # replacing the interpreter on the shebang line with Payette_pying
    #
    # HOWEVER, after writing this, I remembered that we never use f2py from the
    # command line, but import it directly from numpy, so this is unnecessary...
    f2py_file = """#!{0}
# See http://cens.ioc.ee/projects/f2py2e/
import os, sys
for mode in ["g3-numpy", "2e-numeric", "2e-numarray", "2e-numpy"]:
    try:
        i=sys.argv.index("--"+mode)
        del sys.argv[i]
        break
    except ValueError: pass
os.environ["NO_SCIPY_IMPORT"]="f2py"
if mode=="g3-numpy":
    sys.stderr.write("G3 f2py support is not implemented, yet.\\n")
    sys.exit(1)
elif mode=="2e-numeric":
    from f2py2e import main
elif mode=="2e-numarray":
    sys.argv.append("-DNUMARRAY")
    from f2py2e import main
elif mode=="2e-numpy":
    from numpy.f2py import main
else:
    sys.stderr.write("Unknown mode: " + repr(mode) + "\\n")
    sys.exit(1)
main()
""".format(Payette_pyint)
    f2py = os.path.join(Payette_Toolset,"f2py")
    with open(f2py,"w") as f:
        for line in f2py_file: f.write(line)
        pass
    os.chmod(f2py,0o750)
    return f2py


def configure_payette_environment(opts):
    """
       set up the Payette environment and write Payette_config.py
    """
    import shutil
    global Payette_pyint, Payette_Home, Payette_Aux, Payette_Documents
    global Payette_Source, Payette_Tests, Payette_Toolset, Payette_Materials
    global Payette_MIG_Utils, Payette_Materials_Library
    global Payette_Materials_Fortran, Payette_Materials_Fortran_Includes
    global Payette_Materials_File, Payette_Inputs, Payette_Extension_Module_Fext
    global Payette_ostype, Payette_F2Py_Callback, Payette_Payette
    global Payette_runPayette, Payette_testPayette, Payette_rebuildPayette
    global Payette_Built_Executables, Payette_Executables
    global Payette_config_file
    global Payette_Kayenta, Payette_AlegraNevada, Payette_nlopt
    global Payette_fcompiler, Payette_f77exec, Payette_f90exec
    global assertion_errors

    # begin configuration
    assertion_errors = 0
    payette_config = {}

    def check_exists(itemnam,item):
        global assertion_errors
        if not item:
            assertion_errors += 1
            logerr("{0} not found".format(itemnam))
            pass
        elif not os.path.isdir(item) and not os.path.isfile(item):
            assertion_errors += 1
            print("{0} not found".format(item))
        else: pass
        return

    # python interpreter info
    Payette_pyint = os.path.realpath(sys.executable)
    (major, minor, micro, releaselevel, serial) = sys.version_info
    if (major != 3 and major != 2) or (major == 2 and minor < 6):
        raise SystemExit("Payette requires Python >= 2.6\n")

    # compatibility checks
    if "sage" in Payette_pyint:
        logerr("Payette not compatible sage -python")
        assertion_errors += 1
        pass

    # numpy check
    try: import numpy
    except:
        logerr("numpy not importable")
        assertion_errors += 1
        pass

    # scipy check
    try: import scipy
    except:
        logerr("scipy not importable")
        assertion_errors += 1
        pass

    # Payette home
    Payette_Home = os.path.dirname(os.path.dirname(build_payette))
    check_exists("Payette_Home",Payette_Home)

    # Root level directories
    Payette_Aux = os.path.join(Payette_Home,"Aux")
    Payette_Documents = os.path.join(Payette_Home,"Documents")
    Payette_Source = os.path.join(Payette_Home,"Source")
    Payette_Tests = os.path.join(Payette_Home,"Tests")
    Payette_Toolset = os.path.join(Payette_Home,"Toolset")
    check_exists("Payette_Aux",Payette_Aux)
    check_exists("Payette_Documents",Payette_Documents)
    check_exists("Payette_Source",Payette_Source)
    check_exists("Payette_Tests",Payette_Tests)
    check_exists("Payette_Toolset",Payette_Toolset)

    # subdirectories of Payette_Source
    Payette_Materials = os.path.join(Payette_Source,"Materials")
    Payette_MIG_Utils = os.path.join(Payette_Source,"Fortran/MIGUtils/migutils.F")
    check_exists("Payette_Materials",Payette_Materials)
    check_exists("Payette_MIG_Utils",Payette_MIG_Utils)

    # Subdirectories of Payette_Materials
    Payette_Materials_Library = os.path.join(Payette_Materials,"Library")
    Payette_Materials_Fortran = os.path.join(Payette_Materials,"Fortran")
    Payette_Materials_Fortran_Includes = os.path.join(Payette_Materials,
                                                      "Fortran/Includes")
    Payette_Materials_File = os.path.join(Payette_Materials,
                                          "Payette_installed_materials.py")
    check_exists("Payette_Materials_Library",Payette_Materials_Library)
    check_exists("Payette_Materials_Fortran",Payette_Materials_Fortran)
    check_exists("Payette_Materials_Fortran_Includes",
                 Payette_Materials_Fortran_Includes)

    # subdirectories of Payette_Aux
    Payette_Inputs = os.path.join(Payette_Home,"Aux/Inputs")
    check_exists("Payette_Inputs",Payette_Inputs)

    # extension module file extension
    Payette_Extension_Module_Fext = sysconfig.get_config_var("SO")

    # if OSTYPE is not defined, just set it to linux
    if not os.getenv("OSTYPE"):
        logwrn("environment variable OSTYPE not set, "
               "setting Payette_ostype to linux")
        Payette_ostype = "linux"
    else:
        Payette_ostype = os.getenv("OSTYPE").lower()
    # f2py call back
    Payette_F2Py_Callback = major != 3 and "linux" not in Payette_ostype.lower()
    Payette_F2Py_Callback = True

    # Payette executables
    Payette_Payette = os.path.join(Payette_Toolset,"Payette")
    Payette_runPayette = os.path.join(Payette_Toolset,"runPayette")
    Payette_extractPayette = os.path.join(Payette_Toolset,"extractPayette.py")
    check_exists("extractPayette",Payette_extractPayette)
    Payette_testPayette = os.path.join(Payette_Toolset,"testPayette")
    Payette_rebuildPayette = os.path.join(Payette_Toolset,"rebuildPayette")
    Payette_Built_Executables = {"runPayette":Payette_runPayette,
                                 "rebuildPayette":Payette_rebuildPayette,
                                 "testPayette":Payette_testPayette}
    Payette_Executables = {"extractPayette.py":Payette_extractPayette}
    for key,val in Payette_Built_Executables.items():
        Payette_Executables[key] = val
        continue

    # configuration files
    Payette_config_file = os.path.join(Payette_Toolset,"Payette_config.py")
    try: os.remove(Payette_config_file)
    except: pass
    try: os.remove(Payette_config_file + "c")
    except: pass

    # auxilary Payette environment variables
    Payette_Kayenta = os.getenv("PAYETTE_KAYENTA")
    Payette_AlegraNevada = os.getenv("PAYETTE_ALEGRANEVADA")
    Payette_nlopt = os.getenv("NLOPTLOC")

    Payette_fcompiler = opts.FCOMPILER
    if not Payette_fcompiler:
        Payette_f77exec, Payette_f90exec = get_fortran_compiler(opts.F77EXEC,
                                                                opts.F90EXEC)
    else: Payette_f77exec, Payette_f90exec = None, None

    if assertion_errors:
        sys.exit("Payette_config.py: ERROR: fix previously trapped errors")
        pass

    # modify sys.path
    if Payette_Home not in sys.path: sys.path.insert(0,Payette_Home)

    # store all of the above information for writing to the Payette_config_file,
    # we waited tpo write it til now so that we would only write it if everything
    # was configured correctly.
    payette_config["Payette_pyint"] = Payette_pyint
    payette_config["Payette_Home"] = Payette_Home
    payette_config["Payette_Aux"] = Payette_Aux
    payette_config["Payette_Documents"] = Payette_Documents
    payette_config["Payette_Source"] = Payette_Source
    payette_config["Payette_Tests"] = Payette_Tests
    payette_config["Payette_Toolset"] = Payette_Toolset
    payette_config["Payette_Materials"] = Payette_Materials
    payette_config["Payette_MIG_Utils"] = Payette_MIG_Utils
    payette_config["Payette_Materials_Library"] = Payette_Materials_Library
    payette_config["Payette_Materials_Fortran"] = Payette_Materials_Fortran
    payette_config["Payette_Materials_Fortran_Includes"] = (
        Payette_Materials_Fortran_Includes)
    payette_config["Payette_Materials_File"] = Payette_Materials_File
    payette_config["Payette_Inputs"] = Payette_Inputs
    payette_config["Payette_Extension_Module_Fext"] = Payette_Extension_Module_Fext
    payette_config["Payette_ostype"] = Payette_ostype
    payette_config["Payette_F2Py_Callback"] = Payette_F2Py_Callback
    payette_config["Payette_Payette"] = Payette_Payette
    payette_config["Payette_Executables"] = Payette_Executables
    payette_config["Payette_config_file"] = Payette_config_file
    payette_config["Payette_Kayenta"] = Payette_Kayenta
    payette_config["Payette_AlegraNevada"] = Payette_AlegraNevada
    payette_config["Payette_nlopt"] = Payette_nlopt
    payette_config["Payette_fcompiler"] = Payette_fcompiler
    payette_config["Payette_f77exec"] = Payette_f77exec
    payette_config["Payette_f90exec"] = Payette_f90exec

    intro = (
"""# ****************************************************************************** #
#                                                                                #
# This file was generated automatically by the Payette. It contains important    #
# global Payette parameters that are configured at build time.                   #
#                                                                                #
# This file is intended to be imported by Payette using                          #
# "from Payette_config import *"                                                 #
#                                                                                #
# DO NOT EDIT THIS FILE. This entire file is regenerated automatically each time #
# buildPayette is run. Any changes you make to this file will be overwritten.    #
#                                                                                #
# If changes are needed, please contact the Payette developers so that changes   #
# can be made to the build scripts.                                              #
#                                                                                #
# ****************************************************************************** #
import sys
import os
""")

    # write the configuration file
    begmes("writing Payette_config.py",pre=sp)
    with open(Payette_config_file,"w") as f:
        f.write(intro)
        for key,value in payette_config.items():
            if isinstance(value,str):
                f.write('{0} = "{1}"\n'.format(key,value))
            else:
                f.write('{0} = {1}\n'.format(key,value))
                pass
            continue
        f.write("if Payette_Home not in sys.path: sys.path.insert(0,Payette_Home)")
        pass
    endmes("Payette_config.py written")

    # symlink the Payette_config file to the tests directories
    for dirnam, dirs, files in os.walk(Payette_Tests):
        if "__init__.py" not in files: continue
        config_base = os.path.basename(Payette_config_file)
        tests_config = os.path.join(dirnam,config_base)
        try: os.remove(tests_config)
        except: pass
        os.symlink(Payette_config_file,tests_config)
        continue
    return

def write_summary_to_screen():
    from os.path import dirname,realpath,join,splitext,islink
    def no_code_lines(fpath):
        nlines = 0
        if splitext(fpath)[1] not in Payette_Code_Exts: return nlines
        for line in open(fpath,"r").readlines():
            line = line.strip().split()
            if not line or line[0] == "#": continue
            nlines += 1
            continue
        return nlines

    rootd = dirname(dirname(realpath(__file__)))
    Payette_Dirs,Payette_Files = [], []
    Payette_Code_Exts = [".py",".pyf","",".F",".C",".f"]
    Payette_Exts = Payette_Code_Exts + [".inp",".tex",".pdf"]
    for dirnam,dirs,files in os.walk(rootd):
        if ".svn" in dirnam: continue
        Payette_Dirs.extend([join(dirnam,d) for d in dirs])
        Payette_Files.extend([join(dirnam,f) for f in files
                              if not islink(join(dirnam,f))
                              and splitext(f)[1] in Payette_Exts])
        continue
    No_Payette_Lines = sum([no_code_lines(f) for f in Payette_Files])
    No_Payette_Dirs = len(Payette_Dirs)
    No_Payette_Files = len(Payette_Files)
    No_Payette_Inp_Files = len([x for x in Payette_Files if x.endswith(".inp")])
    No_Payette_Py_Files = len([x for x in Payette_Files
                              if x.endswith(".py") or x.endswith(".pyf")])
    print("Payette: An Object Oriented Material Model Driver")
    print("Summary of Project:")
    print("\tNumber of files in project:         {0:d}".format(No_Payette_Files))
    print("\tNumber of diectories in project:    {0:d}".format(No_Payette_Dirs))
    print("\tNumber of input files in project:   {0:d}".format(No_Payette_Inp_Files))
    print("\tNumber of python files in project:  {0:d}".format(No_Payette_Py_Files))
    print("\tNumber of lines of code in project: {0:d}".format(No_Payette_Lines))
    return



def cleanPayette():
    """
    NAME
       cleanPayetteLibs

    PURPOSE
       clean Payette of any automatically generated files

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """
    from distutils import sysconfig
    from fnmatch import fnmatch

    soext = sysconfig.get_config_var("SO")
    toolsd = os.path.dirname(os.path.realpath(__file__))
    homed = os.path.dirname(toolsd)
    rootd = os.path.dirname(homed)

    pats_to_remove = ["*.pyc","*.pyo","Payette_config.py",
                      "Payette_installed_materials.py","*.{0}".format(soext),
                      "f2py","rebuildPayette","runPayette","testPayette",
                      "*.log","*.echo","*.prf","*.diff","*.xout","*.out"]
    for dirnam, dirs, files in os.walk(rootd):
        [os.remove(os.path.join(dirnam,f)) for f in files if
         any(fnmatch(f,p) for p in pats_to_remove)]
        continue
    return

if __name__ == "__main__":

    toolsd,buildf = os.path.split(os.path.realpath(__file__))
    if sys.argv[0] != buildf:
        # we want to enforce that buildPayette.py is executed from the Toolset
        # directory, except in special cases...

        if not any(["../Toolset/buildPayette.py" in sys.argv,
                    "-W" in sys.argv,
                    "-h" in sys.argv or "--help" in sys.argv,
                    os.path.realpath(os.getcwd()) == this_d
                    ]):
            sys.exit("buildPayette must be executed from {0}".format(toolsd))
            pass
        pass

    build_Payette = buildPayette(len(sys.argv[1:]),sys.argv[1:])

    error, warn = 0, 0
    if build_Payette == 0:
        loginf("buildPayette succeeded\n")

    elif build_Payette == 55:
        warn += 1
        logwrn("buildPayette failed to build one or more material libraries\n")

    elif build_Payette == 75:
        error += 1
        logerr("buildPayette failed due to an unknown error\n")

    else:
        logerr("buildPayette failed\n")
        error += 1

    if createx and not error:
        msg = ("the {0} executable scripts were built in:\n\t{1}"
               .format(", ".join(Payette_Built_Executables),Payette_Toolset))
        loginf(msg)
        pass

    if not error and not warn: logmes("Enjoy Payette!")
    elif warn: logmes("You've been warned, tread lightly!")
    else: logmes("Better luck next time!")

