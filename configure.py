"""
NAME
   configure.py

PURPOSE
   Configure Payette and write Payette_config.py
"""

from __future__ import print_function
import os,sys
import optparse
from distutils import sysconfig

# spacing used for logs to console
sp = "      "

configure = os.path.realpath(__file__)
rootd = os.path.dirname(configure)
toolsd = os.path.join(rootd,"Toolset")

def configurePayette(argc,argv):

    """ create and write configuration file """

    # *************************************************************************
    # -- command line option parsing
    usage = ("usage: python %prog [options]\nmust be executed from "
             "{0}".format(os.path.dirname(rootd)))
    parser = optparse.OptionParser(usage = usage, version = "%prog 1.0")
    parser.add_option("-o",
                      dest = "OPTIONS",
                      action = "append",
                      default = [],
                      help = "Options to build (accumulated) [default: %default]")
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
    parser.add_option("--no-callback",
                      dest = "NOCALLBACK",
                      action = "store_true",
                      default = False,
                      help = ("Compile with f2py callbacks from fortran routines"
                              " [default: %default]"))

    (opts,args) = parser.parse_args(argv)
    if len(args) > 0:
        parser.print_help()
        parser.error("{0} does not require arguments, only options"
                     .format(configure))
        pass

    # intro message
    loginf("Configuring Payette: An Object Oriented Material Model Driver\n",
           pre="\n")

    # python interpreter info
    Payette_pyint = os.path.realpath(sys.executable)

    # compatibility checks
    (major, minor, micro, releaselevel, serial) = sys.version_info
    if (major != 3 and major != 2) or (major == 2 and minor < 6):
        raise SystemExit("Payette requires Python >= 2.6\n")

    # clean up first
    cleanPayette()

    # configure Payette
    loginf("configuring Payette environment")

    # begin configuration
    errors = 0
    payette_config = {}
    payette_environ = {}

    def check_exists(itemnam,item):
        if not item:
            errors += 1
            logerr("{0} not found".format(itemnam))
            pass
        elif not os.path.isdir(item) and not os.path.isfile(item):
            errors += 1
            print("{0} not found".format(item))
        else: pass
        return

    # running with sage
    sage = True if "sage" in Payette_pyint else False

    # numpy check
    try: import numpy
    except:
        logerr("numpy not importable")
        errors += 1
        pass

    # scipy check
    try: import scipy
    except:
        logerr("scipy not importable")
        errors += 1
        pass

    # Payette home
    Payette_Root = rootd
    check_exists("Payette_Root",Payette_Root)

    # Root level directories
    Payette_Aux = os.path.join(Payette_Root,"Aux")
    Payette_Documents = os.path.join(Payette_Root,"Documents")
    Payette_Source = os.path.join(Payette_Root,"Source")
    Payette_Tests = os.path.join(Payette_Root,"Tests")
    Payette_Toolset = os.path.join(Payette_Root,"Toolset")
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
    Payette_Inputs = os.path.join(Payette_Root,"Aux/Inputs")
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
        pass

    # f2py call back
    if opts.NOCALLBACK: Payette_F2Py_Callback = False
    elif major == 3: Payette_F2Py_Callback = False
    elif sage: Payette_F2Py_Callback = False
    else: Payette_F2Py_Callback = True

    Payette_fcompiler = opts.FCOMPILER
    if not Payette_fcompiler:
        Payette_f77exec, Payette_f90exec = get_fortran_compiler(opts.F77EXEC,
                                                                opts.F90EXEC)
    else: Payette_f77exec, Payette_f90exec = None, None

    if errors:
        sys.exit("Payette_config.py: ERROR: fix previously trapped errors")
        pass

    # modify sys.path
    if Payette_Root not in sys.path: sys.path.insert(0,Payette_Root)

    # write our own f2py script
    Payette_f2py = write_f2py(Payette_pyint,Payette_Toolset)

    if Payette_fcompiler:
        f2pyopts = ["--fcompiler={0}".format(Payette_fcompiler)]
    else:
        f2pyopts = ["--f77exec={0}".format(Payette_f77exec),
                    "--f90exec={0}".format(Payette_f90exec)]
        pass

    # Payette executables
    Payette_Payette = os.path.join(Payette_Source,"Payette_main.py")
    Payette_runPayette = os.path.join(Payette_Toolset,"runPayette")
    Payette_buildPayette = os.path.join(Payette_Toolset,"buildPayette")
    Payette_cleanPayette = os.path.join(Payette_Toolset,"cleanPayette")
    Payette_extractPayette = os.path.join(Payette_Toolset,"extractPayette.py")
    check_exists("extractPayette",Payette_extractPayette)
    Payette_testPayette = os.path.join(Payette_Toolset,"testPayette")
    Payette_Built_Executables = {"runPayette":Payette_runPayette,
                                 "testPayette":Payette_testPayette,
                                 "buildPayette":Payette_buildPayette,
                                 "cleanPayette":Payette_cleanPayette,
                                 "f2py":Payette_f2py}
    Payette_Executables = {"extractPayette.py":Payette_extractPayette}
    for key,val in Payette_Built_Executables.items():
        Payette_Executables[key] = val
        continue

    # configuration files
    Payette_config_file = os.path.join(Payette_Root,"Payette_config.py")
    try: os.remove(Payette_config_file)
    except: pass
    try: os.remove(Payette_config_file + "c")
    except: pass

    # auxilary Payette environment variables
    Payette_Kayenta = os.getenv("PAYETTE_KAYENTA")
    Payette_AlegraNevada = os.getenv("PAYETTE_ALEGRANEVADA")
    Payette_nlopt = os.getenv("NLOPTLOC")


    # store all of the above information for writing to the Payette_config_file,
    # we waited tpo write it til now so that we would only write it if everything
    # was configured correctly.
    payette_config["Payette_pyint"] = Payette_pyint
    payette_config["Payette_Root"] = Payette_Root
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
    payette_config["Payette_f2py"] = Payette_f2py
    payette_config["Payette_runPayette"] = Payette_runPayette
    payette_config["Payette_testPayette"] = Payette_testPayette
    payette_config["Payette_buildPayette"] = Payette_buildPayette
    payette_config["Payette_cleanPayette"] = Payette_cleanPayette
    payette_config["Payette_Built_Executables"] = Payette_Built_Executables

    # set up the environment
    envs = ["MPLCONFIGDIR","PYTHONPATH","ECLDIR","GPDOCDIR","RHOME",
            "GP_DATA_DIR","PKG_CONFIG_PATH","PYTHONHOME","LD_LIBRARY_PATH",
            "LIBRARY_PATH","DYLD_LIBRARY_PATH","PATH","SINGULAR_EXECUTABLE",
            "SINGULARPATH"]
    if sage:
        for key,value in os.environ.items():
            if sage and ("sage" in value.lower() or "sage" in key.lower()):
                envs.append(key)
                pass
            continue
        pass

    for env in envs:
        try: payette_environ[env] = os.environ[env]
        except: pass
        continue

    add_python_path = False
    if "PYTHONPATH" in payette_environ:
        if Payette_Root not in payette_environ["PYTHONPATH"].split(os.pathsep):
            payette_environ["PYTHONPATH"] += os.pathsep + Payette_Root
            add_python_path = True
            pass
        pass
    else:
        payette_environ["PYTHONPATH"] = Payette_Root
        add_python_path = True
        pass



    # write the the configuration file
    begmes("writing Payette_config.py",pre=sp)
    with open(Payette_config_file,"w") as f:
        f.write(intro())
        for key,value in payette_config.items():
            f.write(dictfrmt(key,value) + "\n")
            continue
        f.write("if Payette_Root not in sys.path: sys.path.insert(0,Payette_Root)\n")
        for key,value in payette_environ.items():
            f.write('os.environ["{0}"] = "{1}"\n'.format(key,value))
            continue
        f.write("Payette_built = False")
        pass
    endmes("Payette_config.py written")

    try: import Payette_config
    except ImportError:
        if not os.path.isfile(os.path.dirname(configure),"Payette_config.py"):
            logerr("Payette_config.py not written")
            sys.exit(1)
        else: raise

    loginf("Payette environment configured")

    # remove the executables built by Payette
    for key,val in Payette_Built_Executables.items():
        try: os.remove(val)
        except: pass
        continue

    # create the runPayette, testPayette, and buildPayette
    # executables
    loginf("writing executable scripts")

    for item in [[Payette_runPayette,"run"],[Payette_testPayette,"test"],
                 [Payette_buildPayette,"build"],[Payette_cleanPayette,"clean"]]:
        script,typ = item
        begmes("writing %s"%(os.path.basename(script)),pre=sp)
        with open(script,"w") as f:
            f.write("#!/bin/sh -f\n")
            for key,value in payette_environ.items():
                f.write("export {0}={1}\n".format(key,value))
                continue
            if script == Payette_cleanPayette:
                f.write("{0} {1} {2} $* 2>&1\n"
                        .format(Payette_pyint,configure,typ))
            else:
                f.write("{0} {1} {2} $* 2>&1\n"
                        .format(Payette_pyint,Payette_Payette,typ))
                pass
            pass
        os.chmod(script,0o750)
        endmes("{0} script written".format(os.path.basename(script)))
        continue

    loginf("executable scripts written\n")

    if not errors and add_python_path:
        return -1
    return errors

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

def dictfrmt(key,value):
    if isinstance(value,str):
        return '{0} = "{1}"'.format(key,value)
    return '{0} = {1}'.format(key,value)

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


def write_f2py(Payette_pyint,destdir):
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
    f2py = os.path.join(destdir,"f2py")
    with open(f2py,"w") as f:
        for line in f2py_file: f.write(line)
        pass
    os.chmod(f2py,0o750)
    return f2py

def intro():
    return """# ****************************************************************************** #
#                                                                                #
# This file was generated automatically by the Payette. It contains important    #
# global Payette parameters that are configured at build time.                   #
#                                                                                #
# This file is intended to be imported by Payette using                          #
# "from Payette_config import *"                                                 #
#                                                                                #
# DO NOT EDIT THIS FILE. This entire file is regenerated automatically each time #
# configure.py is run. Any changes you make to this file will be overwritten.    #
#                                                                                #
# If changes are needed, please contact the Payette developers so that changes   #
# can be made to the build scripts.                                              #
#                                                                                #
# ****************************************************************************** #
import sys
import os
"""

def cleanPayette():
    """ clean Payette of any automatically generated files """
    from distutils import sysconfig
    from fnmatch import fnmatch

    soext = sysconfig.get_config_var("SO")

    pats_to_remove = ["*.pyc","*.pyo","Payette_config.py",
                      "Payette_installed_materials.py","*{0}".format(soext),
                      "*.log","*.echo","*.prf","*.diff","*.xout","*.out",
                      "*.math1","*.math2","*.props","*.vtable","*.dtable"]
    pats_to_remove.extend(["runPayette","testPayette","buildPayette",
                           "cleanPayette","f2py"])

    for dirnam, dirs, files in os.walk(rootd):
        if ".svn" in dirnam: continue
        [os.remove(os.path.join(dirnam,f)) for f in files if
         any(fnmatch(f,p) for p in pats_to_remove)]
        continue

    return


if __name__ == "__main__":

    if "clean" in sys.argv:
        loginf("cleaning Payette")
        cleanPayette()
        loginf("Payette cleaned")
        sys.exit(0)

    if sys.argv[0] != os.path.basename(__file__):
        sys.exit("configure.py must be executed from {0}".format(rootd))
        pass

    configure_Payette = configurePayette(len(sys.argv[1:]),sys.argv[1:])

    if configure_Payette <= 0:
        loginf("configure succeeded")

        if configure_Payette < 0:
            loginf("\n\n*** IMPORTANT ***\nAdd the Payette root directory to your "
                   "PYTHONPATH environment variable to complete the configuration")
            pass

        pass

    else:
        loginf("configure failed\n")

