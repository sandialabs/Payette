#!/usr/bin/env python
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
import os
import optparse
import tempfile


###############################################################################
#                               TESTS_COMMON.PY                               #
###############################################################################
tests_common = (
    """
import os,sys

payette_home = os.getenv("PAYETTE_HOME")
assert payette_home, "PAYETTE_HOME environment variable not found"

tests_d = os.path.join(payette_home,"Tests")
source_d = os.path.join(payette_home,'Source')
tools_d = os.path.join(payette_home,'Toolset')

if "PAYETTE_DBG" in os.environ:
    print("__file__:    {0}".format(os.path.realpath(__file__)))
    print("tests_d:     {0}".format(tests_d))
    print("source_d:    {0}".format(source_d))
    print("tools_d:     {0}".format(tools_d))
    print("")

sys.path.append(payette_home)
sys.path.append(tools_d)
sys.path.append(source_d)
sys.path.append(tests_d)
""")

###############################################################################
#                                BENCHMARK.PY                                 #
###############################################################################
benchmark_py = (
    """
#!/usr/bin/env python
import os
import subprocess
from tests_common import *

Payette_test = True
fdir = os.path.dirname(os.path.realpath(__file__))
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
infile = '%s.inp'%(os.path.join(fdir,name))
outfile = '%s.out'%(name)
keywords = ['auto-generated','fast','verification']
executable = os.path.join(tools_d,'payette')
runcommand = [executable,'--no-writeprops','--no-restart','--proportional',infile]
owner = 'auto-generated'
date = 'today'
description = "Auto-generated simulation."

def performCalcs():
    with open('%s.echo'%(name),'w') as f:
        run = subprocess.Popen(runcommand,stdout=f,stderr=subprocess.STDOUT)
        run.wait()

        if check:
            self.check_setup()

        pass
    return run.returncode

def runTest():
    perform_calcs = performCalcs()
    if perform_calcs != 0: return 2
    return 0
if __name__=="__main__":
    print(runTest())
""")


###############################################################################
#                          RandomPayetteBoundary.py                           #
###############################################################################
# What's shown here in the next 50 lines or so was originally a stand-alone
# file (just uncomment the "if __name__..." part).
def FormatPayetteLeg(i, t, n, e11, e22, e33, e12, e23, e13):
    fmt = lambda x: "{0:.6e}".format(x)
    msg = ("      {0}, {1}, {2}, 222222, ".format(i, fmt(t), n) +
           " {0}, {1}, {2},".format(fmt(e11), fmt(e22), fmt(e33)) +
           " {0}, {1}, {2}\n".format(fmt(e12), fmt(e23), fmt(e13)))
    return msg


def RandomPayetteBoundary():
    import math
    import random

    # Set the Seth-Hill coefficient for strain definitions
    kappa = 0
    # Set the number of legs
    numlegs = 4
    # Set the number of timesteps per leg
    n = 200
    # Set the range (small,large) for the timestep size and
    # strain component size as a log10 value.
    log10_timestep_range = [-6, 0]
    log10_strain_range = [-6, 0]

    # Begin boundary header
    msg = ("  begin boundary\n" +
           "    kappa = {0}\n".format(kappa) +
           "    tfac  = 1\n" +
           "    amplitude = 1\n" +
           "    begin legs\n")

    # Leg setup.
    t, i = 0.0, 0
    randstrain = lambda: 10 ** random.uniform(*log10_strain_range)
    randtimestep = lambda: 10 ** random.uniform(*log10_timestep_range)

    # Create the legs
    msg += FormatPayetteLeg(i, t, 0, 0, 0, 0, 0, 0, 0)
    for idx in range(0, numlegs):
        i += 1
        t += randtimestep()
        msg += FormatPayetteLeg(
            i, t, n, randstrain(), randstrain(), randstrain(),
            randstrain(), randstrain(), randstrain())

    # Finish the boudary section
    msg += "    end legs\n  end boundary\n"
    return msg

# if __name__=="__main__":
#    print(RandomPayetteBoundary())
###############################################################################
#                        END RandomPayetteBoundary.py                         #
###############################################################################


class ModelSet:
    def __init__(self, name, input_d, output_d):
        self.name = name
        self.input_d = os.path.realpath(input_d)
        self.output_d = os.path.realpath(output_d)
        self.materials = []

        if not os.path.isdir(self.input_d):
            sys.exit("Cannot create ModelSet() type.\n" +
                     "Input directory does not exist:\n" +
                     "({0})".format(self.input_d))

        if not os.path.isdir(self.output_d):
            os.makedirs(self.output_d)

        if not os.path.isdir(self.output_d):
            sys.exit("Cannot create ModelSet() type.\n" +
                     "Output directory does not exist:\n" +
                     "({0})".format(self.output_d))

        # Get all the files (complete path) in the input_d
        mat_files = [os.path.join(self.input_d, x)
                     for x in os.listdir(self.input_d)]
        mat_files = [x for x in mat_files if os.path.isfile(x)]
        # Remove files bigger than 100 KB (why would an input file be larger?)
        mat_files = [x for x in mat_files if os.path.getsize(x) < 1024 * 100]
        # Remove files that don't end in '.dat'
        mat_files = [x for x in mat_files if x.upper().endswith(".DAT")]
        self.materials = mat_files


def run(input_opts):
    # This will be the default material directory.
    file_d = os.path.dirname(os.path.realpath(__file__))
    material_d = os.path.join(file_d, "TestMaterials")
    output_d = os.path.join(file_d, "BatchOutput")

    # A bunch of Parser stuff
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage, version="%prog 1.0")

    parser.add_option('-n', '--num-realizations',
                      dest="NUM_REALIZATIONS",
                      action='store',
                      type=int,
                      default=10,
                      help="Set the number of realizations per "
                           "material input file [default: %default].")

    parser.add_option('-d', '--input-directory',
                      dest="MATERIAL_DIRECTORY",
                      action='store',
                      default=material_d,
                      help="Change the directory in which to look for "
                           "material input files [default: %default].")

    parser.add_option('-o', '--output-directory',
                      dest="OUTPUT_DIRECTORY",
                      action='store',
                      default=output_d,
                      help="Change the output directory where the random inputs"
                           "will be put [default: %default].")

    (opts, args) = parser.parse_args(input_opts)

    ###########################################################################
    #                 MAKE SURE THE USER INPUTS ARE SANE                      #
    ###########################################################################
    if opts.NUM_REALIZATIONS < 0:
        sys.exit("Cannot create a negative number of realizations.\n" +
                 "Given: {0}".format(opts.NUM_REALIZATIONS))

    opts.MATERIAL_DIRECTORY = os.path.realpath(opts.MATERIAL_DIRECTORY)
    if not os.path.isdir(opts.MATERIAL_DIRECTORY):
        sys.exit("The material directory cannot be found at:\n" +
                 "{0}".format(opts.MATERIAL_DIRECTORY))

    opts.OUTPUT_DIRECTORY = os.path.realpath(opts.OUTPUT_DIRECTORY)
    if not os.path.isdir(opts.OUTPUT_DIRECTORY):
        print("\nOutput directory not found.\n" +
              "({0})\n".format(opts.OUTPUT_DIRECTORY) +
              "Creating it...\n")
        os.mkdir(opts.OUTPUT_DIRECTORY)

    if not os.path.isdir(opts.OUTPUT_DIRECTORY):
        sys.exit("\nOutput directory could not be created." +
                 "\n({0})".format(opts.OUTPUT_DIRECTORY))

    ###########################################################################
    #                     WALK THE MATERIAL DIRECTORY                         #
    ###########################################################################

    # The models to be used are whatever the names of the directories in
    # the material directory.
    material_models = [x for x in os.listdir(opts.MATERIAL_DIRECTORY)
                       if os.path.isdir(os.path.join(opts.MATERIAL_DIRECTORY, x))]

    material_model_db = []
    for model_name in material_models:
        material_model_db.append(
            ModelSet(model_name,
                     os.path.join(
                     opts.MATERIAL_DIRECTORY, model_name),
                     os.path.join(opts.OUTPUT_DIRECTORY, model_name)))

    ###########################################################################
    #                      GENERATE THE REALIZATIONS                          #
    ###########################################################################

    for mat_model in material_model_db:
        TESTSCOMMON = open(
            os.path.join(mat_model.output_d, "tests_common.py"), "w")
        TESTSCOMMON.write(tests_common)
        TESTSCOMMON.close()
        for material in mat_model.materials:
            # The simulation prefix is just the model and the material file
            # name put together with an underscore.
            sim_prefix = mat_model.name + "_" + \
                os.path.splitext(os.path.basename(material))[0]
            materialtxt = open(material, "r").read()

            for incr in range(0, opts.NUM_REALIZATIONS):
                # Create the temporary file and open it
                SIM_FNO, SIM_N = tempfile.mkstemp(suffix=".inp",
                                                  prefix=sim_prefix + "_",
                                                  dir=mat_model.output_d)
                SIM_F = os.fdopen(SIM_FNO, "w")

                sim_name = os.path.splitext(os.path.basename(SIM_N))[0]
                inp = ("begin simulation {0}\n".format(sim_name) +
                       "  title {0}\n".format(sim_name) +
                       "  begin material\n" +
                       "    constitutive model {0}\n".format(mat_model.name) +
                       materialtxt + "\n" +
                       "  end material\n")

                inp += RandomPayetteBoundary()
                inp += "end simulation"

                print("Test written: {0}".format(SIM_N))
                SIM_F.write(inp)
                SIM_F.close()
                BENCHMARK_F = open(os.path.splitext(SIM_N)[0] + ".py", "w")
                BENCHMARK_F.write(benchmark_py)
                BENCHMARK_F.close()

###############################################################################
#                                    MAIN                                     #
###############################################################################
if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    run(sys.argv[1:])
