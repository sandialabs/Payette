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

import os

from Source.Payette_utils import *
import Source.Payette_container as pcntnr
try: from Source.Materials.Payette_installed_materials import *
except ImportError:
    sys.exit("buildPayette must be run to create "
             "Source/Materials/Payette_installed_materials.py")


class Options(object):
    def __init__(self):
        self.debug = False
        self.verbosity = 4
        self.keep = False
        self.sqa = False
        self.write_vandd_table = False
        self.use_table = False
        self.debug = False
        self.testrestart = False
        self.norestart = True
        self.nowriteprops = True

class PayetteBarf(object):

    """
        general Payette barf class. converts barf file to Payette class object
        and runs the driver
    """

    def __init__(self,barf_file):

        self.barf_file = barf_file
        self.lines = open(self.barf_file,"r").readlines()

        self._get_barf_info()

        self.inform("running barf file: {0}".format(self.barf_file))
        self.inform("constitutive model: {0}".format(self.constitutive_model))
        self.inform("constitutive model version: {0}".format(self.version))
        self.inform("constitutive model revision: {0}".format(self.revision))
        self.inform("barf message: {0}".format(self.message))

        self._convert_to_Payette()

        # parse the input
        self.payette_input_dict = readUserInput(self.payette_input.split("\n"))

        # dummpy options to send to Payette
        opts = Options()

        # instantiate the Payette object
        for key, val in self.payette_input_dict.items():
            the_model = pcntnr.Payette(key,val,opts)
            break

        sys.exit("here in Payette_barf, not done yet")
        # the model has been set up, now advance the stress and state variables
        # to where they need to be based on barf file
        the_model.matdat.advanceData("extra variables",self.extra)
        the_model.matdat.advanceData("stress",self.stress)
        the_model.matdat.advanceData("derived constants")
        the_model.simdat.advanceData("rate of deformation",self.strain_rate)

        pass

    def _get_barf_info(self):

        """
            read the first line of the barf file and get info

            the header of all barf files are of the form:

            model version X.x
            svn revsion XXX
            barf message
            ----------------------------------------
            x = nblk   y=iblk
            xx = ninsv
            xxx = nprop
            xxxxxx.E+xx = dt
            ################ property array
            .
            .
            ################ derived constants array
            .
            .
            ################ stress array
            .
            .
            ################ strain rate
            .
            .
            ################ state variable array
            .
            .
        """

        iam = "PayetteBarf._get_barf_info(self)"

        # get the constitutive model
        cmod,tmp,version = self.lines[0].lower().split()
        if cmod not in Payette_Installed_Materials:
            error(iam,"constitutive model {0} not installed".format(cmod))
            pass

        self.constitutive_model = cmod
        self.version = version
        self.revision = self.lines[1].split()[2]
        self.message = self.lines[2].strip()
        self.name = self.message.replace(" ","_")
        self.time_step = None

        # instantiate the constitutive model class
        cmod = Payette_Constitutive_Models[cmod]["class name"]()

        in_props, props = 0, []
        in_derived_consts, self.derived_consts = 0, []
        in_stress, self.stress = 0, []
        in_strain_rate, self.strain_rate = 0, []
        in_extra, self.extra = 0, []

        idx, ixv = 0, 0
        for line in self.lines:

            line = line.strip().split()
            if not line: continue

            if "dt" in line:
                self.time_step = line[0]
                continue

            if "#####" in line[0]:
                if "property" in line[1]:
                    in_props = 1
                    continue
                elif "derived" in line[1]:
                    in_props = 0
                    in_derived_consts = 1
                    continue
                elif "stress" in line[1]:
                    in_derived_consts = 0
                    in_stress = 1
                    continue
                elif "strain" in line[1]:
                    in_stress = 0
                    in_strain_rate = 1
                    continue
                elif "state" in line[1]:
                    in_strain_rate = 0
                    in_extra = 1
                    continue
                pass
            elif "relevant" in line[0]:
                break

            if in_props:
                nam = cmod.parameter_table_idx_map[idx]
                val = line[1]
                props.append("{0} = {1}".format(nam,val))
                idx += 1
                if idx == 59: eos = val > 0

            elif in_derived_consts:
                self.derived_consts.append(line[1])

            elif in_stress:
                self.stress.append(line[0])

            elif in_strain_rate:
                self.strain_rate.append(line[0])

            elif in_extra:
                self.extra.append(line[1])
                if eos and ixv == 44:
                    bmod = line[1]
                if eos and ixv == 45:
                    smod = line[1]
                ixv += 1

            else:
                pass

            continue

        if len(self.strain_rate) != 6:
            ls = len(self.strain_rate)
            self.error(iam,"len(strain_rate) = {0} != 6".format(ls))
            pass

        if len(self.stress) != 6:
            ls = len(self.stress)
            self.error(iam,"len(stress) = {0} != 6".format(ls))
            pass

        if self.time_step == None:
            self.error(iam,"time step not found")
            pass

        if eos:
            for iprop, prop in enumerate(props):
                prop = prop.split()
                if "b0" in prop[0].lower():
                    props[iprop] = "B0 = {0}".format(bmod)
                    continue
                if "g0" in prop[0].lower():
                    props[iprop] = "G0 = {0}".format(smod)
                    break
                continue
            pass

        self.props = "\n".join(props)
        pass

    def _convert_to_Payette(self):

        self.payette_input = """
begin simulation {0}
begin boundary
begin legs
0 0 0 111111 0 0 0 0 0 0.
1 {1} 1 111111 {2} {3} {4} {5} {6} {7}
end legs
end boundary
begin material
constitutive model {8}
{9}
end material
end simulation
""".format(self.name, self.time_step,
           self.strain_rate[0], self.strain_rate[1], self.strain_rate[2],
           self.strain_rate[3], self.strain_rate[4], self.strain_rate[5],
           self.constitutive_model, self.props)

        return

    def error(self,caller,message):
        sys.exit("ERROR: {0} [reported by: {1}]".format(message,caller))

    def inform(self,message):
        print("INFO: {0}".format(message))
