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

"""Process and run a barf file.
Barf files are dumps from a material model of the form:

    model version X.x
    svn revsion xxx
    barf message
    ----------------------------------------
    x = nblk   y=iblk
    xx = nxtra
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
import sys
import numpy as np
import pickle
import re

import Source.Payette_utils as pu
import Source.Payette_driver as pdrvr
import Source.Payette_container as pcntnr
import Source.Payette_model_index as pmi
import Source.Payette_input_parser as pip
from Source.Payette_container import PayetteError as PayetteError


class PayetteBarf(object):
    """ General Payette barf class. Converts barf file to Payette class object
    and runs the driver.

    """

    def __init__(self, barf_file):

        self.ipfailstat = 59

        pu.log_message("running barf file: {0}".format(barf_file))

        self.barf = {}
        self.barf["lines"] = []
        lines = open(barf_file, "r").read()
        lines = re.sub(r"[0-9]\+100", "E+100", lines)
        for line in lines.split("\n"):
            line = " ".join(line.strip().split())
            if not line.split():
                continue
            self.barf["lines"].append(line)
            continue

        self.model_index = pmi.ModelIndex()

        # get the barf info
        self.get_barf_info()
        self.read_barf_file()

        # write message to screen
        pu.log_message("constitutive model: {0}"
                       .format(self.barf["constitutive model"]))
        pu.log_message("constitutive model version: {0}"
                       .format(self.barf["model version"]))
        pu.log_message("constitutive model revision: {0}"
                       .format(self.barf["model revision"]))
        pu.log_message("barf message: {0}"
                       .format(self.barf["barf message"]))

        # convert the barf file to a payette input
        self.payette_input = self._convert_to_payette()

        pass

    def run_job(self, *args, **kwargs):
        """Run the barf file"""

        self.model = pcntnr.Payette(self.payette_input)
        self.model.write_input = True

        # the model has been set up, now advance the stress and state variables
        # to where they need to be based on barf file
        simdat = self.model.simulation_data()
        material = self.model.material
        if self.test_barf:
            material.constitutive_model.ui[self.ipfailstat] = 25.
        matdat = material.material_data()
        material.constitutive_model.dc = self.barf["derived constants"]
        matdat.store("stress", self.barf["stress"])
        matdat.store("rate of deformation", self.barf["strain rate"])
        matdat.store("__xtra__", self.barf["__xtra__"])
        matdat.advance()

        try:
            pdrvr.solid_driver(self.model)
            self.message = "WARNING: Kayenta did not bomb"
            retcode = -1

        except PayetteError as e:
            # the code bombed, let's see if we reproduced the original
            msg = re.sub(
                r"ERROR|BOMBED|:", "", e.message).split("[")[0].strip()
            if msg == self.barf["barf message"]:
                self.message = (
                    "INFO: Kayenta bombed and barf message '{0}' "
                    "was successfully reproduced".format(msg))
                retcode = 0

            else:
                self.message = (
                    "WARNING: Kayenta bombed with '{0}' which differs "
                    "from '{1}'".format(msg, self.barf["barf message"]))
                retcode = 1
        sys.stdout.write(self.message + "\n")
        return {"retcode": retcode}

    def finish(self):
        self.model.finish()
        return 0

    def get_barf_info(self):
        """Read the first line of the barf file and get info. """

        # get the model name
        tmp = self.barf["lines"][0].lower().split()
        model_name, version = tmp[0], tmp[2]

        # get the constitutive model
        cmod = self.model_index.constitutive_model_object(model_name)
        cmod = cmod(self.model_index.control_file(model_name))

        message = self.barf["lines"][2].strip()

        self.barf["constitutive model instance"] = cmod
        self.barf["constitutive model"] = model_name
        self.barf["model version"] = version
        self.barf["model revision"] = self.barf["lines"][1].split()[2]
        self.barf["barf message"] = self.barf["lines"][2].strip()
        self.barf["name"] = message.replace(" ", "_").replace(".", "")
        self.barf["time step"] = None

        return

    def read_barf_file(self):
        """ Read the barf file. """

        dtime = eval(self.get_block("dt", block_delim=None)[0])
        parameters = self.parse_parameters(self.get_block("property"))
        derived_consts = self.get_block("derived", place=1)
        stress = self.get_block("stress", place=0)
        strain_rate = self.get_block("strain", place=0)
        extra_variables = self.get_block("state", place=1)

        if self.barf["using eos"]:
            # if the code used a host EOS, replace the bulk and shear moduli
            # with current values
            bmod = extra_variables[44]
            smod = extra_variables[45]
            for idx, param in enumerate(parameters):
                param = param.split()
                if "b0" in param[0].lower():
                    parameters[idx] = "B0 = {0}".format(bmod)
                    continue

                if "g0" in param[0].lower():
                    parameters[idx] = "G0 = {0}".format(smod)
                    break

                continue

        if len(strain_rate) != 6:
            pu.report_and_raise_error(
                "len(strain_rate) = {0} != 6".format(len(strain_rate)))
        else:
            self.barf["strain rate"] = np.array(strain_rate)

        if len(stress) != 6:
            pu.report_and_raise_error(
                "len(stress) = {0} != 6".format(len(stress)))
        else:
            self.barf["stress"] = np.array(stress)

        if not dtime:
            pu.report_and_raise_error(
                "invalid timestep {0}".format(dtime))
        else:
            self.barf["time step"] = dtime

        self.barf["parameters"] = "\n".join(parameters)
        self.barf["__xtra__"] = np.array(extra_variables)
        self.barf["derived constants"] = np.array(derived_consts)

        # see if this was a barf file produced by the test barf functionality
        failstat = re.search("(?i)failstat.*", self.barf["parameters"])
        s, e = failstat.start(), failstat.end()
        self.test_barf = bool(re.search("\.25", self.barf["parameters"][s:e]))

        return

    def _convert_to_payette(self):
        """ Convert a barf file to Payette input. """

        name = self.barf["name"]
        dtime = self.barf["time step"]
        rod = self.barf["strain rate"]
        cmod = self.barf["constitutive model"]
        params = self.barf["parameters"]
        return pip.parse_user_input("""\
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
""".format(name, dtime,
           rod[0], rod[1], rod[2], rod[3], rod[4], rod[5],
           cmod, params))[0]

    def get_block(self, name, block_delim="#####", place=None):
        """ Find the strain rate in the barf file """

        in_block = False
        block = []
        for line in self.barf["lines"]:

            line = line.split()

            if block_delim is None:
                # if no block delimeter, then the line is of the form
                #           val = name
                if name in line:
                    block = line
                    break

            else:

                if in_block and block_delim in line[0]:
                    break

                if "relevant" in line[0]:
                    break

                if block_delim in line[0] and name in line[1]:
                    in_block = True
                    continue

                if in_block:
                    if place is None:
                        block.append(line)

                    else:
                        val = line[place]
                        if line[place] == "NaN":
                            block.append(np.nan)
                        else:
                            block.append(eval(line[place]))

                continue

            continue

        return block

    def parse_parameters(self, params):
        """ parse the parameters """

        cmod = self.barf["constitutive model instance"]
        props = []
        for idx, val in params:

            idx = int(idx) - 1
            nam = cmod.parameter_table_idx_map[idx]

            if idx == 60:
                self.barf["using eos"] = val > 0
                val = "0."

            props.append("{0} = {1}".format(nam, val))
            continue

        return props
