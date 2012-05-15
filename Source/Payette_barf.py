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

import Payette_config as pc
import Source.Payette_utils as pu
import Source.Payette_driver as pdrvr
import Source.Payette_container as pcntnr

def error(caller, message):
    """ Give error message and exit. """

    sys.exit("ERROR: {0} [reported by: {1}]".format(message, caller))


def inform(message):
    """ Write info to user. """

    sys.stdout.write("INFO: {0}\n".format(message))
    return


class PayetteBarf(object):
    """ General Payette barf class. Converts barf file to Payette class object
    and runs the driver.

    """

    def __init__(self, barf_file, opts):

        inform("running barf file: {0}".format(barf_file))

        self.barf = {}
        self.barf["lines"] = []
        for line in open(barf_file, "r").readlines():
            if not line.split():
                continue
            self.barf["lines"].append(line)
            continue

        # get the barf info
        self.get_barf_info()
        self.read_barf_file()

        # write message to screen
        inform("constitutive model: {0}"
               .format(self.barf["constitutive model"]))
        inform("constitutive model version: {0}"
               .format(self.barf["model version"]))
        inform("constitutive model revision: {0}"
               .format(self.barf["model revision"]))
        inform("barf message: {0}"
               .format(self.barf["barf message"]))

        # convert the barf file to a payette input
        self.payette_input = self._convert_to_payette()

        # parse the input
        input_dict = pu.read_input(self.payette_input)

        # instantiate the Payette object
        for key, val in input_dict.items():
            the_model = pcntnr.Payette(key, val, opts)
            break

        # the model has been set up, now advance the stress and state variables
        # to where they need to be based on barf file
        simdat = the_model.simulation_data()
        material = the_model.material
        matdat = material.material_data()
        material.constitutive_model.dc = self.barf["derived constsants"]
        matdat.advance_data("stress", self.barf["stress"])
        simdat.advance_data("rate of deformation", self.barf["strain rate"])
        matdat.advance_data("extra variables", self.barf["extra variables"])

        pdrvr.solid_driver(the_model)

        pass

    def get_barf_info(self):
        """Read the first line of the barf file and get info. """

        iam = "PayetteBarf.get_barf_info(self)"

        # get the constitutive model
        with open(pc.PC_MTLS_FILE, "rb") as fobj:
            constitutive_models = pickle.load(fobj)
        tmp = self.barf["lines"][0].lower().split()
        cmod, version = tmp[0], tmp[2]
        constitutive_model = constitutive_models.get(cmod)
        if constitutive_model is None:
            error(iam, "constitutive model {0} not installed".format(cmod))

        message = self.barf["lines"][2].strip()

        # get the material's constitutive model object
        py_mod = constitutive_model["module"]
        py_path = [os.path.dirname(constitutive_model["file"])]
        cls_nam = constitutive_model["class name"]
        fobj, pathname, description = imp.find_module(py_mod, py_path)
        py_module = imp.load_module(py_mod, fobj, pathname, description)
        fobj.close()
        cmod_obj = getattr(py_module, cls_nam)

        self.barf["constitutive model instance"] = cmod_obj
        self.barf["constitutive model"] = cmod
        self.barf["model version"] = version
        self.barf["model revision"] = self.barf["lines"][1].split()[2]
        self.barf["barf message"] = self.barf["lines"][2].strip()
        self.barf["name"] = message.replace(" ", "_")
        self.barf["time step"] = None

        return

    def read_barf_file(self):
        """ Read the barf file. """

        iam = "PayetteBarf.read_barf_file(self)"

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
            error(iam, "len(strain_rate) = {0} != 6".format(len(strain_rate)))
        else:
            self.barf["strain rate"] = np.array(strain_rate)

        if len(stress) != 6:
            error(iam, "len(stress) = {0} != 6".format(len(stress)))
        else:
            self.barf["stress"] = np.array(stress)

        if not dtime:
            error(iam, "invalid timestep {0}".format(dtime))
        else:
            self.barf["time step"] = dtime

        self.barf["parameters"] = "\n".join(parameters)
        self.barf["extra variables"] = np.array(extra_variables)
        self.barf["derived constsants"] = np.array(derived_consts)

        return

    def _convert_to_payette(self):
        """ Convert a barf file to Payette input. """

        name = self.barf["name"]
        dtime = self.barf["time step"]
        rod = self.barf["strain rate"]
        cmod = self.barf["constitutive model"]
        params = self.barf["parameters"]
        input_file = """
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
           cmod, params)

        return input_file.split("\n")

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

            if idx == 59:
                self.barf["using eos"] = val > 0
                val = 0

            props.append("{0} = {1}".format(nam, val))
            continue

        return props
