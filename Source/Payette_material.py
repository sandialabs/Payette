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

import copy
import sys
import logging

from Source.Payette_utils import *

import Source.Payette_installed_materials as pim
from Source.Payette_data_container import DataContainer

class Material:
    '''
    CLASS NAME
       Material

    PURPOSE
       Material is the container for material related data for a Payette object. It
       is instantiated by the Payette object.

    DATA
       constitutiveModel: Material's constitutive model

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self, constitutive_model_name, simdat, user_params, f_params ):

        iam = "Material.__init__(self, constitutive_model_name)"
        cmod = pim.PAYETTE_CONSTITUTIVE_MODELS[
            constitutive_model_name]["class name"]

        # instantiate the constiutive model
        self.constitutive_model = cmod()

        # check if the model was successfully imported
        if not self.constitutive_model.imported:
            msg = ("Error importing the {0} material model.\n"
                   "If the material model is a fortran extension library, "
                   "it probably was not built correctly.\nTo check, go to "
                   "{1}/Source/Materials/Library\nand try importing "
                   "the material's extension module directly in a python "
                   "session.\nIf it does not import, you will need to rebuild "
                   "the extension module.\n"
                   "If rebuilding Payette does not fix the problem, "
                   "please contact the Payette\ndevelopers."
                   .format(self.constitutive_model.name,PC_MTLS_LIBRARY))
            reportError(iam,msg)
            pass

        # register common parameters
        self.extra_vars_registered = False
        self.matdat = DataContainer(self.constitutive_model.name)

        # register obligatory variables
        self.matdat.registerData("stress","SymTensor",
                                 init_val = np.zeros(6),
                                 plot_key = "sig")
        self.matdat.registerData("stress rate", "SymTensor",
                                 init_val = np.zeros(6),
                                 plot_key = "dsigdt")
        self.matdat.registerData("failed","Boolean",
                                 init_val = False)

        # set up the constitutive model
        self.constitutive_model.setUp(simdat, self.matdat, user_params, f_params)
        self.constitutive_model.checkSetUp()
        self.constitutive_model.initializeState(simdat, self.matdat)

        param_table = [None]*self.constitutive_model.nprop
        for key, dic in self.constitutive_model.parameter_table.items():
            idx = dic['ui pos']
            val1 = self.constitutive_model.ui0[idx]
            val2 = self.constitutive_model.ui[idx]
            param_table[idx] = {"name":key,
                                "initial value":val1,
                                "adjusted value":val2}
            continue

        self.matdat.registerData("jacobian","Matrix",
                                 init_val = self.constitutive_model.J0)
        self.matdat.registerOption("parameter table",param_table)
        self.matdat.registerOption("electric field model",
                                   self.constitutive_model.electric_field_model)
        self.matdat.registerOption("multi level fail",
                                   self.constitutive_model.multi_level_fail_model)
        pass

    def materialData(self):
        return self.matdat

    def constitutiveModel(self):
        return self.constitutive_model

    def electricFieldModel(self):
        return self.constitutive_model.electric_field_model

    def updateState(self,simdat, matdat):
        return self.constitutive_model.updateState(simdat, matdat)

    def jacobian(self,simdat, matdat):
        return self.constitutive_model.jacobian(simdat,matdat)

