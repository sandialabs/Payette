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
import sys
import os
import imp
import numpy as np

import Payette_config as pc
import Source.Payette_utils as pu
from Source.Payette_data_container import DataContainer

class Material:
    '''
    CLASS NAME
       Material

    PURPOSE
       Material is the container for material related data for a Payette object. It
       is instantiated by the Payette object.

    DATA
       ConstitutiveModel: Material's constitutive model

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self, model_name, user_params, *args, **kwargs):

        iam = "Material.__init__"

        # get the material's constitutive model object
        cmod = pu.get_constitutive_model_object(model_name)

        # instantiate the constiutive model
        self.constitutive_model = cmod(*args, **kwargs)

        # check if the model was successfully imported
        if not self.constitutive_model.imported:
            msg = """Error importing the {0} material model.
If the material model is a fortran extension library, it probably was not
built correctly.  To check, go to

{1}

and try importing the material's extension module directly in a python
session.  If it does not import, you will need to rebuild the extension module.
If rebuilding Payette does not fix the problem, please contact the
Payette developers.""".format(self.constitutive_model.name, pc.PC_MTLS_LIBRARY)
            pu.reportError(iam, msg)

        self.eos_model = self.constitutive_model.eos_model

        # initialize material data container
        self.matdat = DataContainer(self.constitutive_model.name)
        self.extra_vars_registered = False

        # register default data
        if self.eos_model:
            register_default_data = self.register_default_eos_data
        else:
            register_default_data = self.register_default_data

        register_default_data()

        # set up the constitutive model
        self.constitutive_model._parse_user_params(user_params)
        self.constitutive_model.set_up(self.matdat)
        self.constitutive_model.finish_setup(self.matdat)
        self.constitutive_model.initialize_state(self.matdat)

        param_table = [None] * self.constitutive_model.nprop
        for key, dic in self.constitutive_model.parameter_table.items():
            idx = dic['ui pos']
            val1 = self.constitutive_model.ui0[idx]
            val2 = self.constitutive_model.ui[idx]
            param_table[idx] = {"name":key,
                                "initial value":val1,
                                "adjusted value":val2}
            continue

        # register param table
        self.matdat.register_option("parameter table", param_table)

        pass

    def register_default_data(self):
        """Register the default data for the material """

        # register obligatory data

        # plotable data
        self.matdat.register_data("stress", "SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="sig")
        self.matdat.register_data("stress rate", "SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="dsigdt")
        self.matdat.register_data("strain","SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="strain")
        self.matdat.register_data("deformation gradient","Tensor",
                                  init_val="Identity",
                                  plot_key="F")
        self.matdat.register_data("rate of deformation","SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="d")
        self.matdat.register_data("vorticity","Tensor",
                                  init_val=np.zeros(9),
                                  plot_key="w")
        self.matdat.register_data("equivalent strain","Scalar",
                                  init_val=0.,
                                  plot_key="eqveps")
        self.matdat.register_data("pressure","Scalar",
                                  init_val=0.,
                                  plot_key="pressure")

        if self.constitutive_model.electric_field_model:
            # electric field model data
            self.matdat.register_data("permittivity","SymTensor",
                                     init_val=np.zeros(6),
                                     plot_key="permtv")
            self.matdat.register_data("electric field","Vector",
                                     init_val=np.zeros(3),
                                     plot_key="efield")

        # non-plotable data
        self.matdat.register_data("prescribed stress", "Array",
                                 init_val=np.zeros(6))
        self.matdat.register_data("prescribed stress components",
                                 "Integer Array",
                                 init_val=np.zeros(6,dtype=int))
        self.matdat.register_data("prescribed strain","SymTensor",
                                 init_val=np.zeros(6))
        self.matdat.register_data("strain rate","SymTensor",
                                 init_val=np.zeros(6))
        self.matdat.register_data("prescribed deformation gradient","Tensor",
                                 init_val=np.zeros(9))
        self.matdat.register_data("deformation gradient rate","Tensor",
                                 init_val=np.zeros(9))
        self.matdat.register_data("rotation","Tensor",
                                 init_val="Identity")
        self.matdat.register_data("rotation rate","Tensor",
                                 init_val=np.zeros(9))
        return

    def register_default_eos_data(self):
        self.matdat.register_data("density", "Scalar",
                                 init_val=0.,
                                 plot_key="rho")
        self.matdat.register_data("temperature", "Scalar",
                                 init_val=0.,
                                 plot_key="temp")
        self.matdat.register_data("energy", "Scalar",
                                 init_val=0.,
                                 plot_key="enrg")
        self.matdat.register_data("pressure", "Scalar",
                                 init_val=0.,
                                 plot_key="pres")
        return

    def material_data(self):
        return self.matdat

    def update_state(self, simdat, matdat):
        return self.constitutive_model.update_state(simdat, matdat)

    def jacobian(self, simdat, matdat):
        return self.constitutive_model.jacobian(simdat, matdat)

