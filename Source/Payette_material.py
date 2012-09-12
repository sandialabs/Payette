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

"""Main Payette class """

import numpy as np

import Payette_config as pc
import Source.Payette_utils as pu
import Source.Payette_model_index as pmi
from Source.Payette_data_container import DataContainer


class Material:
    """Material is the container for material related data for a Payette
    object. It is instantiated by the Payette object.

    Notes
    -----

    """

    def __init__(self, model_name, user_params, *args, **kwargs):

        # get the material's constitutive model object
        self.model_index = pmi.ModelIndex()
        control_file = self.model_index.control_file(model_name)
        cmod = self.model_index.constitutive_model_object(model_name)

        # instantiate the constiutive model
        self.constitutive_model = cmod(control_file, *args, **kwargs)

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
            pu.report_and_raise_error(msg)

        self.eos_model = self.constitutive_model.eos_model
        self.material_type = self.constitutive_model.material_type

        # density
        self._initial_density = 1.

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
        self.constitutive_model.parse_user_params(user_params)
        self.constitutive_model.set_up(self.matdat)
        self.constitutive_model.finish_setup(self.matdat)
        self.constitutive_model.initialize_state(self.matdat)

        param_table = [None] * self.constitutive_model.nprop
        for key, dic in self.constitutive_model.parameter_table.items():
            idx = dic['ui pos']
            val1 = self.constitutive_model.ui0[idx]
            val2 = self.constitutive_model.ui[idx]
            param_table[idx] = {"name": key,
                                "initial value": val1,
                                "adjusted value": val2}
            continue

        pass


    def register_default_data(self):
        """Register the default data for the material """

        # register obligatory data

        # plotable data
        self.matdat.register_data("stress", "SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="sig",
                                  units="PRESSURE_UNITS")
        self.matdat.register_data("stress rate", "SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="dsigdt",
                                  units="PRESSURE_UNITS_OVER_TIME_UNITS")
        self.matdat.register_data("strain", "SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="strain",
                                  units="NO_UNITS")
        self.matdat.register_data("deformation gradient", "Tensor",
                                  init_val="Identity",
                                  plot_key="F",
                                  units="NO_UNITS")
        self.matdat.register_data("rate of deformation", "SymTensor",
                                  init_val=np.zeros(6),
                                  plot_key="d",
                                  units="NO_UNITS_OVER_TIME_UNITS")
        self.matdat.register_data("vorticity", "Tensor",
                                  init_val=np.zeros(9),
                                  plot_key="w",
                                  units="VORTICITY_UNITS")
        self.matdat.register_data("equivalent strain", "Scalar",
                                  init_val=0.,
                                  plot_key="eqveps",
                                  units="NO_UNITS")
        self.matdat.register_data("pressure", "Scalar",
                                  init_val=0.,
                                  plot_key="pressure",
                                  units="PRESSURE_UNITS")

        if self.constitutive_model.electric_field_model:
            # electric field model data
            self.matdat.register_data("permittivity", "SymTensor",
                                     init_val=np.zeros(6),
                                     plot_key="permtv",
                                     units="PERMITTIVITY_UNITS")
            self.matdat.register_data("electric field", "Vector",
                                     init_val=np.zeros(3),
                                     plot_key="efield",
                                     units="ELECTRIC_FIELD_UNITS")

        # non-plotable data
        self.matdat.register_data("prescribed stress", "Array",
                                 init_val=np.zeros(6),
                                  units="PRESSURE_UNITS")
        self.matdat.register_data("prescribed stress components",
                                 "Integer Array",
                                 init_val=np.zeros(6, dtype=int),
                                 units="NO_UNITS")
        self.matdat.register_data("prescribed strain", "SymTensor",
                                 init_val=np.zeros(6),
                                 units="NO_UNITS")
        self.matdat.register_data("strain rate", "SymTensor",
                                 init_val=np.zeros(6),
                                 units="NO_UNITS_OVER_TIME_UNITS")
        self.matdat.register_data("prescribed deformation gradient", "Tensor",
                                 init_val=np.zeros(9),
                                 units="NO_UNITS")
        self.matdat.register_data("deformation gradient rate", "Tensor",
                                 init_val=np.zeros(9),
                                 units="NO_UNITS_OVER_TIME_UNITS")
        self.matdat.register_data("rotation", "Tensor",
                                 init_val="Identity",
                                 units="NO_UNITS")
        self.matdat.register_data("rotation rate", "Tensor",
                                 init_val=np.zeros(9),
                                 units="NO_UNITS_OVER_TIME_UNITS")
        return

    def register_default_eos_data(self):
        """Register default data to the matdat data container for eos jobs"""

        self.matdat.register_data("density", "Scalar",
                                 init_val=0.,
                                 plot_key="rho",
                                 units="DENSITY_UNITS")
        self.matdat.register_data("temperature", "Scalar",
                                 init_val=0.,
                                 plot_key="temp",
                                 units="TEMPERATURE_UNITS")
        self.matdat.register_data("energy", "Scalar",
                                 init_val=0.,
                                 plot_key="enrg",
                                 units="SPECIFIC_ENERGY_UNITS")
        self.matdat.register_data("pressure", "Scalar",
                                 init_val=0.,
                                 plot_key="pres",
                                 units="PRESSURE_UNITS")
        return

    def material_data(self):
        """return the material data container"""
        return self.matdat

    def update_state(self, simdat, matdat):
        """update the material state"""
        return self.constitutive_model.update_state(simdat, matdat)

    def jacobian(self, simdat, matdat):
        """return the material Jacobian matrix"""
        return self.constitutive_model.jacobian(simdat, matdat)

    def initial_density(self):
        return self.constitutive_model.initial_density()
