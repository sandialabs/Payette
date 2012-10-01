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
import numpy as np

import Source.Payette_utils as pu
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
from Source.Payette_unit_manager import UnitManager as UnitManager

class IdealGas(ConstitutiveModelPrototype):
    def __init__(self, control_file, *args, **kwargs):
        super(IdealGas, self).__init__(
            control_file, *args, **kwargs)
        self.eos_model = True
        #self.code = "python"
        self.imported = True

        self.num_ui = 2

        # register parameters
        self.register_parameters_from_control_file()
        self.ui = np.zeros(self.num_ui)
        pass

    # Public methods
    def set_up(self,matdat):

        self.parse_parameters()
        self.ui = self.ui0

        # Variables already registered:
        #   density, temperature, energy, pressure
        matdat.register_data("soundspeed", "Scalar",
                             init_val = 0.,
                             plot_key = "SNDSPD",
                             units="VELOCITY_UNITS")
        matdat.register_data("dpdr", "Scalar",
                             init_val = 0.,
                             plot_key = "DPDR",
                             units="PRESSURE_UNITS_OVER_DENSITY_UNITS")
        matdat.register_data("dpdt", "Scalar",
                             init_val = 0.,
                             plot_key = "DPDT",
                             units="PRESSURE_UNITS_OVER_TEMPERATURE_UNITS")
        matdat.register_data("dedt", "Scalar",
                             init_val = 0.,
                             plot_key = "DEDT",
                             units="SPECIFIC_ENERGY_UNITS_OVER_TEMPERATURE_UNITS")
        matdat.register_data("dedr", "Scalar",
                             init_val = 0.,
                             plot_key = "DEDR",
                             units="SPECIFIC_ENERGY_UNITS_OVER_DENSITY_UNITS")
        pass

    def evaluate_eos(self, simdat, matdat, unit_system, rho=None, temp=None, enrg=None):
        """
          Evaluate the eos - rho and temp are in CGSEV

          By the end of this routine, the following variables should be
          updated and stored in matdat:
                  density, temperature, energy, pressure
        """
        M = self.ui[0]
        CV = self.ui[1]
        R = UnitManager.transform(8.3144621,
            "ENERGY_UNITS_OVER_TEMPERATURE_UNITS_OVER_DISCRETE_AMOUNT",
                                                     "SI", unit_system)

        if rho != None and temp != None:
            enrg = CV * R * temp
        elif rho != None and enrg != None:
            temp = enrg / CV / R
        else:
            pu.report_and_raise_error("evaluate_eos not used correctly.")

        P = R * temp * rho / M

        # make sure we store the "big three"
        matdat.store_data("density", rho)
        matdat.store_data("temperature", temp)
        matdat.store_data("energy", enrg)

        matdat.store_data("pressure", P)
        matdat.store_data("dpdr", R * temp / M)
        matdat.store_data("dpdt", R * rho / M)
        matdat.store_data("dedt", CV * R)
        matdat.store_data("dedr", CV * P * M / rho ** 2)
        matdat.store_data("soundspeed", (R * temp / M) ** 2)

        matdat.advance_all_data()
        return


    def update_state(self,simdat,matdat):
        """update the material state"""
        pu.report_and_raise_error("MGR EOS does not provide update_state")
        return
