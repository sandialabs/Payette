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

import Source.__config__ as cfg
from Source.Payette_utils import (
    report_and_raise_error, log_warning, log_message, parse_token)
from Source.Payette_tensor import SYM_MAP, iso, dev, mag, ddp
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
from Toolset.elastic_conversion import compute_elastic_constants

try:
    import plastic as mtllib
    imported = True
except:
    imported = False

class Plastic(ConstitutiveModelPrototype):
    """ Plasticity model. """

    def __init__(self, control_file, *args, **kwargs):
        super(Plastic, self).__init__(control_file, *args, **kwargs)

        # register parameters
        self.imported = imported
        self.register_parameters_from_control_file()

        pass

    # public methods
    def set_up(self, matdat):

        # parse parameters
        self.parse_parameters()

        # check parameters
        a = [np.array(self.ui0)]
        if cfg.F2PY["callback"]:
            a.extend([report_and_raise_error, log_message])
        self.ui = mtllib.plastic_chk(*a)

        a = [np.array(self.ui)]
        if cfg.F2PY["callback"]:
            a.extend([report_and_raise_error, log_message])
        nxtra, namea, keya, xtra, iadvct = mtllib.plastic_rxv(*a)
        names = parse_token(nxtra, namea)
        keys = parse_token(nxtra, keya)

        self.nsv = nxtra
        self.bulk_modulus, self.shear_modulus = self.ui[0], self.ui[1]

        # register extra variables
        matdat.register_xtra_vars(nxtra, names, keys, xtra)

        return

    def update_state(self, simdat, matdat):
        """
           update the material state based on current state and strain increment
        """
        # get passed arguments
        dt = simdat.get_data("time step")
        d = matdat.get_data("rate of deformation")
        sigold = matdat.get_data("stress")
        xtra = matdat.get_data("extra variables")

        a = [1, self.nsv, dt, self.ui, sigold, d, xtra]
        if cfg.F2PY["callback"]:
            a.extend([report_and_raise_error, log_message])
        sig, xtra = mtllib.plastic_calc(*a)

        # store updated data
        matdat.store_data("extra variables", xtra)
        matdat.store_data("stress", sig)
        return
