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
import numpy as np
from math import sqrt

from Source.Payette_utils import *
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK
from Source.Payette_tensor import I6, sym_map
from Toolset.elastic_conversion import compute_elastic_constants

try:
    import Source.Materials.Library.elastic as mtllib
    imported = True
except:
    imported = False
    pass

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
attributes = {
    "payette material": True,
    "name": "elastic",
    "aliases": ["hooke"],
    "fortran source": True,
    "build script": os.path.join(THIS_DIR, "Build_elastic.py"),
    "material type": ["mechanical"],
    "default material": True,
    }

w = np.array([1., 1., 1., 2., 2., 2.])

class Elastic(ConstitutiveModelPrototype):
    """ Elasticity model.

    Use MFLG = 0 for python implementation, MFLG = 1 for fortran

    """

    def __init__(self, *args, **kwargs):
        super(Elastic, self).__init__()
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]

        self.code = kwargs["code"]
        self.imported = True if self.code == "python" else imported

        # register parameters
        self.registerParameter("LAM", 0, aliases=[])
        self.registerParameter("G", 1, aliases=["MU", "G0", "SHMOD"])
        self.registerParameter("E", 2, aliases=["YOUNGS"])
        self.registerParameter("NU", 3, aliases=["POISSONS", "POISSONS RATIO"])
        self.registerParameter("K", 4, aliases=["B0", "BKMOD"])
        self.registerParameter("H", 5, aliases=["CONSTRAINED"])
        self.registerParameter("KO", 6, aliases=[])
        self.registerParameter("CL", 7, aliases=[])
        self.registerParameter("CT", 8, aliases=[])
        self.registerParameter("CO", 9, aliases=[])
        self.registerParameter("CR", 10, aliases=[])
        self.registerParameter("RHO", 11, aliases=["DENSITY"])
        self.nprop = len(self.parameter_table.keys())
        pass

    # public methods
    def setUp(self, matdat, user_params):
        iam = self.name + ".setUp"

        # parse parameters
        self.parseParameters(user_params)

        # the elastic model only needs the bulk and shear modulus, but the
        # user could have specified any one of the many elastic moduli.
        # Convert them and get just the bulk and shear modulus
        eui = compute_elastic_constants(*self.ui0[0:12])
        for key, val in eui.items():
            if key.upper() not in self.parameter_table:
                continue
            idx = self.parameter_table[key.upper()]["ui pos"]
            self.ui0[idx] = val

        # Payette wants ui to be the same length as ui0, but we don't want to
        # work with the entire ui, so we only pick out what we want
        mu, k = self.ui0[1], self.ui0[4]
        self.ui = self.ui0
        mui = np.array([k, mu])

        if self.code == "python":
            self.mui = self._py_set_up(mui)
        else:
            self.mui = self._fort_set_up(mui)

        self.bulk_modulus, self.shear_modulus = self.mui[0], self.mui[1]

        return

    def jacobian(self, simdat, matdat):
        v = matdat.getData("prescribed stress components")
        return self.J0[[[x] for x in v],v]

    def updateState(self, simdat, matdat):
        """
           update the material state based on current state and strain increment
        """
        # get passed arguments
        dt = simdat.getData("time step")
        d = matdat.getData("rate of deformation")
        sigold = matdat.getData("stress")

        if self.code == "python":
            sig = _py_update_state(self.mui, dt, d, sigold)

        else:
            a = [1, dt, self.mui, sigold, d, migError, migMessage]
            if not PC_F2PY_CALLBACK:
                a = a[:-2]
            sig = mtllib.elast_calc(*a)

        # store updated data
        matdat.storeData("stress", sig)

    def _py_set_up(self, mui):

        k, mu = mui

        if k <= 0.:
            reportError(iam, "Bulk modulus K must be positive")

        if mu <= 0.:
            reportError(iam, "Shear modulus MU must be positive")

        # poisson's ratio
        nu = (3. * k - 2 * mu) / (6 * k + 2 * mu)
        if nu < 0.:
            reportWarning(iam, "negative Poisson's ratio")

        ui = np.array([k, mu])

        return ui

    def _fort_set_up(self, mui):
        props = np.array(mui)
        a = [props, migError, migMessage]
        if not PC_F2PY_CALLBACK:
            a = a[:-2]
        ui = mtllib.elast_chk(*a)
        return ui


def _py_update_state(ui, dt, d, sigold):

    # strain increment
    de = d * dt

    # user properties
    k, mu = ui

    # useful constants
    twomu = 2. * mu
    alam = k - twomu / 3.

    # elastic stress update
    trde = np.sum(de * I6)
    sig = sigold + alam * trde * I6 + twomu * de
    return sig
