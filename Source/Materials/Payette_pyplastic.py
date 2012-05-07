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
from Payette_tensor import I6, sym_map

attributes = {
    "payette material": True,
    "name": "py plastic",
    "libname": "py_plastic",
    "aliases": [],
    "material type": ["mechanical"],
    "default material": True,
    }


w = np.array([1., 1., 1., 2., 2., 2.])

class PyPlastic(ConstitutiveModelPrototype):
    """ Plasticity model """

    def __init__(self):
        super(PyPlastic, self).__init__()
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = True

        # register parameters
        self.registerParameter("K", 0, aliases=["B0", "BKMOD"])
        self.registerParameter("MU", 1, aliases=["G", "G0", "SHMOD"])
        self.registerParameter("Y", 2, aliases=["A1", "YIELD STRENGTH"])
        self.registerParameter("A", 3, aliases=[])
        self.registerParameter("C", 4, aliases=[])
        self.registerParameter("M", 5, aliases=[])
        self.registerParameter("MFLG", 6, aliases=[])
        self.nprop = len(self.parameter_table.keys())
        pass

    # public methods
    def setUp(self, simdat, matdat, user_params, f_params):
        iam = self.name + ".setUp"

        # parse parameters
        self.parseParameters(user_params, f_params)

        k, mu, y, a, c, m, flg = self.ui0

        if k <= 0.:
            reportError(iam, "Bulk modulus K must be positive")

        if mu <= 0.:
            reportError(iam, "Shear modulus MU must be positive")

        if y <= 0.:
            reportError(iam, "Yield strength Y must be positive")

        if a < 0.:
            reportError(iam,
                        "Kinematic hardening modulus A must be non-negative")

        if c < 0.:
            reportError(iam,
                        "Isotropic hardening modulus C must be non-negative")

        if m < 0:
            reportError(iam,
                        "Isotropic hardening power M must be non-negative")
        elif m == 0:
            if c != 0:
                reportWarning(
                    iam,
                    "Isotropic hardening modulus C being set 0 because M = 0")
            c = 0.

        # poisson's ratio
        nu = (3. * k - 2 * mu) / (6 * k + 2 * mu)
        if nu < 0.:
            reportWarning(iam, "negative Poisson's ratio")

        self.bulk_modulus, self.shear_modulus = k, mu

        self.ui = np.array([k, mu, y, a, c, m, flg])

        # register state variables
        names, keys = [], []

        # equivalent plastic strain
        names.append("equivalent plastic strain")
        keys.append("eqps")

        # back stress
        for i in range(6):
            names.append("{0} component of back stress".format(sym_map[i]))
            keys.append("BSIG{0}".format(sym_map[i]))
            continue

        nxtra = len(keys)
        xtra = np.zeros(nxtra)
        matdat.registerExtraVariables(nxtra, names, keys, xtra)

        return

    def updateState(self, simdat, matdat):
        """
           update the material state based on current state and strain increment
        """
        # get passed arguments
        dt = simdat.getData("time step")
        de = simdat.getData("rate of deformation") * dt
        sigold = matdat.getData("stress")
        xtra = matdat.getData("extra variables")
        eqps = xtra[0]
        bstress = xtra[1:]

        # user properties
        k, mu, y0, a, c, m, flg = self.ui
        a = 2. / 3. * a

        # useful constants
        twomu = 2. * mu
        alam = k - twomu / 3.
        root23 = sqrt(2. / 3.)

        # elastic predictor
        trde = np.sum(de * I6)
        sig = sigold + alam * trde * I6 + twomu * de

        # elastic predictor relative to back stress - shifted stress
        s = sig - bstress

        # deviator of shifted stress and its magnitude
        smean = np.sum(s * I6) / 3.
        ds = s - smean * I6
        dsmag = sqrt(np.sum(w * ds * ds))

        # yield stress
        y = y0 if c == 0. else y0 + c * eqps ** (1 / m)
        radius = root23 * y

        # check yield
        facyld = 0. if dsmag - radius <= 0. else 1.
        dsmag = dsmag + (1. - facyld)

        # increment in plastic strain
        nddp = twomu
        h_kin = a
        h_iso = 0 if c == 0. else root23 * m * c * ((y - y0) / c) ** ((m-1)/m)
        dnom = nddp + h_kin + h_iso

        diff = dsmag - radius
        dlam = facyld * diff / dnom

        # equivalet plastic strain
        eqps += root23 * dlam

        # work with unit tensors
        dlam = dlam / dsmag

        # update back stress
        fac = a * dlam
        bstress = bstress + fac * ds

        # update stress
        fac = twomu * dlam
        sig = sig - fac * ds

        # store data
        xtra = np.concatenate((np.array([eqps]), bstress))
        matdat.storeData("extra variables", xtra)
        matdat.storeData("stress", sig)

        return
