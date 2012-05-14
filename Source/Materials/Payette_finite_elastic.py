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

import Source.Payette_utils as pu
import Source.Materials.tensors as mt
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK
from Source.Payette_tensor import delta
from Toolset.elastic_conversion import compute_elastic_constants

try:
    import Source.Materials.Library.finite_elastic as mtllib
    imported = True
except:
    imported = False
    pass

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
attributes = {
    "payette material": True,
    "name": "finite_elastic",
    "aliases": [],
    "fortran source": True,
    "build script": os.path.join(THIS_DIR, "Build_finite_elastic.py"),
    "material type": ["mechanical"],
    "default material": True,
    }


class FiniteElastic(ConstitutiveModelPrototype):
    """ Finite elasticity model.

    Computes the PK2 stress based on the Green-Lagrange strain and returns the
    Cauchy stress

    """

    def __init__(self, *args, **kwargs):
        super(FiniteElastic, self).__init__()
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = True

        self.code = kwargs["code"]
        self.imported = True if self.code == "python" else imported

        # register parameters
        self.register_parameter("LAM", 0, aliases=[])
        self.register_parameter("G", 1, aliases=["MU", "G0", "SHMOD"])
        self.register_parameter("E", 2, aliases=["YOUNGS"])
        self.register_parameter("NU", 3, aliases=["POISSONS", "POISSONS RATIO"])
        self.register_parameter("K", 4, aliases=["B0", "BKMOD"])
        self.register_parameter("H", 5, aliases=["CONSTRAINED"])
        self.register_parameter("KO", 6, aliases=[])
        self.register_parameter("CL", 7, aliases=[])
        self.register_parameter("CT", 8, aliases=[])
        self.register_parameter("CO", 9, aliases=[])
        self.register_parameter("CR", 10, aliases=[])
        self.register_parameter("RHO", 11, aliases=["DENSITY"])

        self.nprop = len(self.parameter_table.keys())

        pass

    # public methods
    def set_up(self, matdat, user_params):
        iam = self.name + ".set_up"

        # parse parameters
        self.parse_parameters(user_params)

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
        mu, nu, k = self.ui0[1], self.ui0[3], self.ui0[4]
        self.ui = self.ui0
        mui = np.array([mu, nu, k])
        self.bulk_modulus, self.shear_modulus = k, mu

        if self.code == "python":
            self.mui = self._py_set_up(mui)
        else:
            self.mui = self._fort_set_up(mui)

        # register the green lagrange strain and second Piola-Kirchhoff stress
        matdat.register_data("green strain","SymTensor",
                            init_val = np.zeros(6),
                            plot_key = "GREEN_STRAIN")
        matdat.register_data("pk2 stress","SymTensor",
                            init_val = np.zeros(6),
                            plot_key = "PK2")

        return

    def jacobian(self, simdat, matdat):
        v = matdat.get_data("prescribed stress components")
        return self.J0[[[x] for x in v],v]

    def update_state(self, simdat, matdat):
        """
           update the material state based on current state and strain increment
        """
        # deformation gradient and its determinant
        F = matdat.get_data("deformation gradient")

        if self.code == "python":
            E, pk2, sig = _py_update_state(self.mui, F)

        else:
            a = [1, self.mui, F, pu.migError, pu.migMessage]
            if not PC_F2PY_CALLBACK:
                a = a[:-2]
            E, pk2, sig = mtllib.finite_elast_calc(*a)

        matdat.store_data("stress", sig)
        matdat.store_data("pk2 stress", pk2)
        matdat.store_data("green strain", E)

        return

    def _py_set_up(self, mui):

        mu, nu, k = mui
        if nu < 0.:
            pu.reportWarning(iam, "neg Poisson")

        if mu <= 0.:
            pu.reportError(iam, "Shear modulus G must be positive")

        # compute c11, c12, and c44
        c11 = k + 4. / 3. * mu
        c12 = k - 2. / 3. * mu
        c44 = mu

        return np.array([c11, c12, c44])

    def _fort_set_up(self, mui):
        props = np.array(mui)
        a = [props, pu.migError, pu.migMessage]
        if not PC_F2PY_CALLBACK:
            a = a[:-2]
        ui = mtllib.finite_elast_chk(*a)
        return ui


def _py_update_state(ui, F):

    # user input
    c11, c12, c44 = ui

    # green lagrange strain
    E = 0.5 * (mt.tada(F) - delta)

    # PK2 stress
    pk2 = np.zeros(6)
    pk2[0] = c11 * E[0] + c12 * E[1] + c12 * E[2]
    pk2[1] = c12 * E[0] + c11 * E[1] + c12 * E[2]
    pk2[2] = c12 * E[0] + c12 * E[1] + c11 * E[2]
    pk2[3:] = c44 * E[3:]

    sig = mt.push(pk2, F)

    return E, pk2, sig

