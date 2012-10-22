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

from numpy import array, zeros

import Source.__config__ as cfg
from Source.Payette_utils import log_warning, report_and_raise_error, log_message
from Source.Payette_tensor import ata, push, I6
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
from Toolset.elastic_conversion import compute_elastic_constants

try:
    import finite_elastic as mtllib
    imported = True
except:
    imported = False
    pass


class FiniteElastic(ConstitutiveModelPrototype):
    """ Finite elasticity model.

    Computes the PK2 stress based on the Green-Lagrange strain and returns the
    Cauchy stress

    """

    def __init__(self, control_file, *args, **kwargs):
        super(FiniteElastic, self).__init__(control_file, *args, **kwargs)

        self.imported = True if self.code == "python" else imported

        # register parameters
        self.register_parameters_from_control_file()

        pass

    # public methods
    def set_up(self, matdat):
        iam = self.name + ".set_up"

        # parse parameters
        self.parse_parameters()

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
        mui = array([mu, nu, k])
        self.bulk_modulus, self.shear_modulus = k, mu

        if self.code == "python":
            self.mui = self._py_set_up(mui)
        else:
            self.mui = self._fort_set_up(mui)

        self.ui[-3:] = self.mui

        # register the green lagrange strain and second Piola-Kirchhoff stress
        matdat.register("green strain", "SymTensor", plot_key="GREEN_STRAIN")
        matdat.register("pk2 stress", "SymTensor", plot_key="PK2")

        return

    def jacobian(self, simdat, matdat):
        v = matdat.get("prescribed stress components", form="integer array")
        return self.J0[[[x] for x in v], v]

    def update_state(self, simdat, matdat):
        """
           update the material state based on current state and strain increment
        """
        # deformation gradient and its determinant
        F = matdat.get("deformation gradient")

        if self.code == "python":
            E, pk2, sig = _py_update_state(self.mui, F)

        else:
            a = [1, self.mui, F]
            if cfg.F2PY["callback"]:
                a.extend([report_and_raise_error, log_message])
            E, pk2, sig = mtllib.finite_elast_calc(*a)

        matdat.save("stress", sig)
        matdat.save("pk2 stress", pk2)
        matdat.save("green strain", E)

        return

    def _py_set_up(self, mui):

        mu, nu, k = mui
        if nu < 0.:
            log_warning("neg Poisson")

        if mu <= 0.:
            report_and_raise_error("Shear modulus G must be positive")

        # compute c11, c12, and c44
        c11 = k + 4. / 3. * mu
        c12 = k - 2. / 3. * mu
        c44 = mu

        return array([c11, c12, c44])

    def _fort_set_up(self, mui):
        props = array(mui)
        a = [props]
        if cfg.F2PY["callback"]:
            a.extend([report_and_raise_error, log_message])
        ui = mtllib.finite_elast_chk(*a)
        return ui


def _py_update_state(ui, F):

    # user input
    c11, c12, c44 = ui

    # green lagrange strain
    E = 0.5 * (ata(F) - I6)

    # PK2 stress
    pk2 = zeros(6)
    pk2[0] = c11 * E[0] + c12 * E[1] + c12 * E[2]
    pk2[1] = c12 * E[0] + c11 * E[1] + c12 * E[2]
    pk2[2] = c12 * E[0] + c12 * E[1] + c11 * E[2]
    pk2[3:] = c44 * E[3:]

    sig = push(pk2, F)

    return E, pk2, sig
