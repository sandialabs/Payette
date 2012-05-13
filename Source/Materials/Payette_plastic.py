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
from Source.Payette_tensor import sym_map
from Toolset.elastic_conversion import compute_elastic_constants

try:
    import Source.Materials.Library.plastic as mtllib
    imported = True
except:
    imported = False
    pass


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
attributes = {
    "payette material": True,
    "name": "plastic",
    "aliases": ["elastic plastic", "von mises"],
    "fortran source": True,
    "build script": os.path.join(THIS_DIR, "Build_plastic.py"),
    "material type": ["mechanical"],
    "default material": True,
    }

w = np.array([1., 1., 1., 2., 2., 2.])
delta = np.array([1., 1., 1., 0., 0., 0.])

class Plastic(ConstitutiveModelPrototype):
    """ Plasticity model.

    Use MFLG = 0 for python implementation, MFLG = 1 for fortran

    """

    def __init__(self, *args, **kwargs):
        super(Plastic, self).__init__()
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]

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
        self.register_parameter("C0", 9, aliases=[])
        self.register_parameter("CR", 10, aliases=[])
        self.register_parameter("RHO", 11, aliases=["DENSITY"])
        self.register_parameter("Y", 12, aliases=["Y0", "A1", "YIELD STRENGTH"])
        self.register_parameter("A", 13, aliases=[])
        self.register_parameter("C", 14, aliases=[])
        self.register_parameter("M", 15, aliases=[])
        self.nprop = len(self.parameter_table.keys())
        pass

    # public methods
    def set_up(self, matdat, user_params):
        iam = self.name + ".set_up"

        # parse parameters
        self.parse_parameters(user_params)

        # the plastic model only needs the bulk and shear modulus, but the
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
        mui = np.array([k, mu] + self.ui0[12:].tolist())

        if self.code == "python":
            self.mui, nxtra, xtra, names, keys = self._py_set_up(mui)
        else:
            self.mui, nxtra, xtra, names, keys = self._fort_set_up(mui)

        self.nsv = nxtra
        self.bulk_modulus, self.shear_modulus = self.mui[0], self.mui[1]

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

        if self.code == "python":
            sig, xtra = _py_update_state(self.mui, dt, d, sigold, xtra)

        else:
            a = [1, self.nsv, dt, self.mui, sigold, d, xtra, migError, migMessage]
            if not PC_F2PY_CALLBACK:
                a = a[:-2]
            sig, xtra = mtllib.plast_calc(*a)

        # store updated data
        matdat.store_data("extra variables", xtra)
        matdat.store_data("stress", sig)

    def _py_set_up(self, mui):

        iam = "_py_set_up"
        k, mu, y, a, c, m = mui

        if k <= 0.:
            reportError(iam, "Bulk modulus K must be positive")

        if mu <= 0.:
            reportError(iam, "Shear modulus MU must be positive")

        if y < 0.:
            reportError(iam, "Yield strength Y must be positive")

        if y == 0:
            y = 1.e99

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

        ui = np.array([k, mu, y, a, c, m])

        # register state variables
        names, keys = [], []

        # equivalent plastic strain
        names.append("distortional plastic strain")
        keys.append("gam")

        # back stress
        for i in range(6):
            names.append("{0} component of back stress".format(sym_map[i]))
            keys.append("BSIG{0}".format(sym_map[i]))
            continue
        nxtra = len(keys)
        xtra = np.zeros(nxtra)
        return ui, nxtra, xtra, names, keys

    def _fort_set_up(self, mui):
        ui = self._check_props(mui)
        nxtra, namea, keya, xtra, iadvct = self._set_field(ui)
        names = parseToken(nxtra, namea)
        keys = parseToken(nxtra, keya)
        return ui, nxtra, xtra, names, keys

    def _check_props(self, mui):
        props = np.array(mui)
        a = [props, migError, migMessage]
        if not PC_F2PY_CALLBACK:
            a = a[:-2]
        ui = mtllib.plast_chk(*a)
        return ui

    def _set_field(self, ui):
        a = [migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.plast_rxv(*a)

def _py_update_state(ui, dt, d, sigold, xtra):

    iam = "plastic._py_update_state"

    # passed quantities
    de = d * dt
    gam = xtra[0]
    bstress = xtra[1:]
    k, mu, y0, a, c, m = ui
    threek, twomu = 3. * k, 2. * mu

    # elastic predictor
    dsig = threek * _iso(de) + twomu * _dev(de)
    sig = sigold + dsig

    # elastic predictor relative to back stress - shifted stress
    xi = sig - bstress

    # deviator of shifted stress and its magnitude
    xid = _dev(xi)
    rt2j2 = _mag(xid)

    # yield stress
    y = y0 if c == 0. else y0 + c * gam ** (1 / m)
    radius = sqrt(2. / 3.) * y

    # check yield
    facyld = 0. if rt2j2 - radius <= 0. else 1.

    # yield surface normal and return direction
    #                   df/dsig
    #            n = -------------,  p = C : n
    #                 ||df/dsig||
    #
    #            df           xid         || df ||      1
    #           ---- = ----------------,  ||----|| = -------
    #           dsig    root2 * radius    ||dsig||    root2
    n = xid / radius
    p = threek * _iso(n) + twomu * _dev(n)

    # consistency parameter
    #                  n : dsig
    #         dgam = -----------,  H = dfda : ha + dfdy * hy
    #                 n : p - H

    # numerator
    num = _ddp(n, dsig)

    # denominator
    ha = 2. / 3. * a * _dev(n)
    dfda = -xid / sqrt(2.) / radius
    hy = 0. if c == 0. else m * c * ((y - y0) / c) ** ((m - 1) / m)
    dfdy = -1. / sqrt(3.)
    H = sqrt(2.) * (_ddp(dfda, ha) + dfdy * hy)
    dnom = _ddp(n, p) - H + (1. - facyld)

    dgam = facyld * num / dnom
    if dgam < 0.:
        reportError(iam, "negative dgam")

    # equivalet plastic strain
    gam += dgam

    # update back stress
    bstress = bstress + 2. / 3. * a * dgam * _dev(n)

    # update stress
    sig = sig - dgam * p

    # store data
    xtra = np.concatenate((np.array([gam]), bstress))

    return sig, xtra

def _ddp(a, b):
    """ double dot product of symmetric second order tensors a and b """
    return np.sum(w * a * b)

def _mag(a):
    """ magnitude of symmetric second order tensor a """
    return sqrt(_ddp(a, a))

def _dev(a):
    """ deviatoric part of symmetric second order tensor a """
    return a  - _iso(a)

def _iso(a):
    """ isotropic part of symmetric second order tensor a """
    return _ddp(a, delta) / 3. * delta
