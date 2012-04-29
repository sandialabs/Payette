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

from Source.Payette_utils import *
from Source.Payette_constitutive_model import ConstitutiveModelPrototype

from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK

attributes = {
    "payette material": True,
    "name": "finite_elastic",
    "libname": "finite_elastic",
    "aliases": [],
    "material type": ["mechanical"],
    "default material": True,
    }
class FiniteElastic(ConstitutiveModelPrototype):
    """
    CLASS NAME
       PyElastic

    PURPOSE

       Constitutive model for an elastic material in python. When instantiated,
       the Elastic material initializes itself by first checking the user input
       (_check_props) and then initializing any internal state variables
       (_set_field). Then, at each timestep, the driver update the Material state
       by calling updateState.

    METHODS
       _check_props
       _set_field
       updateState

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    def __init__(self):
        super(FiniteElastic, self).__init__()
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = True

        # register parameters
        self.registerParameter("MU", 0, aliases=["SHMOD", "G", "G0"])
        self.registerParameter("NU", 1, aliases=["POISSONS_RATIO"])
        self.registerParameter("K", 2, aliases=["BKMOD", "B0"])
        self.registerParameter("C11", 3, aliases=[], parseable=False)
        self.registerParameter("C12", 4, aliases=[], parseable=False)
        self.registerParameter("C44", 5, aliases=[], parseable=False)
        self.nprop = len(self.parameter_table.keys())

        pass

    # public methods
    def setUp(self,simdat,matdat,user_params,f_params):
        iam = self.name + ".setUp(self,material,props)"

        # parse parameters
        self.parseParameters(user_params, f_params)

        mu, nu = self.ui0[0:2]

        if nu < 0.:
            reportWarning(iam, "neg Poisson")

        if mu <= 0.:
            reportError(iam, "Shear modulus G must be positive")

        # compute bulk modulus, c11, c12, and c44
        k = 2.0 * mu * (1 + nu) / (3. * (1. - 2. * nu))
        c11 = k + 4. / 3. * mu
        c12 = k - 2. / 3. * mu
        c44 = mu

        self.bulk_modulus, self.shear_modulus = k, mu

        self.ui = np.array([mu, nu, k, c11, c12, c44])

        # register the green lagrange strain and second Piola-Kirchhoff stress
        matdat.registerData("green strain","SymTensor",
                            init_val = np.zeros(6),
                            plot_key = "GREEN_STRAIN")
        matdat.registerData("pk2 stress","SymTensor",
                            init_val = np.zeros(6),
                            plot_key = "PK2")


        return

    def jacobian(self,simdat,matdat):
        v = simdat.getData("prescribed stress components")
        return self.J0[[[x] for x in v],v]

    def updateState(self,simdat,matdat):
        """
           update the material state based on current state and strain increment
        """
        # deformation gradient and its determinant
        F = simdat.getData("deformation gradient", form="Matrix")
        jac = np.linalg.det(F)

        # green lagrange strain
        E = 1. / 2. * (np.dot(F.T, F) - np.eye(3))

        # PK2 stress
        c11, c12, c44 = self.ui[3:]
        stress = np.zeros((3,3))
        stress[0, 0] = c11 * E[0, 0] + c12 * E[1, 1] + c12 * E[2, 2]
        stress[1, 1] = c12 * E[0, 0] + c11 * E[1, 1] + c12 * E[2, 2]
        stress[2, 2] = c12 * E[0, 0] + c12 * E[1, 1] + c11 * E[2, 2]
        stress[0, 1] = c44 * E[0, 1]
        stress[1, 2] = c44 * E[1, 2]
        stress[0, 2] = c44 * E[0, 2]
        stress[1, 0] = stress[0, 1]
        stress[2, 1] = stress[1, 2]
        stress[2, 0] = stress[0, 2]

        cauchy_stress = (1. / jac) * np.dot(F, np.dot(stress, F.T))

        matdat.storeData("stress", cauchy_stress)
        matdat.storeData("pk2 stress", stress)
        matdat.storeData("green strain", E)

        return
