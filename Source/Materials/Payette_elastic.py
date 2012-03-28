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

try:
    import Source.Materials.Library.elastic as mtllib
    imported = True
except:
    imported = False
    pass

from Payette_config import PAYETTE_MATERIALS_FORTRAN, PAYETTE_F2PY_CALLBACK

attributes = {
    "payette material":True,
    "name":"elastic",
    "fortran source":True,
    "build script":os.path.join(PAYETTE_MATERIALS_FORTRAN,"Elastic/build.py"),
    "aliases":["hooke","elasticity"],
    "material type":["mechanical"]
    }

class Elastic(ConstitutiveModelPrototype):
    """
    CLASS NAME
       Elastic

    PURPOSE
       Constitutive model for an elastic material. When instantiated, the Elastic
       material initializes itself by first checking the user input
       (_check_props) and then initializing any internal state variables
       (_set_field). Then, at each timestep, the driver update the Material state
       by calling updateState.

    METHODS
       _check_props
       _set_field
       updateState

    FORTRAN
       The core code for the Elastic material is contained in
       Fortran/Elastic/elastic.f.  The module Library/elastic is created by f2py.
       elastic.f defines the following public subroutines

          hookechk: fortran data check routine called by _check_props
          hookerxv: fortran field initialization  routine called by _set_field
          hooke_incremental: fortran stress update called by updateState

       See the documentation in elastic.f for more information.

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """
    def __init__(self):
        ConstitutiveModelPrototype.__init__(self)

        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = imported

        # register parameters
        self.registerParameter("LAM",0,aliases=[])
        self.registerParameter("G",1,aliases=['SHMOD'])
        self.registerParameter("E",2,aliases=['YMOD'])
        self.registerParameter("NU",3,aliases=['POISSONS'])
        self.registerParameter("K",4,aliases=['BKMOD'])
        self.registerParameter("H",5,aliases=[])
        self.registerParameter("KO",6,aliases=[])
        self.registerParameter("CL",7,aliases=[])
        self.registerParameter("CT",8,aliases=[])
        self.registerParameter("CO",9,aliases=[])
        self.registerParameter("CR",10,aliases=[])
        self.registerParameter("RHO",11,aliases=[])
        self.nprop = len(self.parameter_table.keys())
        self.ndc = 0

        pass

    # Public methods
    def setUp(self,simdat,matdat,user_params,f_params):
        iam = self.name + ".setUp(self,material,props)"

        if not imported: return

        # parse parameters
        self.parseParameters(user_params,f_params)

        # check parameters
        self.dc = np.zeros(self.ndc)
        self.ui = self._check_props()
        self.nsv,namea,keya,sv,rdim,iadvct,itype = self._set_field()
        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

        self.bulk_modulus,self.shear_modulus = self.ui[4],self.ui[1]

        pass

    # redefine Jacobian to return initial jacobian
    def jacobian(self,simdat,matdat):
        if not imported: return
        v = simdat.getData("prescribed stress components")
        return self.J0[[[x] for x in v],v]

    def updateState(self,simdat,matdat):
        """
           update the material state based on current state and strain increment
        """
        if not imported: return
        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")

        a = [dt,self.ui,sigold,d,svold,migError,migMessage]
        if not PAYETTE_F2PY_CALLBACK: a = a[:-2]
        sig, sv, usm = mtllib.hooke_incremental(*a)

        matdat.storeData("extra variables",sv)
        matdat.storeData("stress",sig)

        return

    # Private methods
    def _check_props(self):
        props = np.array(self.ui0)
        a = [props,props,props,migError,migMessage]
        if not PAYETTE_F2PY_CALLBACK: a = a[:-2]
        ui = mtllib.hookechk(*a)
        return ui

    def _set_field(self,*args,**kwargs):
        a = [self.ui,self.ui,self.ui,migError,migMessage]
        if not PAYETTE_F2PY_CALLBACK: a = a[:-2]
        field = mtllib.hookerxv(*a)
        return field

