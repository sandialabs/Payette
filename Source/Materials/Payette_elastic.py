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

attributes = {"payette material":True,
              "name":"elastic",
              "fortran source":True,
              "build script":os.path.join(Payette_Materials_Fortran,"Elastic/build.py"),
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

    # Private methods
    def _check_props(self,*args,**kwargs):
        a = [kwargs["props"],kwargs["props"],kwargs["props"],migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        ui = mtllib.hookechk(*a)
        return ui

    def _set_field(self,*args,**kwargs):
        a = [self.ui,self.ui,self.ui,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        else: return mtllib.hookerxv(*a)

    # Public methods
    def setUp(self,payette,props):
        if not imported: return
        self.dc = np.zeros(self.ndc)
        self.ui0 = np.array(props)
        self.ui = self._check_props(props=props)
        (self.nsv,namea,keya,self.sv,
         self.rdim,self.iadvct,self.itype) = self._set_field()
        self.namea = parseToken(self.nsv,namea)
        self.keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        payette.registerExtraVariables(self.nsv,self.namea,self.keya,self.sv)

        self.bulk_modulus,self.shear_modulus = self.ui[4],self.ui[1]
        self.computeInitialJacobian()
        pass

    # redefine Jacobian to return initial jacobian
    def jacobian(self,dt,d,Fold,Fnew,EF,sig,sv,v,*args,**kwargs):
        if not imported: return
        return self.J0[[[x] for x in v],v]

    def updateState(self,*args,**kwargs):
        """
           update the material state based on current state and strain increment
        """
        if not imported: return
        dt,d,fold,fnew,efield,sigold,svold = args
        a = [dt,self.ui,sigold,d,svold,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        sig, sv, usm = mtllib.hooke_incremental(*a)
        self.sv = np.array(sv)
        return sig,sv
