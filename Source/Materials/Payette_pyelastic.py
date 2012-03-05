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

attributes = {"payette material":True,
              "name":"py elastic",
              "aliases":["python elastic","py hooke","python hook"],
              "material type":["mechanical"]
              }
class PyElastic(ConstitutiveModelPrototype):
    """
    CLASS NAME
       PyElastic

    PURPOSE

       Constitutive model for an elastic material in python. When instantiated,
       the Elastic material initializes itself by first checking the user input
       (_check_properties) and then initializing any internal state variables
       (_set_field). Then, at each timestep, the driver update the Material state
       by calling updateState.

    METHODS
       _check_properties
       _set_field
       updateState

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    def __init__(self):
        ConstitutiveModelPrototype.__init__(self)
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = True
        self.registerParameter("K",0,aliases=['BKMOD'])
        self.registerParameter("G",1,aliases=['SHMOD'])
        self.nprop = len(self.parameter_table.keys())
        self.ndc = 0
        pass

    # private methods
    def _check_properties(self,*args,**kwargs):
        K,G = kwargs["props"]

        msg = ""
        if K <= 0.0: msg += "Bulk modulus K must be positive.  "
        if G <= 0.0: msg += "Shear modulus G must be positive"
        if msg: reportError(__file__,msg)

        if 3.*K < 2.*G: msg += "neg Poisson (to avoid warning, set 3*B0>2*G0)"
        if msg: reportWarning(__file__,msg)
        self.bulk_modulus,self.shear_modulus = K,G

        # internally, work with lame parameters
        lam = K - 0.66666666666666666667*G
        return np.array([G,lam])

    def _set_field(self,*args,**kwargs):
        nsv = 2
        sv = np.array([0.,1.])
        namea,keya = ["Free 01","Free 02"], ["F01","F02"]
        rdim,iadvct,itype = [None]*3
        return nsv,namea,keya,sv,rdim,iadvct,itype

    # public methods
    def setUp(self,payette,props):
        self.ui0 = np.array(props)
        self.dc = np.zeros(self.ndc)
        self.ui = self._check_properties(props=props)
        (self.nsv,self.namea,self.keya,self.sv,
         self.rdim,self.iadvct,self.itype) = self._set_field()

        # register the extra variables with the payette object
        payette.registerExtraVariables(self.nsv,self.namea,self.keya,self.sv)

        self.computeInitialJacobian()
        return

    def jacobian(self,dt,d,Fold,Fnew,EF,sig,sv,v,*args,**kwargs):
        return self.J0[[[x] for x in v],v]

    def updateState(self,*args,**kwargs):
        """
           update the material state based on current state and strain increment
        """
        dt,d,fold,fnew,efield,sigold,svold = args
        de, delta = d*dt, np.array([1.,1.,1.,0.,0.,0.])
        dev = (de[0] + de[1] + de[2])
        twog, lam = 2.*self.ui[0], self.ui[1]
        signew = sigold + twog*de + lam*dev*delta
        return signew, self.sv

