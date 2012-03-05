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
import logging
import numpy as np

from Source.Payette_utils import *
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
try:
    import Source.Materials.Library.elastic_plastic as mtllib
    imported = True
except:
    imported = False
    pass

attributes = {"payette material":True,
              "name":"elastic plastic",
              "fortran source":True,
              "build script":os.path.join(Payette_Materials_Fortran,"ElasticPlastic/build.py"),
              "aliases":["elastic perfectly plastic", "von mises"],
              "material type":["mechanical"]
              }

class ElasticPlastic(ConstitutiveModelPrototype):
    def __init__(self):
        ConstitutiveModelPrototype.__init__(self)
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = imported

        self.registerParameter("B0",0,aliases=['BKMOD'])
        self.registerParameter("B1",1,aliases=[])
        self.registerParameter("B2",2,aliases=[])
        self.registerParameter("G0",3,aliases=['SHMOD'])
        self.registerParameter("G1",4,aliases=[])
        self.registerParameter("G2",5,aliases=[])
        self.registerParameter("G3",6,aliases=[])
        self.registerParameter("A1",7,aliases=['yield strength'])
        self.registerParameter("A2",8,aliases=[])
        self.registerParameter("A3",9,aliases=[])
        self.registerParameter("A4",10,aliases=[])
        self.registerParameter("A5",11,aliases=[])
        self.registerParameter("A6",12,aliases=[])
        self.registerParameter("AN",13,aliases=[])
        self.registerParameter("R0",14,aliases=['RHO0'])
        self.registerParameter("T0",15,aliases=['TMPR0'])
        self.registerParameter("C0",16,aliases=[])
        self.registerParameter("S1",17,aliases=['S1MG'])
        self.registerParameter("GRPAR",18,aliases=['GP'])
        self.registerParameter("CV",19,aliases=[])
        self.registerParameter("TM",20,aliases=['TMPRM0'])
        self.registerParameter("T1",21,aliases=[])
        self.registerParameter("T2",22,aliases=[])
        self.registerParameter("T3",23,aliases=[])
        self.registerParameter("T4",24,aliases=[])
        self.registerParameter("TMPRXP",25,aliases=[])
        self.registerParameter("SC",26,aliases=[])
        self.registerParameter("IDK",27,aliases=[])
        self.registerParameter("IDG",28,aliases=[])
        self.registerParameter("A4PF",29,aliases=[])
        self.registerParameter("CHI",30,aliases=['TQC'])
        self.registerParameter("FREE1",31,aliases=['FREE01', 'F1'])
        self.registerParameter("SERIAL",32,aliases=[])
        self.registerParameter("DEJAVU",33,aliases=[])
        self.nprop = len(self.parameter_table.keys())
        self.ndc = 0
        pass

    # Private methods
    def _check_props(self,**kwargs):
        props = kwargs["props"]
        a = [props,props,props,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        ui = mtllib.dmmchk(*a)
        return ui

    def _set_field(self,*args,**kwargs):
        a = [self.ui,self.ui,self.ui,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        return mtllib.dmmrxv(*a)

    # Public methods
    def setUp(self,payette,props):
        self.dc = np.zeros(self.ndc)
        self.ui0 = np.array(props)
        self.ui = self._check_props(props=props)
        (self.nsv,namea,keya,self.sv,
         self.rdim,self.iadvct,self.itype) = self._set_field()
        self.namea = parseToken(self.nsv,namea)
        self.keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        payette.registerExtraVariables(self.nsv,self.namea,self.keya,self.sv)

        self.bulk_modulus,self.shear_modulus = self.ui[0],self.ui[1]
        self.computeInitialJacobian()
        return

    def updateState(self,*args,**kwargs):
        """
           update the material state based on current state and strain increment
        """
        dt,d,fold,fnew,efield,sigold,svold = args
        a = [1,dt,self.ui,sigold,d,svold,migError,migMessage,self.nsv]
        if not Payette_F2Py_Callback:
            args.delete(migMessage)
            args.delete(migError)
            pass
        sig, sv, usm = mtllib.diamm_calc(*args)
        self.sv = np.array(sv)
        return sig,sv
