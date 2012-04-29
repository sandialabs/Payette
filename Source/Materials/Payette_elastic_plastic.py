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

from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK

attributes = {
    "payette material":True,
    "name":"elastic_plastic",
    "fortran source":True,
    "build script":os.path.join(PC_MTLS_FORTRAN,"ElasticPlastic/build.py"),
    "aliases":["inuced anisotropy"],
    "material type":["mechanical"],
    "default material": True,
    }

class ElasticPlastic(ConstitutiveModelPrototype):

    def __init__(self):

        super(ElasticPlastic, self).__init__()

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
        self.registerParameter("R0",14,aliases=[])
        self.registerParameter("T0",15,aliases=[])
        self.registerParameter("C0",16,aliases=[])
        self.registerParameter("S1",17,aliases=['S1MG'])
        self.registerParameter("GP",18,aliases=['GRPAR'])
        self.registerParameter("CV",19,aliases=[])
        self.registerParameter("TM",20,aliases=[])
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
        self.registerParameter("FREE01",31,aliases=['F1'])
        self.registerParameter("SERIAL",32,aliases=[])
        self.registerParameter("DEJAVU",33,aliases=[])
        self.registerParameter("DC1",34,aliases=[])
        self.registerParameter("DC2",35,aliases=[])
        self.registerParameter("DC3",36,aliases=[])
        self.registerParameter("DC4",37,aliases=[])
        self.registerParameter("DC5",38,aliases=[])
        self.registerParameter("DC6",39,aliases=[])
        self.registerParameter("DC7",40,aliases=[])
        self.registerParameter("DC8",41,aliases=[])
        self.registerParameter("DC9",42,aliases=[])
        self.registerParameter("DC10",43,aliases=[])
        self.registerParameter("DC11",44,aliases=[])
        self.registerParameter("DC12",45,aliases=[])
        self.registerParameter("DC13",46,aliases=[])
        self.nprop = len(self.parameter_table.keys())
        self.ndc = 0
        pass

    # Public method
    def setUp(self,simdat,matdat,user_params,f_params):
        iam = self.name + ".setUp(self,material,props)"

        # parse parameters
        self.parseParameters(user_params,f_params)

        self.ui = self._check_props()
        (self.nsv,namea,keya,sv,
         rdim,iadvct,itype) = self._set_field()
        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

        self.bulk_modulus,self.shear_modulus = self.ui[0],self.ui[3]

        pass

    def updateState(self,simdat,matdat):
        '''
           update the material state based on current state and strain increment
        '''
        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")
        a = [1,dt,self.ui,sigold,d,svold,migError,migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        a.append(self.nsv)
        signew,svnew,usm = mtllib.diamm_calc(*a)

        # update data
        matdat.storeData("stress",signew)
        matdat.storeData("extra variables",svnew)

        return


    # Private method
    def _check_props(self,**kwargs):
        props = np.array(self.ui0)
        a = [props,migError,migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        ui = mtllib.dmmchk(*a)
        return ui

    def _set_field(self,*args,**kwargs):
        a = [self.ui,migError,migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.dmmrxv(*a)

