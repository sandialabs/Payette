#
#    Copyright (2011) Sandia Corporation.
#    Use and export of this program may require a license from
#    the United States Government.
#
from __future__ import print_function
import sys
import numpy as np

from Source.Payette_utils import *
from Source.Payette_constitutive_model import ConstitutiveModelPrototype

try:
    import Source.Materials.SNL_kayenta as kayenta
    imported = True
except:
    imported = False
    pass


attributes = {"payette material":True,
              "name":'kayenta_qsfail',
              "fortran source":True,
              "build script":"Not_Needed",
              "depends":"kayenta",
              "aliases":["kayenta quasistatic failure"],
              "material type":["mechanical"]
              }


class KayentaQSFail(ConstitutiveModelPrototype):
    '''
    CLASS NAME
       Kayenta

    PURPOSE
       Constitutive model for a Kayenta material. When instantiated, the Kayenta
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
       ./Fortran/Kayenta/kayenta_f77.f. The module Library/kayenta is created by
       f2py. kayenta_f77.f defines the following public subroutines

          kayenta_chk: fortran data check routine called by _check_props
          kayenta_rxv: fortran field initialization  routine called by _set_field
          kayenta_update_state: fortran stress update called by updateState

       See the documentation in kayenta_f77.f for more information.

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self):
        ConstitutiveModelPrototype.__init__(self)

        if imported:
            kmm = kayenta.Kayenta()
            self.name = attributes["name"]
            self.aliases = attributes["aliases"]
            self.imported = kmm.imported
            self.parameter_table = kmm.parameter_table
            self.nprop = kmm.nprop
            self.ndc = kmm.ndc

            # this model is setup for multi-level failure
            self.multi_level_fail = True
        else:
            pass
        pass

    # Private methods
    def _check_props(self,**kwargs):
        a = [kwargs["props"],kwargs["props"],np.zeros(13),migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        return kayenta.mtllib.kayenta_chk(*a)

    def _set_field(self,*args,**kwargs):
        a = [self.ui,self.ui,self.dc,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        return kayenta.mtllib.kayenta_rxv(*a)

    # Public methods
    def setUp(self,payette,props):
        self.dc = np.zeros(self.ndc)
        self.ui0 = np.array(props)
        self.ui,self.dc = self._check_props(props=props)
        (self.ui,self.nsv,namea,keya,sv,
         self.rdim,self.iadvct,self.itype) = self._set_field()

        self.namea = parseToken(self.nsv,namea)
        self.keya = parseToken(self.nsv,keya)

        # append 2 extra nsvs to sv array
        nxsv = 2
        self.nsv = self.nsv + nxsv
        self.keya.append("CFLG")
        self.keya.append("FRATIO")
        self.namea.append("crack flag")
        self.namea.append("failure ratio")
        sv = np.append(sv,np.zeros(nxsv))

        # setup old and new sv arrays
        self.svold = np.array(sv)
        self.sv = np.array(sv)

        # register the extra variables with the payette object
        payette.registerExtraVariables(self.nsv,self.namea,self.keya,self.sv)

        self.bulk_modulus,self.shear_modulus = self.ui[0],self.ui[5]
        self.J0 = self.computeInitialJacobian()

        # this model is setup for multi-level failure
        self.cflg_idx = self.nsv - 2
        self.fratio_idx = self.nsv - 1
        return

    def updateState(self,*args,**kwargs):
        '''
           update the material state based on current state and strain increment
        '''
        dt,d,fold,fnew,efield,sigold,svold = args
        signew,svnew = np.array(sigold), np.array(svold)
        a = [dt,self.ui,self.dc,d,sigold,signew,svold,svnew,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        updated_state = kayenta.mtllib.kayenta_update_state(*a)
        sigold,signew,svold,svnew,usm = updated_state
        self.sv = np.array(svnew)

        return signew,svnew
