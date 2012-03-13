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

    # Public methods
    def setUp(self,simdat,matdat,user_params,f_params):
        iam = self.name + ".setUp(self,material,props)"

        # parse parameters
        self.parseParameters(user_params,f_params)

        self.dc = np.zeros(self.ndc)
        self.ui, self.dc = self._check_props()
        (self.ui,self.nsv,namea,keya,sv,rdim,iadvct,itype) = self._set_field()

        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # append 2 extra nsvs to sv array
        nxsv = 2
        self.nsv = self.nsv + nxsv
        keya.append("CFLG")
        keya.append("FRATIO")
        namea.append("crack flag")
        namea.append("failure ratio")
        sv = np.append(sv,np.zeros(nxsv))

        # register the extra variables with the payette object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

        self.bulk_modulus,self.shear_modulus = self.ui[0],self.ui[5]
        self.J0 = self.computeInitialJacobian()

        # this model is setup for multi-level failure
        self.cflg_idx = self.nsv - 2
        self.fratio_idx = self.nsv - 1
        return

    def updateState(self,simdat,matdat):
        '''
           update the material state based on current state and strain increment
        '''

        iam = self.name + ".setUp(self,material,props)"

        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")
        signew = np.array(sigold)
        svnew = np.array(svold)

        a = [dt,self.ui,self.dc,d,sigold,signew,svold,svnew,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        updated_state = kayenta.mtllib.kayenta_update_state(*a)

        sigold,signew,svold,svnew,usm = updated_state

        matdat.storeData("stress",signew)
        matdat.storeData("extra variables",svnew)

        return

    # Private methods
    def _check_props(self,**kwargs):
        props = np.array(self.ui0)
        a = [props,props,self.dc,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        return kayenta.mtllib.kayenta_chk(*a)

    def _set_field(self,*args,**kwargs):
        a = [self.ui,self.ui,self.dc,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        return kayenta.mtllib.kayenta_rxv(*a)

