#
#    Copyright (2011) Sandia Corporation.
#    Use and export of this program may require a license from
#    the United States Government.
#
from __future__ import print_function
import sys
import numpy as np

from Source.Payette_utils import *
from Source.Payette_tensor import *
from Source.Payette_constitutive_model import ConstitutiveModelPrototype

from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK

try:
    import Source.Materials.Library.piezo_ceramic as mtllib
    imported = True
except:
    imported = False
    pass

from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK

attributes = {
    "payette material":True,
    "name":"piezo_ceramic",
    "fortran source":True,
    "build script":os.path.join(PC_MTLS_FORTRAN,"PiezoCeramic/build.py"),
    "aliases":["linear piezo","piezo electric"],
    "material type":["electromechanical"]
    }


class PiezoCeramic(ConstitutiveModelPrototype):
    """
    CLASS NAME
       PiezoCeramic

    PURPOSE
       Constitutive model for a PiezoCeramic material. When instantiated, the
       PiezoCeramic material initializes itself by first checking the user input
       (_check_props) and then initializing any internal state variables
       (_set_field). Then, at each timestep, the driver update the Material state
       by calling updateState.

    METHODS
       _check_props
       _set_field
       updateState

    FORTRAN

       The core code for the PiezoCeramic material is contained in
       ./Fortran/PiezoCeramic/piezo_ceramic.F. The module Library/piezo_ceramic
       is created by f2py. piezo_ceramic.f defines the following public subroutines

          qseck2: fortran data check routine called by _check_props
          qsexv2: fortran field initialization  routine called by _set_field
          qsedr2: fortran stress update called by updateState

       See piezo_ceramic.dat for more information.

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """
    def __init__(self):
        ConstitutiveModelPrototype.__init__(self)
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = imported
        self.electric_field_model = True

        self.registerParameter("C11",0,aliases=[])
        self.registerParameter("C33",1,aliases=[])
        self.registerParameter("C12",2,aliases=[])
        self.registerParameter("C13",3,aliases=[])
        self.registerParameter("C44",4,aliases=[])
        self.registerParameter("E311",5,aliases=[])
        self.registerParameter("E113",6,aliases=[])
        self.registerParameter("E333",7,aliases=[])
        self.registerParameter("EP11",8,aliases=[])
        self.registerParameter("EP33",9,aliases=[])
        self.registerParameter("DXR",10,aliases=[])
        self.registerParameter("DYR",11,aliases=[])
        self.registerParameter("DZR",12,aliases=[])
        self.registerParameter("SMF",13,aliases=[])
        self.nprop = len(self.parameter_table.keys())
        self.ndc = 8

    # Public methods
    def setUp(self,simdat,matdat,user_params,f_params):

        iam = self.name + ".setUp(self,payette,props)"

        # parse parameters
        self.parseParameters(user_params,f_params)

        self.dc = np.zeros(self.ndc)
        self.ui,self.dc = self._check_props()
        (self.ui,self.nsv,namea,keya,self.sv,
         rdim,iadvct,itype,iscal) = self._set_field()
        self.sv = np.zeros(self.nsv)
        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register non standard variables
        matdat.registerData("polarization","Vector",
                            init_val = np.zeros(3),
                            plot_key = "polrzn")
        matdat.registerData("electric displacement","Vector",
                            init_val = np.zeros(3),
                            plot_key = "edisp")

        # register the extra variables with the material object
        matdat.registerExtraVariables(self.nsv,namea,keya,self.sv)

        c11,c33,c12,c13 = self.ui[0], self.ui[1], self.ui[2], self.ui[3]
        lam  = 0.5*(c12 + c13)
        twomu  = 0.5*(c11 + c33) - lam
        self.shear_modulus = 0.5*twomu
        self.bulk_modulus = lam + twomu/3.
        return

    def updateState(self,simdat,matdat):
        """
           update the material state based on current state and stretch
        """
        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        Fnew = simdat.getData("deformation gradient",form="Matrix")
        efield = simdat.getData("electric field")
        sigold = matdat.getData("stress")
        dielec = simdat.getData("permittivity")

        # stretch and rotation
        Lstretch = sqrtm( np.dot( Fnew, Fnew.T ) )
        R = np.dot(np.linalg.inv(Lstretch),Fnew)
        Lstretch = toArray(Lstretch,symmetric=True)

        xtra,igeom,dielec,polrzn,scratch = 0,0,np.zeros(6),np.zeros(3),np.zeros(12)

        args = [self.ui,self.ui,self.dc,xtra,Lstretch,R,igeom,efield,dielec,polrzn,
                sigold,scratch,migError,migMessage]
        if not PC_F2PY_CALLBACK: args = args[:-2]
        dielec,polrzn,signew,scratch = mtllib.qsedr2(*args)

        # update data
        simdat.storeData("permittivity",dielec)
        matdat.storeData("stress",signew)
        matdat.storeData("polarization",polrzn)

        return

    # Private methods
    def _check_props(self):
        props = np.array(self.ui0)
        dc = np.zeros(self.ndc)
        args = [props,props,dc,migError,migMessage]
        if not PC_F2PY_CALLBACK: args = args[:-2]
        return mtllib.qseck2(*args)

    def _set_field(self,*args,**kwargs):
        args =[self.ui,self.ui,self.dc,migError,migMessage]
        if not PC_F2PY_CALLBACK: args = args[:-2]
        return mtllib.qsexv2(*args)

