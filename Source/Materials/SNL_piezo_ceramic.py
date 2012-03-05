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
    import Source.Materials.Library.piezo_ceramic as mtllib
    imported = True
except:
    imported = False
    pass

attributes = {"payette material":True,
              "name":"piezo_ceramic",
              "fortran source":True,
              "build script":os.path.join(Payette_Materials_Fortran,"PiezoCeramic/build.py"),
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

    # Private methods
    def _check_props(self,**kwargs):
        props = kwargs["props"]
        dc = np.zeros(self.ndc)
        args = [props,props,dc,migError,migMessage]
        if not Payette_F2Py_Callback: args = args[-2:]
        return mtllib.qseck2(*args)

    def _set_field(self,*args,**kwargs):
        args =[self.ui,self.ui,self.dc,migError,migMessage]
        if not Payette_F2Py_Callback: args = args[:-2]
        return mtllib.qsexv2(*args)

    # Public methods
    def setUp(self,payette,props):
        iam = self.name + ".setUp(self,payette,props)"

        # check that the electric quantities are properly set up
        if "electric field" not in payette.material_data:
            reportError(iam,"efield not registered")
        elif "electric displacement" not in payette.material_data:
            reportError(iam,"edisp not registered")
        elif "polarization" not in payette.material_data:
            reportError(iam,"polrzn not registered")
            pass

        self.dc = np.zeros(self.ndc)
        self.ui0 = np.array(props)
        self.ui,self.dc = self._check_props(props=props)
        (self.ui,self.nsv,namea,keya,self.sv,
         self.rdim,self.iadvct,self.itype,iscal) = self._set_field()
        self.sv = np.zeros(self.nsv)
        self.namea = parseToken(self.nsv,namea)
        self.keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        payette.registerExtraVariables(self.nsv,self.namea,self.keya,self.sv)

        c11,c33,c12,c13 = self.ui[0], self.ui[1], self.ui[2], self.ui[3]
        lam  = 0.5*(c12 + c13)
        twomu  = 0.5*(c11 + c33) - lam
        self.shear_modulus = 0.5*twomu
        self.bulk_modulus = lam + twomu/3.
        self.computeInitialJacobian()
        return

    def updateState(self,*args,**kwargs):
        """
           update the material state based on current state and stretch
        """
        dt,d,fold,fnew,efield,sigold,svold = args
        fnew = np.matrix([ [ fnew[0], fnew[3], fnew[5] ],
                           [ fnew[6], fnew[1], fnew[4] ],
                           [ fnew[8], fnew[7], fnew[2] ] ],dtype="double")
        lam,evec = la.eigh(fnew*fnew.T)
        V = np.dot(np.dot(evec,np.diag(lam**(0.5))),evec.T)
        V = np.array([V[0,0],V[1,1],V[2,2],V[0,1],V[1,2],V[0,2]])
        R = np.eye(3)
        xtra,igeom,dielec,polrzn,scratch = 0,0,np.zeros(6),np.zeros(3),np.zeros(12)
        args = [self.ui,self.ui,self.dc,xtra,V,R,igeom,efield,dielec,polrzn,
                sigold,scratch,migError,migMessage]
        if not Payette_F2Py_Callback: args = args[:-2]
        dielec,polrzn,signew,scratch = mtllib.qsedr2(*args)
        self.sv = np.zeros(self.nsv)
        return signew,self.sv,polrzn
