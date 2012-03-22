#
#    Copyright (2011) Sandia Corporation.
#    Use and export of this program may require a license from
#    the United States Government.
#
from __future__ import print_function
import sys
import numpy as np
import scipy

from Source.Payette_tensor import *
from Source.Payette_utils import *
from Source.Payette_constitutive_model import ConstitutiveModelPrototype

try:
    import Source.Materials.Library.domain_switching_ceramic as mtllib
    imported = True
except:
    imported = False
    pass

attributes = {
    "payette material":True,
    "name":"domain_switching_ceramic",
    "fortran source":True,
    "build script":os.path.join(Payette_Materials_Fortran,
                                "DomainSwitchingCeramic/build.py"),
    "aliases":["multi domain ceramic"],
    "material type":["electromechanical"]
              }

class DomainSwitchingCeramic(ConstitutiveModelPrototype):
    """
    CLASS NAME
       DomainSwitchingCeramic

    PURPOSE
       Constitutive model for a DomainSwitchingCeramic material. When
       instantiated, the DomainSwitchingCeramic material initializes itself by
       first checking the user input (_check_props) and then initializing any
       internal state variables (_set_field). Then, at each timestep, the driver
       update the Material state by calling updateState.

    METHODS
       _check_props
       _set_field
       updateState

    FORTRAN

       The core code for the DomainSwitchingCeramic material is contained in

       ./Fortran/DomainSwitchingCeramic/domain_switching.F.
       ./Fortran/DomainSwitchingCeramic/emech7.F.

       The module Library/domain_switching is created by f2py.
       domain_switching.F defines the following public subroutines

          qseck7: fortran data check routine called by _check_props
          qsexv7: fortran field initialization  routine called by _set_field
          qsedr7: fortran stress update called by updateState

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """
    def __init__(self):
        ConstitutiveModelPrototype.__init__(self)
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = imported
        self.electric_field_model = True

        # register parameters
        nri = 38
        self.registerParameter("C11",0,aliases=[])
        self.registerParameter("C33",1,aliases=[])
        self.registerParameter("C12",2,aliases=[])
        self.registerParameter("C13",3,aliases=[])
        self.registerParameter("C44",4,aliases=[])
        # electrostriction
        self.registerParameter("Q11",5,aliases=[])
        self.registerParameter("Q12",6,aliases=[])
        self.registerParameter("Q44",7,aliases=[])
        # reference quantities
        self.registerParameter("T0",8,aliases=["TMPR"])
        self.registerParameter("R0",9,aliases=["RHO0"])
        # Currie
        self.registerParameter("TC",10,aliases=[])
        self.registerParameter("CC",11,aliases=[])
        # permittivities
        self.registerParameter("EPSC",12,aliases=["EPC"])
        self.registerParameter("EPSA",13,aliases=["EPA"])
        # remanant strains
        self.registerParameter("EREMC",14,aliases=["EC"])
        self.registerParameter("EREMA",15,aliases=["EA"])
        self.registerParameter("CF",16,aliases=["CFIELD"])
        self.registerParameter("A1",17,aliases=["A001"])
        self.registerParameter("A11",18,aliases=["A011"])
        self.registerParameter("A12",19,aliases=["A012"])
        self.registerParameter("A111",20,aliases=["A111"])
        self.registerParameter("A112",21,aliases=["A112"])
        self.registerParameter("A123",22,aliases=["A123"])
        self.registerParameter("W90",23,aliases=[])
        self.registerParameter("W180",24,aliases=[])
        self.registerParameter("AXP",25,aliases=[])
        self.registerParameter("AYP",26,aliases=[])
        self.registerParameter("AZP",27,aliases=[])
        self.registerParameter("P0",28,aliases=[])
        self.registerParameter("POLED",29,aliases=["PLD"])
        self.registerParameter("NBIN",30,aliases=[])
        self.registerParameter("SHMOD",31,aliases=[])
        self.registerParameter("BKMOD",32,aliases=[])
        self.registerParameter("XKSAT",33,aliases=[])
        self.registerParameter("ELFLG",34,aliases=[])
        self.registerParameter("EP1",35,aliases=[],parseable=False)
        self.registerParameter("EP2",36,aliases=[],parseable=False)
        self.registerParameter("EP3",37,aliases=[],parseable=False)
        self.registerParameter("DBG",nri,aliases=["DEBUG"])
        self.registerParameter("FREE0",nri+1,aliases=["NALGO"])
        self.registerParameter("FREE1",nri+2,aliases=["ANAGRAD"])
        self.registerParameter("FREE2",nri+3,aliases=[])
        self.registerParameter("FREE3",nri+4,aliases=[])
        self.registerParameter("FREE4",nri+5,aliases=[])
        self.nprop = len(self.parameter_table.keys())
        pass

    # Public Methods
    def setUp(self,simdat,matdat,user_params,f_params):
        iam = self.name + ".setUp(self,material,props)"

        # parse parameters
        self.parseParameters(user_params,f_params)

        # check parameters
        self.ui, self.dc = self._check_props()
        (self.ui,self.nsv,namea,keya,sv,
         rdim,iadvct,itype,iscal,
         bkd,permtv,polrzn) = self._set_field(self.ui,self.dc)
        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register non standard variables
        matdat.registerData("polarization","Vector",
                            init_val = np.zeros(3),
                            plot_key = "polrzn")
        matdat.registerData("electric displacement","Vector",
                            init_val = np.zeros(3),
                            plot_key = "edisp")
        matdat.registerData("block data","Array",
                            init_val = bkd)

        # register the extra variables with the material object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

        # initial shear and bulk moduli
        self.bulk_modulus = self.ui[self.parameter_table["BKMOD"]["ui pos"]]
        self.shear_modulus = self.ui[self.parameter_table["SHMOD"]["ui pos"]]

        # initial jacobian
        self.computeInitialJacobian(simdat,matdat,isotropic=False)
        pass

    def updateState(self,simdat,matdat):
        """
           update the material state based on current state and stretch
        """
        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        Fnew = simdat.getData("deformation gradient",form="Matrix")
        efield = simdat.getData("electric field")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")

        # right stretch
        Rstretch = sqrtm( np.dot( Fnew.T, Fnew ) )

        # rotation
        rotation = np.dot(Fnew,np.linalg.inv(Rstretch))

        # convert
        rotation = toArray(rotation,symmetric=False)
        Rstretch = toArray(Rstretch,symmetric=True)

        argv = [1,self.ui,self.ui,self.dc,svold,Rstretch,rotation,efield,
                sigold,migError,migMessage]
        if not Payette_F2Py_Callback: argv = argv[:-2]
        svnew,permtv,polrzn,edisp,signew = mtllib.qsedr7(*argv)

        # update data
        simdat.storeData("permittivity",permtv)
        matdat.storeData("stress",signew)
        matdat.storeData("extra variables",svnew)
        matdat.storeData("polarization",polrzn)
        matdat.storeData("electric displacement",edisp)

        return

    # Private Methods
    def _check_props(self,**kwargs):
        props = np.array(self.ui0)
        dc = np.zeros(13)
        argv = [props,props,dc,migError,migMessage]
        if not Payette_F2Py_Callback: argv = argv[-2:]
        props,dc = mtllib.qseck7(*argv)
        return props,dc

    def _set_field(self,*args,**kwargs):
        ui,dc = args[0],args[1]
        argv =[ui,ui,dc,migError,migMessage]
        if not Payette_F2Py_Callback: argv = argv[:-2]

        # request the extra variables
        ui,nsv,namea,keya,sv,rdim,iadvct,itype,iscal = mtllib.qsexv7(*argv)

        # initialize
        lbd = 3*(4+int(ui[self.parameter_table["NBIN"]["ui pos"]]))
        bkd = np.zeros(lbd)
        ibflg = 0
        argv = [ibflg,lbd,ui,ui,dc,nsv,bkd,lbd,sv,migError,migMessage]
        if not Payette_F2Py_Callback: argv = argv[:-2]
        ui,bkd,permtv,polrzn,sv = mtllib.dsc_init(*argv)

        return ui,nsv,namea,keya,sv,rdim,iadvct,itype,iscal,bkd,permtv,polrzn

