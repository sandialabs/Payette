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

from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK

attributes = {
    "payette material":True,
    "name":"domain_switching_ceramic",
    "fortran source":True,
    "build script":os.path.join(PC_MTLS_FORTRAN,
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

          dsc_check: fortran data check routine called by _check_props
          dsc_extra: fortran field initialization  routine called by _set_field
          dsc_drvr: fortran stress update called by updateState

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
        nri, ndc, nei = 20, 5, 6
        # elasticities
        self.registerParameter("C11", 0, aliases=[])
        self.registerParameter("C12", 1, aliases=[])
        self.registerParameter("C44", 2, aliases=[])
        # electrostriction
        self.registerParameter("Q11", 3, aliases=[])
        self.registerParameter("Q12", 4, aliases=[])
        self.registerParameter("Q44", 5, aliases=[])
        # permittivities
        self.registerParameter("EPSC", 6, aliases=["EPC"])
        self.registerParameter("EPSA", 7, aliases=["EPA"])
        # coercive field
        self.registerParameter("CF", 8, aliases=["CFIELD"])
        # higher order coupling constants
        self.registerParameter("A1", 9, aliases=["A001"])
        self.registerParameter("A11", 10, aliases=["A011"])
        self.registerParameter("A12", 11, aliases=["A012"])
        self.registerParameter("A111", 12, aliases=["A111"])
        self.registerParameter("A112", 13, aliases=["A112"])
        self.registerParameter("A123", 14, aliases=["A123"])
        # polarization directons
        self.registerParameter("AXP", 15, aliases=[])
        self.registerParameter("AYP", 16, aliases=[])
        self.registerParameter("AZP", 17, aliases=[])
        # magnitude of spontanious polarization
        self.registerParameter("P0", 18, aliases=[])
        self.registerParameter("XKSAT", 19, aliases=[])
        # derived constants
        self.registerParameter("SHMOD", nri, aliases=[])
        self.registerParameter("BKMOD", nri + 1, aliases=[])
        self.registerParameter("EP1", nri + 2, aliases=[], parseable=False)
        self.registerParameter("EP2", nri + 3, aliases=[], parseable=False)
        self.registerParameter("EP3", nri + 4, aliases=[], parseable=False)
        # extended input
        self.registerParameter("DBG", nri + ndc,aliases=["DEBUG"])
        self.registerParameter("FREE0", nri + ndc + 1, aliases=[])
        self.registerParameter("FREE1", nri + ndc + 2, aliases=[])
        self.registerParameter("FREE2", nri + ndc + 3, aliases=[])
        self.registerParameter("FREE3", nri + ndc + 4, aliases=[])
        self.registerParameter("FREE4", nri + ndc + 5, aliases=[])
        self.nprop = len(self.parameter_table.keys())
        pass

    # Public Methods
    def setUp(self, simdat, matdat, user_params, f_params):
        iam = self.name + ".setUp(self, material, props)"

        # parse parameters
        self.parseParameters(user_params, f_params)

        # check parameters
        self.ui = self._check_props()
        sys.exit("check me")
        field = self._set_field()
        (self.ui, self.nsv, namea, keya, sv,
         rdim, iadvct, itype, iscal, bkd, permtv, polrzn) = field
        namea = parseToken(self.nsv, namea)
        keya = parseToken(self.nsv, keya)

        # register non standard variables
        matdat.registerData("polarization", "Vector",
                            init_val = np.zeros(3),
                            plot_key = "polrzn")
        matdat.registerData("electric displacement", "Vector",
                            init_val = np.zeros(3),
                            plot_key = "edisp")
        matdat.registerData("block data", "Array",
                            init_val = bkd)

        # register the extra variables with the material object
        matdat.registerExtraVariables(self.nsv, namea, keya, sv)

        # initial shear and bulk moduli
        self.bulk_modulus = self.ui[self.parameter_table["BKMOD"]["ui pos"]]
        self.shear_modulus = self.ui[self.parameter_table["SHMOD"]["ui pos"]]

        # initial jacobian
        self.computeInitialJacobian(simdat, matdat, isotropic=False)
        pass

    def updateState(self, simdat, matdat):
        """
           update the material state based on current state and stretch
        """
        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        Fnew = simdat.getData("deformation gradient", form="Matrix")
        efield = simdat.getData("electric field")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")

        # right stretch
        Rstretch = sqrtm( np.dot( Fnew.T, Fnew ) )

        # rotation
        rotation = np.dot(Fnew, np.linalg.inv(Rstretch))

        # convert
        rotation = toArray(rotation, symmetric=False)
        Rstretch = toArray(Rstretch, symmetric=True)

        argv = [1, self.ui, svold, Rstretch, rotation, efield,
                sigold, migError, migMessage]
        if not PC_F2PY_CALLBACK: argv = argv[:-2]
        svnew, permtv, polrzn, edisp, signew = mtllib.dsc_drvr(*argv)

        # update data
        simdat.storeData("permittivity", permtv)
        matdat.storeData("stress", signew)
        matdat.storeData("extra variables", svnew)
        matdat.storeData("polarization", polrzn)
        matdat.storeData("electric displacement", edisp)

        return

    # Private Methods
    def _check_props(self, **kwargs):
        props = np.array(self.ui0)
        argv = [props, migError, migMessage]
        if not PC_F2PY_CALLBACK: argv = argv[:-2]
        props = mtllib.dsc_check(*argv)
        return props

    def _set_field(self):
        ui = np.array(self.ui)
        argv =[ui, migError, migMessage]
        if not PC_F2PY_CALLBACK: argv = argv[:-2]

        # request the extra variables
        (ui, nsv, namea, keya, sv, rdim, iadvct, itype, iscal) = (
            mtllib.dsc_extra(*argv))

        # initialize
        lbd = 3*(4+int(ui[self.parameter_table["NBIN"]["ui pos"]]))
        bkd = np.zeros(lbd)
        ibflg = 0
        argv = [ibflg, lbd, ui, nsv, bkd, lbd, sv, migError, migMessage]
        if not PC_F2PY_CALLBACK: argv = argv[:-2]
        ui, bkd, permtv, polrzn, sv = mtllib.dsc_init(*argv)

        return (ui, nsv, namea, keya, sv, rdim, iadvct, itype, iscal,
                bkd, permtv, polrzn)

