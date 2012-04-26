#
#    Copyright (2011) Sandia Corporation.
#    Use and export of this program may require a license from
#    the United States Government.
#
import sys
import numpy as np

from Source.Payette_utils import *
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
try:
    import Source.Materials.Library.kayenta as mtllib
    imported = True
except:
    imported = False


from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK

attributes = {
    "payette material":True,
    "name":"lambda mie gruneisen",
    "fortran source":True,
    "build script":os.path.join(PC_MTLS_FORTRAN,"Lambda/MieGruneisen/build.py"),
    "aliases":[],
    "material type":["mechanical"]
    }

class LambdaMieGruneisen(ConstitutiveModelPrototype):
    def __init__(self):
        ConstitutiveModelPrototype.__init__(self)
        self.name = attributes["name"]
        self.aliases = attributes["aliases"]
        self.imported = imported

        self.registerParameter("R0",    0, aliases=[])
        self.registerParameter("T0",    1, aliases=[])
        self.registerParameter("CS",    2, aliases=[])
        self.registerParameter("S1",    3, aliases=[])
        self.registerParameter("G0",    4, aliases=[])
        self.registerParameter("CV",    5, aliases=[])
        self.registerParameter("ESFT",  6, aliases=[])
        self.registerParameter("RP",    7, aliases=[])
        self.registerParameter("PS",    8, aliases=[])
        self.registerParameter("PE",    9, aliases=[])
        self.registerParameter("CE",   10, aliases=[])
        self.registerParameter("NSUB", 11, aliases=[])
        self.registerParameter("S2",   12, aliases=[])
        self.registerParameter("TYP",  13, aliases=[])
        self.registerParameter("RO",   14, aliases=[])
        self.registerParameter("TO",   15, aliases=[])
        self.registerParameter("S",    16, aliases=[])
        self.registerParameter("GO",   17, aliases=[])
        self.registerParameter("B",    18, aliases=[])
        self.registerParameter("XB",   19, aliases=[])
        self.registerParameter("NB",   20, aliases=[])
        self.registerParameter("PWR",  21, aliases=[])

        self.ndc = 13
        self.dc = np.zeros(self.ndc)

        pass

    # Public methods
    def setUp(self,simdat,matdat,user_params,f_params):

        iam = self.name + ".setUp(self,material,props)"

        # parse parameters
        self.parseParameters(user_params,f_params)

        self.ui,self.dc = self._check_props()
        self.ui,self.nsv,namea,keya,sv,rdim,iadvct,itype = self._set_field()
        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

        self.bulk_modulus,self.shear_modulus = self.ui[0],self.ui[5]
        pass

    def updateState(self,simdat,matdat):
        '''
           update the material state based on current state and strain increment
        '''
        iam = self.name + ".updateState(self,simdat,matdat)"
        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")

        a = [dt,self.ui,self.ui,self.dc,sigold,d,svold,migError,migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        signew,svnew,usm = mtllib.kayenta_calc(*a)

        if svnew[18] < 0.:
            # Kayenta reached the spall cut off only using a portion of the
            # strain increment.
            void = svnew[47]
            ch = svnew[37]
            n = simdat.getData("number of steps")
            msg = ( "Kayenta returned with CRACK < 0, requesting void of "
                    "[{0}] on step [{1}] and coher of [{2}]".format(void,n,ch) )
            reportMessage(iam,msg)
            pass

        matdat.storeData("stress",signew)
        matdat.storeData("extra variables",svnew)

        return

    # Private methods
    def _check_props(self):
        props = np.array(self.ui0)
        a = [props,props,self.dc]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.lambda_prefix_eosmgi(*a)

    def _set_field(self):
        a = [self.ui,self.ui,self.dc,migError,migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.kayenta_rxv(*a)

