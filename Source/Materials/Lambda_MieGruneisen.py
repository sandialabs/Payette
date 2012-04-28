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
    import Source.Materials.Library.lambda_mie_gruneisen as mtllib
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

        self.num_ui = 22
        self.num_gc = 0
        self.num_dc = 13
        self.num_vi = 5

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
        self.nprop = len(self.parameter_table.keys())

        self.ui = np.zeros(self.num_ui)
        self.dc = np.zeros(self.num_dc)
        self.gc = np.zeros(self.num_gc)
        self.vi = np.zeros(self.num_vi)
        self.ui[1] = 2.567981e-2
        self.uc = " "
        self.nsv = 0

        pass

    # Public methods
    def setUp(self,simdat,matdat,user_params,f_params):

        iam = self.name + ".setUp(self,material,props)"

        # parse parameters
        self.parseParameters(user_params,f_params)

        self.ui, self.gc, self.dc, self.vi = self._check_props()
#        self.dc = np.array(self.dc, dtype=np.float64)
        self.nsv, namea, keya, sv, rdim, iadvct, itype = self._set_field()
        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

        self.bulk_modulus = self.ui[0] * self.ui[2] ** 2
        self.shear_modulus = 1.0
        pass

    def updateState(self,simdat,matdat):
        '''
           update the material state based on current state and strain increment
        '''
        iam = self.name + ".updateState(self,simdat,matdat)"
        print("who am i = ",iam)


        rho = 9.0
        temp = 300.0
        for rho in np.linspace(8.0,16.0,30):
            for temp in np.linspace(200.,25000.,30):
                a = [1,
                     self.ui0,
                     self.gc,
                     self.dc,
                     rho,
                     temp,
                     matdat.getData("extra variables"),
                     migError,
                     migMessage]
                if not PC_F2PY_CALLBACK: a = a[:-2]
                pres, enrg, cs, scratch = mtllib.lambda_prefix_eosmgr(*a)
                fmt = lambda x: "{0:20.10e}".format(float(x))
                print("{0}{1}{2}{3}".format(fmt(rho*1000),fmt(temp),fmt(pres/10.0),fmt(enrg)))
        
#        print("pressure: ", pres)
#        print("energy:   ", enrg)
#        print("soundspd: ", cs)
#        print("scratch:  ", scratch)
        sys.exit("halt")

        return

    # Private methods
    def _check_props(self):
        ui = np.array(self.ui0)
        gc = np.array(self.gc)
        dc = np.array(self.dc)
        uc = np.array(self.uc)
        mdc = self.num_dc
        ndc = 0
        vi = np.array(self.vi)
        a = [ui, gc, dc, uc, mdc, ndc, vi, migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        print(mtllib.lambda_prefix_eosmgi.__doc__)
        return mtllib.lambda_prefix_eosmgi(*a)

    def _set_field(self):
        ui = np.array(self.ui)
        gc = np.array(self.gc)
        dc = np.array(self.dc)
        a = [ui, gc, dc, migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        print(mtllib.lambda_prefix_eosmgk.__doc__)
        return  mtllib.lambda_prefix_eosmgk(*a)

