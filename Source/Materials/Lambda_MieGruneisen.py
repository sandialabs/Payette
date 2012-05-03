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
        super(LambdaMieGruneisen, self).__init__()
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
        self.surface_increments = 10
        for idx, tok in enumerate(user_params):
            if tok.startswith("surface_increments"):
                self.surface_increments = int("".join(tok.split()[1:]))
                user_params.pop(idx)
                break

        self.path_increments = 10
        for idx, tok in enumerate(user_params):
            if tok.startswith("path_increments"):
                self.path_increments = int("".join(tok.split()[1:]))
                user_params.pop(idx)
                break

        self.density_range = [0.0, 0.0, 0.0]
        for idx, tok in enumerate(user_params):
            if tok.startswith("density_range"):
                self.density_range = [float(x) for x in tok.split()[1:]]
                user_params.pop(idx)
                break

        self.temperature_range = [0.0, 0.0, 0.0]
        for idx, tok in enumerate(user_params):
            if tok.startswith("temperature_range"):
                self.temperature_range = [float(x) for x in tok.split()[1:]]
                user_params.pop(idx)
                break

        self.isotherm = None
        for idx, tok in enumerate(user_params):
            if tok.startswith("path isotherm"):
                self.isotherm = [float(x) for x in tok.split()[2:]]
                if len(self.isotherm) != 2:
                    sys.exit("Invalid isotherm given")
                if not self.density_range[1] <= self.isotherm[0] <= self.density_range[2]:
                    sys.exit("Initial density for isotherm not on surface")
                if not self.temperature_range[1] <= self.isotherm[1] <= self.temperature_range[2]:
                    sys.exit("Initial temperature for isotherm not on surface")
                user_params.pop(idx)
                break

        self.parseParameters(user_params,f_params)

        self.ui, self.gc, self.dc, self.vi = self._check_props()
        self.nsv, namea, keya, sv, rdim, iadvct, itype = self._set_field()
        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

        self.bulk_modulus = self.ui[0] * self.ui[2] ** 2
        self.shear_modulus = 1.0
        pass

    def run_eosmgr(self,simdat,matdat,rho,temp):
        '''
           runs eosmgr and returns the pressure, energy, soundspeed, and scratch

           input units:
               rho is  g/cm^3 ( 1 g/cm^3 = 1000 kg/m^3 )
               temp is eV     (     1 eV = 11604.505 K )

           output units:
               pres is in barye (  1 barye = 1/10 Pa      )
               enrg is in erg   (    1 erg = 1.0e-7 Joule )
               cs is in cm/s    ( 1 cm/sec = 0.01 m/sec   )
        '''
        a = [1, self.ui, self.gc, self.dc, rho, temp,
             matdat.getData("extra variables"), migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.lambda_prefix_eosmgr(*a) # pressure, energy, soundspeed, scratch
 
    def run_eosmgv(self,simdat,matdat,rho,enrg):
        '''
           runs eosmgv and returns the pressure, temperature, soundspeed, and scratch

           input units:
               rho is g/cm^3 (   1 g/cm^3 = 1000 kg/m^3           )
               enrg is erg   ( 1 erg/gram = 1.0e-4 Joule/kilogram )

           output units:
               pres is in barye (  1 barye = 1/10 Pa     )
               temp is in eV    (     1 eV = 11604.505 K )
               cs is in cm/s    ( 1 cm/sec = 0.01 m/sec  )

        '''
        a = [1, self.ui, self.gc, self.dc, rho, enrg,
             matdat.getData("extra variables"), migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.lambda_prefix_eosmgv(*a) # return pres, temp, cs, scratch

    def updateState(self,simdat,matdat):
        '''
           update the material state based on current state and strain increment
        '''
        def compare(x,y,name):
            scale = max(abs(x),abs(y))
            diff = abs(x-y)
            if diff/scale > 100.0*np.finfo(np.float).eps:
                print("{0} values diff. Relative difference: {1:10.3e}".format(name,float(diff/scale)))
                
        iam = self.name + ".updateState(self,simdat,matdat)"

        K2eV = 8.617343e-5
        erg2joule = 1.0e-4
        out_f = simdat.getOption("outfile")
        OUT_F = open(out_f,"w")
        fmt = lambda x: "{0:20.10e}".format(float(x))
        fmtxt = lambda x: "{0:>20s}".format(str(x))
        msg = "{0}\n".format("".join([fmtxt("DENSITY"),
                                      fmtxt("PRESSURE"),
                                      fmtxt("SOUNDSPEED"),
                                      fmtxt("TEMPERATURE"),
                                      fmtxt("TEMPERATURE_V"),
                                      fmtxt("ENERGY")]))
        OUT_F.write(msg)

        for rho in np.linspace(self.density_range[1],self.density_range[2],self.surface_increments):
            for temp in np.linspace(self.temperature_range[1],self.temperature_range[2],self.surface_increments):
                tmprho = rho/1000.0
                tmptemp = K2eV*temp
                r_pres, r_enrg, r_cs, scratch = self.run_eosmgr(simdat,matdat,tmprho,tmptemp)
                r_dpdr, r_dpdt, r_dedt, r_dedr = scratch[0][:4]
                v_pres, v_temp, v_cs, scratch = self.run_eosmgv(simdat,matdat,tmprho,r_enrg)
                v_dpdr, v_dpdt, v_dedt, v_dedr = scratch[0][:4]
                compare(r_pres,v_pres,"pressure")
                compare(r_cs,v_cs,"cs")
                compare(r_dpdr,v_dpdr,"dpdr")
                compare(r_dpdt,v_dpdt,"dpdt")
                compare(r_dedt,v_dedt,"dedt")
                compare(r_dedr,v_dedr,"dedr")
                array2print = [rho,
                              r_pres/10.0,
                              r_cs/100.0,
                              temp,
                              v_temp/K2eV,
                              erg2joule*r_enrg]
                msg = "{0}\n".format("".join([fmt(x) for x in array2print]))
                OUT_F.write(msg)
        OUT_F.close()
        print("Surface file: {0}".format(out_f))


        isotherm_f = os.path.splitext(out_f)[0]+".isotherm"
        OUT_F = open(isotherm_f,"w")
        msg = "{0}\n".format("".join([fmtxt("DENSITY"),
                                      fmtxt("PRESSURE"),
                                      fmtxt("SOUNDSPEED"),
                                      fmtxt("TEMPERATURE"),
                                      fmtxt("ENERGY")]))
        OUT_F.write(msg)
        if self.isotherm != None:
            rho0 = self.isotherm[0]
            temp0 = self.isotherm[1]
            for rho in np.linspace(rho0, self.density_range[2], self.path_increments):
                tmprho = rho/1000.0
                tmptemp = K2eV*temp0
                r_pres, r_enrg, r_cs, scratch = self.run_eosmgr(simdat,matdat,tmprho,tmptemp)
                v_pres, v_temp, v_cs, scratch = self.run_eosmgv(simdat,matdat,tmprho,r_enrg)
                msg = "{0}\n".format("".join([fmt(rho),
                                              fmt(r_pres/10.0),
                                              fmt(r_cs/100.0),
                                              fmt(temp0),
                                              fmt(erg2joule*r_enrg)]))
                OUT_F.write(msg)
            OUT_F.close()
            print("Isotherm file: {0}".format(isotherm_f))

        sys.exit() 
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
        return mtllib.lambda_prefix_eosmgi(*a)

    def _set_field(self):
        ui = np.array(self.ui)
        gc = np.array(self.gc)
        dc = np.array(self.dc)
        a = [ui, gc, dc, migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return  mtllib.lambda_prefix_eosmgk(*a)

