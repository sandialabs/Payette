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
        self.surface_increments = None
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

        self.hugoniot = None
        for idx, tok in enumerate(user_params):
            if tok.startswith("path hugoniot"):
                self.hugoniot = [float(x) for x in tok.split()[2:]]
                if len(self.hugoniot) != 2:
                    sys.exit("Invalid hugoniot given")
                if not self.density_range[1] <= self.hugoniot[0] <= self.density_range[2]:
                    sys.exit("Initial density for hugoniot not on surface")
                if not self.temperature_range[1] <= self.hugoniot[1] <= self.temperature_range[2]:
                    sys.exit("Initial temperature for hugoniot not on surface")
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

    def run_eosmgr(self, rho, temp):
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
             self.alpha, migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.lambda_prefix_eosmgr(*a) # pressure, energy, soundspeed, scratch
 
    def run_eosmgv(self, rho, enrg):
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
             self.alpha, migError, migMessage]
        if not PC_F2PY_CALLBACK: a = a[:-2]
        return mtllib.lambda_prefix_eosmgv(*a) # return pres, temp, cs, scratch

    def setup_eos(self, simdat, matdat):
        '''
          Save all the things needed from simdat and matdat in the local object
        '''
        iam = self.name + ".setup_eos(simdat,matdat)"
        self.alpha = matdat.getData("extra variables")
        self.outdata = {"DENSITY": 0.0,
                        "TEMPERATURE": 0.0,
                        "ENERGY": 0.0,
                        "PRESSURE": 0.0,
                        "SOUNDSPEED": 0.0,
                        "DPDR":0.0,
                        "DPDT":0.0,
                        "DEDT":0.0,
                        "DEDR":0.0}
        return

    def update_val(self, token, val):
        iam = self.name + ".update_val(token, val)"
        if token not in self.outdata.keys():
            sys.exit("{0} not found in {1}"
                     .format(repr(token),repr(self.outdata.keys())))
        self.outdata[token] = val

    def evaluate_eos(self, rho=None, temp=None, enrg=None):
        '''
          Evaluate the eos - rho and temp are in CGSEV
          return a dictionary of all variables
        '''
        iam = self.name + ".evaluate_eos(rho, temp)"

        def compare(x, y, name):
            scale = max(abs(x), abs(y))
            if scale == 0.0:
                scale = 1.0
            diff = abs(x - y)
            if diff / scale > 100.0 * np.finfo(np.float).eps:
                print("{0} values diff. Relative difference: {1:10.3e}"
                      .format(name,float(diff/scale)))


        if rho != None and temp != None:
            r_pres, r_enrg, r_cs, scratch = self.run_eosmgr(rho,temp)
            r_dpdr, r_dpdt, r_dedt, r_dedr = scratch[0][:4]
            v_pres, v_temp, v_cs, scratch = self.run_eosmgv(rho,r_enrg)
            v_dpdr, v_dpdt, v_dedt, v_dedr = scratch[0][:4]
            self.update_val("DENSITY", rho)
            self.update_val("TEMPERATURE", temp)
            self.update_val("ENERGY", r_enrg)
        elif rho != None and enrg != None:
            v_pres, v_temp, v_cs, scratch = self.run_eosmgv(rho,enrg)
            v_dpdr, v_dpdt, v_dedt, v_dedr = scratch[0][:4]
            r_pres, r_enrg, r_cs, scratch = self.run_eosmgr(rho,v_temp)
            r_dpdr, r_dpdt, r_dedt, r_dedr = scratch[0][:4]
            self.update_val("DENSITY", rho)
            self.update_val("TEMPERATURE", v_temp)
            self.update_val("ENERGY", enrg)
        else:
            sys.exit(iam+": not used correctly.")
 
        compare(r_pres, v_pres, "pressure")
        compare(r_cs, v_cs, "cs")
        compare(r_dpdr,v_dpdr, "dpdr")
        compare(r_dpdt, v_dpdt, "dpdt")
        compare(r_dedt, v_dedt, "dedt")
        compare(r_dedr, v_dedr, "dedr")

        self.update_val("DPDR", r_dpdr)
        self.update_val("DPDT", r_dpdt)
        self.update_val("DEDT", r_dedt)
        self.update_val("DEDR", r_dedr)
        self.update_val("PRESSURE", r_pres)
        self.update_val("SOUNDSPEED", r_cs)

        return


    def updateState(self,simdat,matdat):
        '''
           update the material state based on current state and strain increment
        '''
        self.setup_eos(simdat, matdat)
        iam = self.name + ".updateState(self,simdat,matdat)"

        K2eV = 8.617343e-5
        erg2joule = 1.0e-4
        fmt = lambda x: "{0:20.10e}".format(float(x))
        fmtxt = lambda x: "{0:>20s}".format(str(x))

        if self.surface_increments != None:
            loginf("Begin surface")
            rho_0, rho_l, rho_u = self.density_range
            temp_0, temp_l, temp_u = self.temperature_range

            out_f = simdat.getOption("outfile")
            OUT_F = open(out_f,"w")
            DEJAVU = False
            for rho in np.linspace(rho_l, rho_u, self.surface_increments):
                for temp in np.linspace(temp_l, temp_u, self.surface_increments):
                    # convert to CGSEV from MKSK
                    tmprho = rho/1000.0
                    tmptemp = K2eV*temp

                    self.evaluate_eos(rho=tmprho, temp=tmptemp)

                    # Write the headers if this is the first time through
                    if not DEJAVU:
                        DEJAVU = True
                        headers = [x[0] for x in sorted(self.outdata.iteritems())]
                        msg = "{0}\n".format("".join([fmtxt(x) for x in headers]))
                        OUT_F.write(msg)
    
                    # Convert from CGSEV to MKSK
                    self.outdata["DENSITY"] *= 1000.0
                    self.outdata["ENERGY"] *= erg2joule
                    self.outdata["PRESSURE"] /= 10.0
                    self.outdata["SOUNDSPEED"] /= 100.0
                    self.outdata["TEMPERATURE"] /= K2eV

                    # Write the values
                    values = [x[1] for x in sorted(self.outdata.iteritems())]
                    msg = "{0}\n".format("".join([fmt(x) for x in values]))
                    OUT_F.write(msg)

            OUT_F.close()
            loginf("End surface")
            loginf("Surface file: {0}".format(out_f))


        if self.isotherm != None:
            loginf("Begin isotherm")
            isotherm_f = os.path.splitext(out_f)[0]+".isotherm"
            OUT_F = open(isotherm_f,"w")

            rho0 = self.isotherm[0]
            temp0 = self.isotherm[1]
            idx = 0
            DEJAVU = False
            for rho in np.linspace(rho0, rho_u, self.path_increments):
                idx += 1
                if idx%int(self.path_increments/10.0) == 0:
                    loginf("Isotherm step {0}/{1}".format(idx,self.path_increments))
                tmprho = rho/1000.0
                tmptemp = K2eV*temp0

                self.evaluate_eos(rho=tmprho, temp=tmptemp)

                # Write the headers if this is the first time through
                if not DEJAVU:
                    DEJAVU = True
                    headers = [x[0] for x in sorted(self.outdata.iteritems())]
                    msg = "{0}\n".format("".join([fmtxt(x) for x in headers]))
                    OUT_F.write(msg)

                # Convert from CGSEV to MKSK
                self.outdata["DENSITY"] *= 1000.0
                self.outdata["ENERGY"] *= erg2joule
                self.outdata["PRESSURE"] /= 10.0
                self.outdata["SOUNDSPEED"] /= 100.0
                self.outdata["TEMPERATURE"] /= K2eV

                # Write the values
                values = [x[1] for x in sorted(self.outdata.iteritems())]
                msg = "{0}\n".format("".join([fmt(x) for x in values]))
                OUT_F.write(msg)

            OUT_F.close()
            loginf("End isotherm")
            loginf("Isotherm file: {0}".format(isotherm_f))


        if self.hugoniot != None:
            loginf("Begin Hugoniot")


            hugoniot_f = os.path.splitext(out_f)[0]+".hugoniot"
            OUT_F = open(hugoniot_f,"w")

            init_density_MKSK = self.hugoniot[0]
            init_temperature_MKSK = self.hugoniot[1]

            # Convert to CGSEV
            init_density = init_density_MKSK/1000.0
            init_temperature = K2eV*init_temperature_MKSK

            self.evaluate_eos(rho=init_density, temp=init_temperature)

            init_energy = self.outdata["ENERGY"]
            init_pressure = self.outdata["PRESSURE"]

            idx = 0
            DEJAVU = False
            e = init_energy
            for rho in np.linspace(init_density_MKSK, rho_u, self.path_increments):
                idx += 1
                if idx%int(self.path_increments/10.0) == 0:
                    loginf("Hugoniot step {0}/{1}".format(idx,self.path_increments))

                #
                # Here we solve the Rankine-Hugoniot equation as
                # a function of energy with constant density:
                #
                # E-E0 == 0.5*[P(E,V)+P0]*(V0-V)
                #
                # Where V0 = 1/rho0 and V = 1/rho. We rewrite it as:
                #
                # 0.5*[P(E,V)+P0]*(V0-V)-E+E0 == 0.0 = f(E)
                #
                # The derivative is given by:
                #
                # df(E)/dE = 0.5*(dP/dE)*(1/rho0 - 1/rho) - 1
                #
                # The solution to the first equation is found by a simple
                # application of newton's method:
                #
                # x_n+1 = x_n - f(E)/(df(E)/dE)
                #

                r = rho/1000.0
                a = (1./init_density - 1./r)/2.

                converged_idx = 0
                CONVERGED = False
                while not CONVERGED:
                    converged_idx += 1

                    self.evaluate_eos(rho=r, enrg=e)

                    f = (self.outdata["PRESSURE"] + init_pressure)*a - e + init_energy
                    df = self.outdata["DPDT"]/self.outdata["DEDT"]*a - 1.0
                    e = e - f/df*1.5
                    errval = abs(f/init_energy)
                    if errval < 1.0e-9 or converged_idx > 100:
                        CONVERGED = True
                        if converged_idx > 100:
                            loginf("Max iterations reached (tol = {0:14.10e}).\n".format(1.0e-9)+
                                   "rel error   = {0:14.10e}\n".format(float(errval))+
                                   "abs error   = {0:14.10e}\n".format(float(f))+
                                   "func val    = {0:14.10e}\n".format(float(f))+
                                   "init_energy = {0:14.10e}\n".format(float(init_energy)))

                # Write the headers if this is the first time through
                if not DEJAVU:
                    DEJAVU = True
                    headers = [x[0] for x in sorted(self.outdata.iteritems())]
                    msg = "{0}\n".format("".join([fmtxt(x) for x in headers]))
                    OUT_F.write(msg)

                # Convert from CGSEV to MKSK
                self.outdata["DENSITY"] *= 1000.0
                self.outdata["ENERGY"] *= erg2joule
                self.outdata["PRESSURE"] /= 10.0
                self.outdata["SOUNDSPEED"] /= 100.0
                self.outdata["TEMPERATURE"] /= K2eV

                # Write the values
                values = [x[1] for x in sorted(self.outdata.iteritems())]
                msg = "{0}\n".format("".join([fmt(x) for x in values]))
                OUT_F.write(msg)

            OUT_F.close()
            loginf("End Hugoniot")
            loginf("Hugoniot file: {0}".format(hugoniot_f))


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

