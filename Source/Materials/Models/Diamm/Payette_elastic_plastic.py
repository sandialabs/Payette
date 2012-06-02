# The MIT License

# Copyright (c) 2011 Tim Fuller

# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from os.path import dirname, join, realpath
from numpy import array

from Source.Payette_utils import parse_token, report_and_raise_error, log_message
from Source.Payette_constitutive_model import ConstitutiveModelPrototype
try:
    import Source.Materials.Library.elastic_plastic as mtllib
    imported = True
except:
    imported = False
    pass

from Payette_config import PC_F2PY_CALLBACK

THIS_DIR = dirname(realpath(__file__))
attributes = {
    "payette material": False, # deactivated for now
    "name": "elastic_plastic",
    "code types": ("fortran", ),
    "fortran build script": join(THIS_DIR, "Build_elastic_plastic.py"),
    "aliases": ["inuced anisotropy"],
    "material type": ["mechanical"],
    "default material": True,
    }

class ElasticPlastic(ConstitutiveModelPrototype):

    def __init__(self, *args, **kwargs):

        super(ElasticPlastic, self).__init__(attributes, *args, **kwargs)

        self.imported = imported

        self.register_parameter("B0",0,aliases=['BKMOD'])
        self.register_parameter("B1",1,aliases=[])
        self.register_parameter("B2",2,aliases=[])
        self.register_parameter("G0",3,aliases=['SHMOD'])
        self.register_parameter("G1",4,aliases=[])
        self.register_parameter("G2",5,aliases=[])
        self.register_parameter("G3",6,aliases=[])
        self.register_parameter("A1",7,aliases=['yield strength'])
        self.register_parameter("A2",8,aliases=[])
        self.register_parameter("A3",9,aliases=[])
        self.register_parameter("A4",10,aliases=[])
        self.register_parameter("A5",11,aliases=[])
        self.register_parameter("A6",12,aliases=[])
        self.register_parameter("AN",13,aliases=[])
        self.register_parameter("R0",14,aliases=[])
        self.register_parameter("T0",15,aliases=[])
        self.register_parameter("C0",16,aliases=[])
        self.register_parameter("S1",17,aliases=['S1MG'])
        self.register_parameter("GP",18,aliases=['GRPAR'])
        self.register_parameter("CV",19,aliases=[])
        self.register_parameter("TM",20,aliases=[])
        self.register_parameter("T1",21,aliases=[])
        self.register_parameter("T2",22,aliases=[])
        self.register_parameter("T3",23,aliases=[])
        self.register_parameter("T4",24,aliases=[])
        self.register_parameter("TMPRXP",25,aliases=[])
        self.register_parameter("SC",26,aliases=[])
        self.register_parameter("IDK",27,aliases=[])
        self.register_parameter("IDG",28,aliases=[])
        self.register_parameter("A4PF",29,aliases=[])
        self.register_parameter("CHI",30,aliases=['TQC'])
        self.register_parameter("FREE01",31,aliases=['F1'])
        self.register_parameter("SERIAL",32,aliases=[])
        self.register_parameter("DEJAVU",33,aliases=[])
        self.register_parameter("DC1",34,aliases=[])
        self.register_parameter("DC2",35,aliases=[])
        self.register_parameter("DC3",36,aliases=[])
        self.register_parameter("DC4",37,aliases=[])
        self.register_parameter("DC5",38,aliases=[])
        self.register_parameter("DC6",39,aliases=[])
        self.register_parameter("DC7",40,aliases=[])
        self.register_parameter("DC8",41,aliases=[])
        self.register_parameter("DC9",42,aliases=[])
        self.register_parameter("DC10",43,aliases=[])
        self.register_parameter("DC11",44,aliases=[])
        self.register_parameter("DC12",45,aliases=[])
        self.register_parameter("DC13",46,aliases=[])
        self.nprop = len(self.parameter_table.keys())
        self.ndc = 0
        pass

    # Public method
    def set_up(self, matdat):
        iam = self.name + ".set_up(self,material,props)"

        # parse parameters
        self.parse_parameters()

        self.ui = self._check_props()
        (self.nsv,namea,keya,sv,rdim,iadvct,itype) = self._set_field()
        namea = parse_token(self.nsv,namea)
        keya = parse_token(self.nsv,keya)

        # register the extra variables with the payette object
        matdat.register_xtra_vars(self.nsv,namea,keya,sv)

        self.bulk_modulus,self.shear_modulus = self.ui[0],self.ui[3]

        pass

    def update_state(self, simdat, matdat):
        '''
           update the material state based on current state and strain increment
        '''
        dt = simdat.get_data("time step")
        d = matdat.get_data("rate of deformation")
        sigold = matdat.get_data("stress")
        svold = matdat.get_data("extra variables")
        a = [1, dt, self.ui, sigold, d, svold]
        if PC_F2PY_CALLBACK:
            a.extend([report_and_raise_error, log_message])
        a.append(self.nsv)
        signew, svnew, usm = mtllib.diamm_calc(*a)

        # update data
        matdat.store_data("stress",signew)
        matdat.store_data("extra variables",svnew)

        return


    # Private method
    def _check_props(self,**kwargs):
        props = array(self.ui0)
        a = [props]
        if PC_F2PY_CALLBACK:
            a.extend([report_and_raise_error, log_message])
        ui = mtllib.dmmchk(*a)
        return ui

    def _set_field(self,*args,**kwargs):
        a = [self.ui]
        if PC_F2PY_CALLBACK:
            a.extend([report_and_raise_error, log_message])
        return mtllib.dmmrxv(*a)

