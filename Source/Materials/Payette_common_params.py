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

'''
NAME
   Payette_common_params

PURPOSE
   All material data in the Payette/Source/Materials/ directory are given in
   terms of the common properties listed below.

   When adding a new model to Payette, the parameters for that model must be
   added to the common_mtl_params dictionary if it does not already exist. The
   format of the dictionary is:

              'descriptive parameter name':'parameter type'

   valid types are: float, int, string, bool

   By using common parameter names, multiple material models can share parameter
   files for different materials without modification
'''

common_mtl_params = {
    # elasticities
    'youngs modulus':'float',
    'poissons ratio':'float',
    'first lame constant':'float',
    'second lame constant':'float',
    'constrained modulus':'float',
    'bulk modulus':'float',
    'bulk modulus id':'int',
    'ratio lateral to axial stress in uniaxial strain':'float',
    'first nonlinear bulk modulus parameter':'float',
    'second nonlinear bulk modulus parameter':'float',
    'third nonlinear bulk modulus parameter':'float',
    'fourth nonlinear bulk modulus parameter':'float',
    'deriv bulk bulk modulus wrt pressure ':'float',
    'shear modulus':'float',
    'shear modulus id':'int',
    'first nonlinear shear modulus parameter':'float',
    'second nonlinear shear modulus parameter':'float',
    'third nonlinear shear modulus parameter':'float',
    'fourth nonlinear shear modulus parameter':'float',
    'deriv shear modulus wrt pressure':'float',
    'deriv shear modulus wrt temperature':'float',
    'constrained modulus':'float',

    # wave speeds
    'longitudinal wave speed':'float',
    'transverse wave speed':'float',
    'bulk wave speed':'float',
    'thin rod elastic wave speed':'float',

    # strength parameters
    'shear strength':'float',
    'tensile strength':'float',
    'yield strength':'float',
    'ultimate tensile strength':'float',
    'compressive yield strength':'float',
    'ultimate bearing strength':'float',
    'bearing strength':'float',
    'flexural strength':'float',
    'friction angle':'float',
    'drucker prager slope':'float',
    'hard power law coef':'float',

    # mass parameters
    'initial density':'float',
    'density':'float',

    # derived constants
    'dc1':'float',
    'dc2':'float',
    'dc3':'float',
    'dc4':'float',
    'dc5':'float',
    'dc6':'float',
    'dc7':'float',
    'dc8':'float',
    'dc9':'float',
    'dc10':'float',
    'dc11':'float',
    'dc12':'float',
    'dc13':'float',

    # temperature and energetic properties
    'initial temperature':'float',
    'initial spec int energy':'float',
    'gruneissen parameter':'float',
    'lin slope us up':'float',
    'quad slope us up':'float',
    'specific heat':'float',
    'melt temperature':'float',
    'coef thermal exp':'float',

    # hardness parameters
    'brinell hardness':'float',
    'knoop hardness':'float',
    'rockwell a hardness':'float',
    'rockwell b hardness':'float',
    'rockwell c hardness':'float',
    'vickers hardness':'float',

    # joint parameters
    'isotropic joint spacing':'float',
    'isotropic joint shear stiffness':'float',

    'joint spacing':'float',
    'joint shear stiffness':'float',
    'joint normal stiffness':'float',
    'shear failure parameter 1':'float',
    'shear failure parameter 2':'float',
    'shear failure parameter 3':'float',
    'shear failure parameter 4':'float',
    'init value of XL for pore collapse':'float',
    'pressure-volume parameter 1':'float',
    'pressure-volume parameter 2':'float',
    'compaction volume strain asymptote':'float',
    'shear failure Shape parameter':'float',
    'TXE/TXC Strength ratio':'float',
    'initial shear yield offset':'float',
    'kinematic hardening parameter':'float',
    'tension cut-off value of I1':'float',
    'tension cut-off of principal stress':'float',
    'relaxation time constant 1':'float',
    'relaxation time constant 2':'float',
    'relaxation time constant 3':'float',
    'relaxation time constant 4':'float',
    'relaxation time constant 5':'float',
    'relaxation time constant 6':'float',
    'relaxation time constant 7':'float',
    'octahedral shape ID':'float',
    'potential function parameter 1':'float',
    'potential function parameter 2':'float',
    'potential function parameter 3':'float',
    'potential function parameter 4':'float',
    'subcycle control parameter':'float',
    'dejavu':'float',
    'peak I1 hydrostatic tension strength':'float',
    'peak high pressure shear strength':'float',
    'Initial slope of limit surface at PEAKI1I':'float',
    'failed PEAKI1I':'float',
    'failed STREN':'float',
    'failure handling option':'float',
    'failed FSLOPE':'float',
    'failure statistics':'float',
    'equation of state ID':'float',
    'host eos boolean':'float',
    'free parameter 1':'float',
    'free parameter 2':'float',
    'free parameter 3':'float',
    'free parameter 4':'float',
    'free parameter 5':'float',
    'free parameter 6':'float',
    'plastic dilatation limit':'float',
    'fracture cut-off of principal stress':'float',
    'high pressure yield surface slope':'float',
    'failed YSLOPE':'float',
    'homologous temperature exponent':'float',
    'free thermal parameter 1':'float',
    'free thermal parameter 2':'float',
    'free thermal parameter 3':'float',
    'initial melt temperature':'float',
    'initial density':'float',
    'initial temperature':'float',
    'initial soundspeed':'float',
    'coefficient of linear term in US-UP fit':'float',
    'gruneisen parameter':'float',
    'specific heat':'float',
    'mg eos energy zero shift':'float',
    'initial porous density':'float',
    'crushup pressure':'float',
    'pressure at elastic limit':'float',
    'sound speed in foam':'float',
    'number of mg eos subcycles':'float',
    'coefficient of quad term in US-UP fit':'float',
    'mg eos 2 model type':'float',
    'density alias':'float',
    'temperature alias':'float',
    'S1 alias':'float',
    'GRPAR alias':'float',
    'coefficient in low-pressure mg eos term':'float',
    'constant in low-pressure mg eos term':'float',
    'power of compression in low-pressure me eos term':'float',
    'power for alpha integration':'float',
    'hardening modulus':'float',
    'hardening power':'float',
    'taylor quinney coefficient':'float',
    'induced anistropy flag':'int',
    'serial run flag':'int',

    # piezo ceramic model
    'c11':'float',
    'c33':'float',
    'c12':'float',
    'c13':'float',
    'c44':'float',
    'e311':'float',
    'e113':'float',
    'e333':'float',
    'ep11':'float',
    'ep33':'float',
    'dxr':'float',
    'dyr':'float',
    'dzr':'float',
    'smf':'float',

    # Failure
    'failed speed':'float',
    'fail 0':'float',
    'fail 1':'float',
    'fail 2':'float',
    'fail 3':'float',
    'fail 4':'float',
    'fail 5':'float',
    'fail 6':'float',
    'fail 7':'float',
    'fail 8':'float',
    'fail 9':'float',

    # orthotropic
    'space1':'float',
    'space2':'float',
    'space3':'float',
    'shrstiff2':'float',
    'shrstiff3':'float',
    'shrstiff1':'float',
    'ckn01':'float',
    'ckn03':'float',
    'ckn02':'float',
    'vmax1':'float',
    'vmax3':'float',
    'vmax2':'float',

    # domain switching model
    'c11':'float',
    'c33':'float',
    'c12':'float',
    'c13':'float',
    'c44':'float',
    'q11':'float',
    'q12':'float',
    'q44':'float',
    'reference temperature':'float',
    'reference density':'float',
    'currie temperature':'float',
    'currie constant':'float',
    'epsc':'float',
    'epsa':'float',
    'eremc':'float',
    'eremc':'float',
    'coercive field':'float',
    'a1':'float',
    'a11':'float',
    'a12':'float',
    'a111':'float',
    'a112':'float',
    'a123':'float',
    'w90':'float',
    'w180':'float',
    'axp':'float',
    'ayp':'float',
    'azp':'float',
    'p0':'float',
    'poled':'float',
    'refine':'float',
    'nbin':'float',
    'dbg':'float',
    'free0':'float',
    'free1':'float',
    'free2':'float',
    'free3':'float',
    'free4':'float'
    }

