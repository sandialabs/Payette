# Copyright (2011) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
# rights in this software.

# The MIT License

# Copyright (c) Sandia Corporation

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

''' Material property file
    MATERIAL: Titanium Ti-6Al-4V
    SOURCE: from MatWeb (www.matweb.com)
    UNITS: in Kg/m/s/K
'''
parameters = {'initial density':4430,
              'brinell hardness':334,
              'knoop hardness':363,
              'rockwell c hardness':36,
              'vickers hardness':349,
              'tensile strength':880e6,
              'ultimate tensile strength':950e6,
              'shear strength':550e6,
              'ultimate bearing strength':1860e6,
              'bearing strength':1480e6,
              'compressive yield strength':970e6,
              'shear modulus':44.0e9,
              'youngs modulus':113.8e9,
              'poissons ratio':0.342,
              'specific heat':526.3,
              'melt temperature':1873,
              'coef thermal exp':860}
