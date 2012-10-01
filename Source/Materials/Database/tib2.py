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

"""Material property file
   MATERIAL:  Titanium Diboride (TiB2)
   SOURCE NIST
   UNITS Kg/m/s/K
"""
parameters = {"initial density":4500,
              "vickers hardness":25,
              "flexural strength":400e6,
              "compressive yield strength":1.8e9,
              "shear modulus":255e9,
              "deriv shear modulus wrt pressure":9,
              "deriv bulk modulus wrt pressure":2,
              "bulk modulus":240e9,
              "youngs modulus":565e9,
              "poissons ratio":0.108,
              "specific heat":617,
              "melt temperature":2976,
              "coef thermal exp":860,
              "lin slope us up":1,
              "gruneissen parameter":1.71,
              "longitudinal wave speed":11000.4,
              "shear wave speed":754}
