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

''' Material properties file
    MATERIAL: 6061 T6 Aluminum
    SOURCE:   MatWeb (www.matweb.com)
    UNITS:    Kg/m/s/K
'''

parameters = {
    "Initial density":2700,
    "Brinell hardness":95,
    "Knoop hardness":120,
    "Rockwell A hardness":40,
    "Rockwell B hardness":60,
    "Vicker's Hardness":107,
    "Tensile strength":276e6,
    "Ultimate tensile strength":310e6,
    "Shear strength":207e6,
    "Ultimate bearing strength":607e6,
    "Bearing strength":386e6,
    "Power law hardening modulus":114e6,
    "Power law hardening power":.42,
    "Shear modulus":26.0e9,
    "Derivative of shear modulus wrt pressure":1.8,
    "Derivative of shear modulus wrt temperature":.000170016,
    "Young's modulus":68.9e9,
    "Poisson's ratio":0.330,
    "Specific heat":896,
    "Melt temperature":855,
    "Coefficient of thermal expansion":167,
    "Gruneisen parameter":1.97,
    "Linear slope Us/Up":1.4,
    "Bulk wave speed":5240,
    }
