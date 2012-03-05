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

import copy
import sys
import logging

from Source.Payette_utils import *

try: from Source.Materials.Payette_installed_materials import *
except ImportError:
    reportError(__file__,"buildPayette.py must be run to create "
                "Source/Materials/Payette_installed_materials.py")

class Material:
    '''
    CLASS NAME
       Material

    PURPOSE
       Material is the container for material related data for a Payette object. It
       is instantiated by the Payette object.

    DATA
       constitutiveModel: Material's constitutive model

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self, constitutive_model ):
        self.cm = constitutive_model
        pass

    def constitutiveModel(self):
        return self.cm
