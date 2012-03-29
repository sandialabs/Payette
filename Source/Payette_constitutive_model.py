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

import numpy as np
import math
from Source.Payette_tensor import *

material_idx = 0

class ConstitutiveModelPrototype:
    '''
    CLASS NAME
       ConstitutiveModelPrototype

    PURPOSE
       Prototype class from which constutive models are derived

    METHODS
       setField
       checkProperties
       updateState
       modelParameters
       internalStateVariables
       derivedConstants
       isvKeys

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self):
        self.registered_params = []
        self.registered_param_idxs = []
        self.registered_params_and_aliases = []
        self.name = None
        self.aliases = []
        self.parameter_table = {}
        self.parameter_table_idx_map = {}
        self.nprop, self.ndc, self.nsv = 0, 0, 0
        self.J0 = None
        self.ui = np.zeros(self.nprop)
        self.dc = np.zeros(self.ndc)
        self.bulk_modulus, self.shear_modulus = 0., 0.
        self.electric_field_model = False
        self.multi_level_fail = False
        self.multi_level_fail_model = False
        self.cflg_idx, self.fratio_idx = None, None
        self.errors = 0
        pass

    def setUp(self,*args,**kwargs):
        """
           set up model
        """
        reportError(__file__,'Consitutive model must provide setUp method')
        return 1

    def checkSetUp(self):
        """
           check that model is properly set up
        """

        name = self.name
        iam = name + ".checkSetup"

        if self.errors:
            reportError(iam,"previously encountered parsing errors")
            pass

        if not any( self.ui ):
            reportError(iam,"empty ui array")
            pass

        if not self.bulk_modulus:
            reportError(iam,"bulk modulus not defined")
            pass

        if not self.shear_modulus:
            reportError(iam,"shear modulus not defined")
            pass

        if self.J0 is None:
            self.computeInitialJacobian()
            pass

        if not any( x for y in self.J0 for x in y ):
            reportError(iam,"iniatial Jacobian is empty")
            pass

        pass


    def registerParameter(self,param_name,param_idx,aliases=[],parseable=True):

        iam = self.name + ".registerParameter"
        if not isinstance(param_name,str):
            reportError(iam,"parameter name must be a string, got {0}"
                        .format(param_name))
            pass
        if not isinstance(param_idx,int):
            reportError(iam,"parameter index must be an int, got {0}"
                        .format(param_idx))
            pass
        if not isinstance(aliases,list):
            reportError(iam,"aliases must be a list, got {0}".format(aliases))
            pass

        # register full name, low case name, and aliases
        full_name = param_name
        param_name = param_name.lower().replace(" ","_")
        param_names = [param_name]
        param_names.extend([x.lower().replace(" ","_") for x in aliases])

        dupl_name = [x for x in param_names
                     if x in self.registered_params_and_aliases]
        if dupl_name:
            reportWarning(iam,"duplicate parameter names: {0}"
                          .format(", ".join(param_names)))
            self.errors += 1
            pass

        if param_idx in self.registered_param_idxs:
            reportWarning(iam,"duplicate ui location [{0}] in parameter table"
                          .format(", ".join(param_names)))
            self.errors += 1
            pass
        self.registered_params.extend(full_name)
        self.registered_params_and_aliases.extend(param_names)
        self.registered_param_idxs.append(param_idx)

        # populate parameter_table
        self.parameter_table[full_name] = { "name":full_name,
                                            "names":param_names,
                                            "ui pos":param_idx,
                                            "parseable":parseable}
        self.parameter_table_idx_map[param_idx] = full_name
        pass

    def parseParameters(self,user_params,param_file=None):
        """
           parse user input and populate the params array
        """
        iam = self.name + ".parseParameters"
        param_file = None # deactivated for now
        self.ui0 = np.zeros(self.nprop)

        # get parameters from parameters file (if specified) and put in
        # parameters array
        if param_file:
            keys = self.constitutive_model.parameter_table.keys()
            keyslow = [key.lower() for key in keys]
            for f_param in param_file:
                f_param_low = f_param.lower()
                if f_param_low in keyslow:
                    dic = self.constitutive_model.parameter_table[
                        keys[keyslow.index(f_param_low)]]
                    i = dic['ui pos']
                    parameters[i] = float(param_file[k])
                    pass
                continue
            pass

        # now look at the parameters in the input file
        if not param_file and len(user_params) == 0:
            reportError(iam,"No parameters given")
            return 1

        for param in user_params:

            name, val = self._parse_param_line(iam,param)

            # we have the name and value, now we need to get its position in the
            # ui array
            found = False
            try:
                ui_idx = self.parameter_table[name]["ui pos"]
                found = True

            except KeyError:
                # look for alias
                for parameter in self.parameter_table:
                    if name in self.parameter_table[parameter]["names"]:
                        ui_idx = self.parameter_table[parameter]["ui pos"]
                        found = True
                        break
                    continue
                pass

            if not found:
                reportWarning(iam,"unregistered parameter [{0}], will be ignored "
                              .format(name))
                continue

            self.ui0[ui_idx] = val
            continue

        return

    def _parse_param_line(self,caller,param_line):
        # we want to return a name, value pair for a string that comes in as
        # any of the following:
        #      name = value
        #      long_name = value
        #      long name = value
        #      name,value
        #      name value
        # etc.

        # strip and lowcase param_line
        param_line = param_line.strip().lower()

        # replace "=" and "," with spaces and split
        param_line = param_line.replace("="," ").replace(","," ").split()
        # param_line is now of form
        #     param_line = [ string, string, ..., string ]
        # we need to assume that the last string is the value and anything up
        # to the last is the name
        name = "_".join(param_line[0:len(param_line)-1])
        try:
            val = float(param_line[-1])
        except:
            reportError(caller,"could not convert parameter {0} to float"
                        .format(param_line[-1]))
            pass
        return name,val

    def intializeState(self,*args,**kwargs):
        pass

    def updateState(self,material_data):
        reportError(__file__,'Consitutive model must provide updateState method')
        return 1

    def initialParameters(self):
        return self.ui0

    def checkedParameters(self):
        return self.ui

    def modelParameters(self):
        return self.ui

    def initialJacobian(self):
        return self.J0

    def internalStateVariables(self):
        return self.sv

    def derivedConstants(self):
        return self.dc

    def isvKeys(self):
        return self.keya

    def computeInitialJacobian(self,simdat=None,matdat=None,isotropic=True):
        '''
        NAME
           computeInitialJacobian

        PURPOSE
           compute the initial material Jacobian J = dsig/deps, assuming
           isotropy
        '''
        if isotropic:
            J0 = np.zeros((6,6))
            threek,twog = 3.*self.bulk_modulus,2.*self.shear_modulus
            pr = (threek-twog)/(2.*threek+twog)
            c1,c2 = (1-pr)/(1+pr), pr/(1+pr)
            for i in range(3): J0[i,i] = threek*c1
            for i in range(3,6): J0[i,i] = twog
            J0[0,1],J0[0,2],J0[1,2],J0[1,0],J0[2,0],J0[2,1] = [threek*c2]*6
            self.J0 = np.array(J0)
            return

        else:
            # material is not isotropic, numerically compute the jacobian
            simdat.stashData("prescribed stress components")
            simdat.storeData("prescribed stress components",[0,1,2,3,4,5])
            self.J0 = self.jacobian(simdat,matdat)
            simdat.unstashData("prescribed stress components")
            return

        return


    def jacobian(self,simdat,matdat):
        '''
        NAME
           jacobian

        PURPOSE:
           Numerically compute and return a specified submatrix, Jsub, of the
           Jacobian matrix J = J_ij = dsigi/depsj. The submatrix returned is the
           one formed by the intersections of the rows and columns specified in
           the vector subscript array, v. That is, Jsub = J[v,v]. The physical
           array containing this submatrix is assumed to be dimensioned
           Jsub[nv,nv], where nv is the number of elements in v. Note that in the
           special case v = [1,2,3,4,5,6], with nv = 6, the matrix that is
           returned is the full Jacobian matrix, J.

           The components of Jsub are computed numerically using a centered
           differencing scheme which requires two calls to the material model
           subroutine for each element of v. The centering is about the point eps
           = epsold + d * dt, where d is the rate-of-strain array.

        INPUT:
           dt:   timestep
           d:    symmetric part of velocity gradient
           sig:  stress
           sv:   state variables
           v:     v
           args: not used
           kwargs: not used

        OUTPUT
           J: Jacobian of the deformation J = dsig/deps

        HISTORY
           This subroutine is a python implementation of a routine by the same name
           in Tom Pucick's MMD driver.

        AUTHORS
           Tom Pucick, original fortran implementation in the MMD driver
           Tim Fuller, Sandial National Laboratories, tjfulle@sandia.gov
        '''
    # local variables
        epsilon = 2.2e-16
        v = simdat.getData("prescribed stress components")
        nv = len(v)
        deps,Jsub = math.sqrt(epsilon),np.zeros((nv,nv))

        d = simdat.getData("rate of deformation")
        Fold = simdat.getData("deformation gradient",form="Matrix")
        dt = simdat.getData("time step")
        dtime = 1 if dt == 0. else dt

        # stash the data
        simdat.stashData("time step")
        simdat.stashData("rate of deformation")
        simdat.stashData("deformation gradient")

        for n in range(nv):
            # perturb forward
            dp = np.array(d)
            dp[v[n]] = d[v[n]] + (deps/dtime)/2.
            fp = Fold + np.dot(toMatrix(dp),Fold)*dtime
            simdat.storeData("rate of deformation",dp,old=True)
            simdat.storeData("deformation gradient",fp,old=True)
            self.updateState(simdat,matdat)
            sigp = matdat.getData("stress",cur=True)

            # perturb backward
            dm = np.array(d)
            dm[v[n]] = d[v[n]] - (deps/dtime)/2.
            fm = Fold + np.dot(toMatrix(dm),Fold)*dtime
            simdat.storeData("rate of deformation",dm,old=True)
            simdat.storeData("deformation gradient",fm,old=True)
            self.updateState(simdat,matdat)
            sigm = matdat.getData("stress",cur=True)

            Jsub[n,:] = (sigp[v] - sigm[v])/deps
            continue

        # restore data
        simdat.unstashData("time step")
        simdat.unstashData("deformation gradient")
        simdat.unstashData("rate of deformation")

        return Jsub

