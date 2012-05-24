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
import os
import imp
import Payette_config as pc
import Source.Payette_tensor as pt
import Source.Payette_utils as pu

material_idx = 0

class ConstitutiveModelPrototype(object):
    '''
    CLASS NAME
       ConstitutiveModelPrototype

    PURPOSE
       Prototype class from which constutive models are derived

    METHODS
       setField
       checkProperties
       update_state
       derivedConstants
       isvKeys

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self, init_data, *args, **kwargs):
        """Initialize the ConstitutiveModelPrototype object.

        The __init__ functions should be called by each constituve model that
        derives from the ConstitutiveModelPrototype base class.

        Parameters
        ----------
        init_data : dict
          keys
          name : str
            name of constitutive model
          aliases : list
            list of aliases
          code types : tuple
            tuple of types of code in which the model is implemented
          default code : tuple
            tuple of types of code in which the model is implemented
          material type : list
            type of material
        *args : list, optional
        **kwargs : dict, optional
          keys
          code : string
            code type specified by user

        """

        # pass init_data to class data
        self.name = init_data["name"]
        iam = self.name + "__init__"
        self.aliases = init_data["aliases"]
        if not isinstance(self.aliases, (list, tuple)):
            self.aliases = [self.aliases]

        # which code to use
        code_types = init_data["code types"]
        if not isinstance(code_types, (list, tuple)):
            code_types = [code_types]
        self.code = kwargs.get("code")
        if self.code is None:
            self.code = code_types[0]
        elif self.code not in code_types:
            pu.reportError(iam, ("requested code type {0} not supported by {1}"
                                 .format(self.code, self.name)))

        # @mswan{ -> this would be new
        # location of material data file - if any
        self.mtldat_f = init_data.get("material database")
        if self.mtldat_f is not None and not os.path.isfile(self.mtldat_f):
            pu.reportError(
                iam, "material data file {0} not found".format(self.mtldat_f))
        # @mswan}

        # specialized models
        mat_typ = init_data["material type"]
        self.electric_field_model = "electromechanical" in mat_typ
        self.eos_model = "eos" in mat_typ

        # data to be initialized later
        self.registered_params = []
        self.registered_param_idxs = []
        self.registered_params_and_aliases = []
        self.parameter_table = {}
        self.user_input_params = {}
        self.parameter_table_idx_map = {}
        self.nprop = 0
        self.ndc = 0
        self.nxtra = 0
        self.J0 = None
        self.ui = np.zeros(self.nprop)
        self.dc = np.zeros(self.ndc)
        self.bulk_modulus = 0.
        self.shear_modulus = 0.
        self.errors = 0

        pass

    def set_up(self, *args):
        """ set up model """
        pu.reportError(__file__,
                       'Constitutive model must provide set_up method')
        return

    def update_state(self, *args):
        """ update state """
        pu.reportError(__file__,
                       'Constitutive model must provide update_state method')
        return

    def finish_setup(self, matdat):
        """ check that model is properly set up """

        name = self.name
        iam = name + ".finish_setup"

        if self.errors:
            pu.reportError(iam, "previously encountered parsing errors")

        if not any(self.ui):
            pu.reportError(iam, "empty ui array")

        if self.eos_model:
            # mss: do something here? But is seems unnecessary.
            return

        if not self.bulk_modulus:
            pu.reportError(iam, "bulk modulus not defined")

        if not self.shear_modulus:
            pu.reportError(iam, "shear modulus not defined")

        if self.J0 is None:
            self.compute_init_jacobian()

        if not any(x for y in self.J0 for x in y):
            pu.reportError(iam, "iniatial Jacobian is empty")

        matdat.register_data("jacobian", "Matrix", init_val=self.J0)
        matdat.register_option("efield sim", self.electric_field_model)

        return

    def register_parameter(self, param_name, param_idx,
                           aliases=None, parseable=True, default=0.,
                           description="No description available"):

        if aliases is None:
            aliases = []

        iam = self.name + ".registerParameter"
        if not isinstance(param_name, str):
            pu.reportError(iam,"parameter name must be a string, got {0}"
                        .format(param_name))

        if not isinstance(param_idx, int):
            pu.reportError(iam,"parameter index must be an int, got {0}"
                        .format(param_idx))

        if not isinstance(aliases, list):
            pu.reportError(iam,"aliases must be a list, got {0}".format(aliases))
            pass

        # register full name, low case name, and aliases
        full_name = param_name
        param_name = param_name.lower().replace(" ","_")
        param_names = [param_name]
        param_names.extend([x.lower().replace(" ","_") for x in aliases])

        dupl_name = [x for x in param_names
                     if x in self.registered_params_and_aliases]

        if dupl_name:
            pu.reportWarning(iam,"duplicate parameter names: {0}"
                          .format(", ".join(param_names)))
            self.errors += 1

        if param_idx in self.registered_param_idxs:
            pu.reportWarning(iam,"duplicate ui location [{0}] in parameter table"
                          .format(", ".join(param_names)))
            self.errors += 1

        self.registered_params.append(full_name)
        self.registered_params_and_aliases.extend(param_names)
        self.registered_param_idxs.append(param_idx)

        # populate parameter_table
        self.parameter_table[full_name] = {"name": full_name,
                                           "names": param_names,
                                           "ui pos": param_idx,
                                           "parseable": parseable,
                                           "default value": default,
                                           "description": description,}
        self.parameter_table_idx_map[param_idx] = full_name
        return

    def get_parameter_names_and_values(self, default=True):
        """Returns a 2D list of names and values
               [ ["name",  "description",  val],
                 ["name2", "description2", val2],
                 [...] ]
        """
        table = [None] * self.nprop
        for param, param_dict in self.parameter_table.items():
            idx = param_dict["ui pos"]
            name = param_dict["name"]
            val = param_dict["default value"] if default else self.ui[idx]
            desc = param_dict["description"]
            table[idx] = [name, desc, val]
        return table

    def get_parameter_names(self, aliases=False):
        """Returns a list of parameter names, and optionally aliases"""
        param_names = [None] * self.nprop
        for p, pdict in self.parameter_table.items():
            idx = pdict["ui pos"]
            param_names[idx] = pdict["names"] if aliases else pdict["name"]
        return param_names

    def parse_parameters(self, *args):
        """ populate the materials ui array from the self.params dict """

        iam = self.name + ".parse_parameters"

        self.ui0 = np.zeros(self.nprop)
        self.ui = np.zeros(self.nprop)
        for param, param_dict in self.parameter_table.items():
            idx = param_dict["ui pos"]
            self.ui0[idx] = param_dict["default value"]
            self.ui[idx] = param_dict["default value"]
            continue

        # we have the name and value, now we need to get its position in the
        # ui array
        ignored, not_parseable = [], []
        for param, param_val in self.user_input_params.items():
            param_nam = None
            if param in self.parameter_table:
                param_nam = param

            else:
                # look for alias
                for key, val in self.parameter_table.items():
                    if param.lower() in val["names"]:
                        param_nam = key
                        break
                    continue

            if param_nam is None:
                ignored.append(param)
                continue

            if not self.parameter_table[param_nam]["parseable"]:
                not_parseable.append(param)
                continue

            ui_idx = self.parameter_table[param_nam]["ui pos"]
            self.ui0[ui_idx] = param_val
            continue

        if ignored:
            pu.reportWarning(iam,
                             "ignoring unregistered parameters: {0}"
                             .format(", ".join(ignored)))
        if not_parseable:
            pu.reportWarning(iam,
                             "ignoring unparseable parameters: {0}"
                             .format(", ".join(not_parseable)))

        return

    def _parse_user_params(self, user_params):
        """ read the user params and populate user_input_params """
        iam = self.name + "._parse_user_params"

        if not user_params:
            pu.reportError(iam, "no parameters found")
            return 1

        errors = 0
        for line in user_params:
            # strip and lowcase the line
            line = line.strip().lower()

            # replace "=" and "," with spaces and split
            for char in "=,":
                line = line.replace(char, " ")
                continue
            line = line.split()

            # look for user specified materials from the material and matlabel
            # shortcuts

            # @mswan{ -> the "material" part is what I had done before. The
            #            "matlabel" stuff would be new, if implemented
            if line[0] == "material" or line[0] == "matlabel":
                if self.mtldat_f is None:
                    msg = ("requested matlabel but "+ self.name +
                           " does not provide a material data file")
                    pu.reportError(iam, msg)

                try:
                    matlabel = line[1]
                except IndexError:
                    pu.reportError(iam, "empty matlabel encountered")

                # matlabel found, now parse the file for names and values
                mtldat = self._parse_mtldb_file(matlabel)
                for name, val in mtldat:
                    self.user_input_params[name] = val
                    continue

                continue
            # }@mswan

            # the line is now of form
            #     line = [string, string, ..., string]
            # we assume that the last string is the value and anything up
            # to the last is the name
            name = "_".join(line[0:-1])
            val = line[-1]
            try:
                val = float(val)
            except:
                errors += 1
                msg = ("could not convert {0} for parameter {1} to float"
                       .format(val, name))
                pu.reportWarning(iam, msg)
                continue

            self.user_input_params[name] = val
            continue

        if errors:
            pu.reportError(iam, "stopping due to previous errors")

        return

    # @mswan{
    def _parse_mtldb_file(self, material):
        """Parse the material database file

        Parameters
        ----------
        material : str
          name of material

        Returns
        -------
        mtldat : list
          list of tuples of (name, val) pairs

        """
        iam = self.name + "._parse_mtldb_file"

        fnam, fext = os.path.splitext(self.mtldat_f)
        if fext == ".py":
            py_mod, py_path = pu.get_module_name_and_path(self.mtldat_f)
            fobj, pathname, description = imp.find_module(py_mod, py_path)
            py_module = imp.load_module(py_mod, fobj, pathname, description)
            fobj.close()

            __all__ = getattr(py_module, "__all__")
            if __all__ is None or not isinstance(__all__, dict):
                pu.reportError(iam,
                               ("__all__ attribute in {0} not defined"
                                .format(self.mtldat_f)))

            # look for name of material in file
            mtl_nam = None
            if material in __all__:
                mtl_nam = material
            else:
                for name, aliases in __all__.items():
                    if material in aliases:
                        mtl_nam = name
                        break
                    continue
            if mtl_nam is None:
                pu.reportError(iam, ("material {0} not found in {1}"
                                     .format(material, self.mtldat_f)))

            params = getattr(py_module, mtl_nam)
            mtldat = []
            for key, val in params.items():
                if key.lower() == "units":
                    continue
                mtldat.append((key, float(val)))
                continue
        else:
            reportError(iam, "mtldat file parsing not enabled for this file type")

        return mtldat

    # @mswan}

    def initialize_state(self, material_data):
        pass

    def compute_init_jacobian(self, simdat=None, matdat=None, isotropic=True):
        '''
        NAME
           compute_init_jacobian

        PURPOSE
           compute the initial material Jacobian J = dsig/deps, assuming
           isotropy
        '''
        if isotropic:
            J0 = np.zeros((6, 6))
            threek, twog = 3. * self.bulk_modulus, 2. * self.shear_modulus
            pr = (threek - twog) / (2. * threek + twog)
            c1, c2 = (1 - pr) / (1 + pr), pr / (1 + pr)

            # set diagonal
            for i in range(3):
                J0[i,i] = threek * c1
            for i in range(3, 6):
                J0[i,i] = twog

            # off diagonal
            (          J0[0, 1], J0[0, 2],
             J0[1, 0],           J0[1, 2],
             J0[2, 0], J0[2, 1]           ) = [threek * c2] * 6
            self.J0 = np.array(J0)

            return

        else:
            # material is not isotropic, numerically compute the jacobian
            matdat.stash_data("prescribed stress components")
            matdat.store_data("prescribed stress components",[0,1,2,3,4,5])
            self.J0 = self.jacobian(simdat, matdat)
            matdat.unstash_data("prescribed stress components")
            return

        return


    def jacobian(self, simdat, matdat):
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
        v = matdat.get_data("prescribed stress components")
        nv = len(v)
        deps,Jsub = math.sqrt(epsilon),np.zeros((nv,nv))

        dt = simdat.get_data("time step")
        d = matdat.get_data("rate of deformation")
        Fold = matdat.get_data("deformation gradient",form="Matrix")
        dtime = 1 if dt == 0. else dt

        # stash the data
        simdat.stash_data("time step")
        matdat.stash_data("rate of deformation")
        matdat.stash_data("deformation gradient")

        for n in range(nv):
            # perturb forward
            dp = np.array(d)
            dp[v[n]] = d[v[n]] + (deps/dtime)/2.
            fp = Fold + np.dot(pt.to_matrix(dp),Fold)*dtime
            matdat.store_data("rate of deformation",dp,old=True)
            matdat.store_data("deformation gradient",fp,old=True)
            self.update_state(simdat, matdat)
            sigp = matdat.get_data("stress",cur=True)

            # perturb backward
            dm = np.array(d)
            dm[v[n]] = d[v[n]] - (deps/dtime)/2.
            fm = Fold + np.dot(pt.to_matrix(dm),Fold)*dtime
            matdat.store_data("rate of deformation",dm,old=True)
            matdat.store_data("deformation gradient",fm,old=True)
            self.update_state(simdat, matdat)
            sigm = matdat.get_data("stress",cur=True)

            Jsub[n,:] = (sigp[v] - sigm[v])/deps
            continue

        # restore data
        simdat.unstash_data("time step")
        matdat.unstash_data("deformation gradient")
        matdat.unstash_data("rate of deformation")

        return Jsub

    # The methods below are going to be depricated in favor of
    # under_score_separated names. We keep them here for compatibility.
    def parseParameters(self, *args):
        self.parse_parameters(*args)
        return
    def setUp(self, *args, **kwargs):
        """ set up model """
        self.set_up(*args, **kwargs)
        return
    def updateState(self, *args):
        self.update_state(*args)
        return
    def registerParameter(self, *args):
        self.register_parameter(*args)
        return



