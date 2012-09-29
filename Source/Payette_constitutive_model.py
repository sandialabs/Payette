"""Main constitutive model class file."""

import os, imp, re, warnings
import math
import numpy as np

import Source.__runopts__ as ro
import Source.Payette_tensor as pt
import Source.Payette_utils as pu
import Source.Payette_xml_parser as px
from Source.Payette_xml_parser import XMLParserError as XMLParserError
from Source.Payette_unit_manager import UnitManager as UnitManager
from Source.Payette_input_parser import I_EQ


class ConstitutiveModelPrototype(object):
    """Prototype class from which constutive models are derived

    Public Methods
    --------------
    __init__
    set_up
    update_state
    finish_setup
    register_parameter
    get_parameter_names_and_values
    get_parameter_names
    parse_parameters
    parse_user_params
    initialize_state
    compute_init_jacobian
    jacobian

    """

    def __init__(self, control_file, *args, **kwargs):
        """Initialize the ConstitutiveModelPrototype object.

        The __init__ functions should be called by each constituve model that
        derives from the ConstitutiveModelPrototype base class.

        Parameters
        ----------
        control file : str
          path to the material control file
        *args : list, optional
        **kwargs : dict, optional
          keys
          code : string
            code type specified by user

        """

        # location of control file - if any
        self.control_file = control_file

        # read in the control file
        self.xml_obj = px.XMLParser(self.control_file)
        info = self.xml_obj.get_payette_info()
        self.name, self.aliases, self.material_type, source_types = info

        if not isinstance(self.aliases, (list, tuple)):
            self.aliases = [self.aliases]

        # which code to use
        if not isinstance(source_types, (list, tuple)):
            source_types = [source_types]
        self.code = kwargs.get("code")
        if self.code is None:
            self.code = source_types[0]
        elif self.code not in source_types:
            pu.report_and_raise_error(
                "requested code type {0} not supported by {1}"
                .format(self.code, self.name))

        pu.log_message("using {0} implementation of the '{1}' constitutive model"
                       .format(self.code, self.name))

        # specialized models
        if self.material_type is None:
            self.material_type = "eos"
        self.electric_field_model = "electromechanical" in self.material_type.lower()
        self.eos_model = "eos" in self.material_type.lower()

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
        self.ui0 = np.zeros(self.nprop)
        self.ui = np.zeros(self.nprop)
        self.dc = np.zeros(self.ndc)
        self.bulk_modulus = 0.
        self.shear_modulus = 0.

        pass

    def set_up(self, *args):
        """ set up model """
        pu.report_and_raise_error(
            'Constitutive model must provide set_up method')
        return

    def update_state(self, *args):
        """ update state """
        pu.report_and_raise_error(
            'Constitutive model must provide update_state method')
        return

    def finish_setup(self, matdat):
        """ check that model is properly set up """

        if pu.error_count():
            pu.report_and_raise_error("previously encountered parsing errors")

        if not any(self.ui):
            pu.report_and_raise_error("empty ui array")

        if self.eos_model:
            # mss: do something here? But is seems unnecessary.
            return

        if not self.bulk_modulus:
            pu.report_and_raise_error("bulk modulus not defined")

        if not self.shear_modulus:
            pu.report_and_raise_error("shear modulus not defined")

        if self.J0 is None:
            self.compute_init_jacobian()

        if not any(x for y in self.J0 for x in y):
            pu.report_and_raise_error("iniatial Jacobian is empty")

        matdat.register_data("jacobian", "Matrix", init_val=self.J0,
                             units="NO_UNITS")
        ro.set_global_option("EFIELD_SIM", self.electric_field_model)

        return

    def register_parameters_from_control_file(self):
        """Register parameters from the control file """
        params = self.xml_obj.get_sorted_parameters()
        for idx, pm in enumerate(params):
            self.register_parameter(
                pm["name"], idx, aliases=pm["aliases"],
                default=pm["default"], parseable=pm["parseable"],
                units=pm["units"])
            continue
        return

    def register_parameter(self, param_name, param_idx, aliases=None,
                           parseable=True, default=0., units=None,
                           description="No description available"):
        """Register parameters from the material model to the consitutive
        model base class

        Parameters
        ----------
        param_name : str
          paramter name
        param_idx : int
          index of where the parameter is located in the material user input
          array
        aliases : list
          list of aliases
        parseable : bool
          Boolean of whether the user input is read in from the input file
        default : float
          default value
        description : str
          long description of parameter

        """

        if aliases is None:
            aliases = []

        if units is None or not UnitManager.is_valid_units(units):
            pu.report_and_raise_error(
                "Units '{0}' for {1} is not valid.'".format(units, param_name))

        if not isinstance(param_name, str):
            pu.report_and_raise_error(
                "parameter name must be a string, got {0}".format(param_name))

        if not isinstance(param_idx, int):
            pu.report_and_raise_error(
                "parameter index must be an int, got {0}".format(param_idx))

        if not isinstance(aliases, list):
            pu.report_and_raise_error(
                "aliases must be a list, got {0}".format(aliases))

        # register full name, low case name, and aliases
        full_name = param_name
        param_name = param_name.lower().replace(" ","_")
        param_names = [param_name]
        param_names.extend([x.lower().replace(" ","_") for x in aliases])

        dupl_name = [x for x in param_names
                     if x in self.registered_params_and_aliases]

        if dupl_name:
            pu.report_error(
                "duplicate parameter names: {0}".format(", ".join(param_names)))

        if param_idx in self.registered_param_idxs:
            pu.report_error(
                "duplicate ui location [{0}] in parameter table"
                .format(", ".join(param_names)))

        self.registered_params.append(full_name)
        self.registered_params_and_aliases.extend(param_names)
        self.registered_param_idxs.append(param_idx)

        # populate parameter_table
        self.parameter_table[full_name] = {"name": full_name,
                                           "names": param_names,
                                           "ui pos": param_idx,
                                           "units": units,
                                           "parseable": parseable,
                                           "default value": default,
                                           "description": description,}
        self.parameter_table_idx_map[param_idx] = full_name

        self.nprop += 1

        return

    def ensure_all_parameters_have_valid_units(self):
        """Returns nothing if all parameters have valid units.
        Fails otherwise."""
        for param, param_dict in self.parameter_table.items():
            if not UnitManager.is_valid_units(param_dict['units']):
                pu.report_and_raise_error(
      "Parameter does not have valid units set:\n"+
      "\n".join(["({0}:{1})".format(x, y) for x, y in param_dict.iteritems()]))
        return

    def get_parameter_names_and_values(self, default=True, version=None):
        """Returns a 2D list of names and values
               [ ["name",  "description",  val],
                 ["name2", "description2", val2],
                 [...] ]
        """
        table = [None] * self.nprop
        if version is None:
            if default:
                version = "default"
            else:
                version = "modified"

        if version not in ("default", "unmodified", "modified",):
            pu.report_and_raise_error(
                "unrecognized version {0}".format(version))

        for param, param_dict in self.parameter_table.items():
            idx = param_dict["ui pos"]
            if version == "default":
                val = param_dict["default value"]
            elif version == "modified":
                val = self.ui[idx]
            else:
                val = self.ui0[idx]

            desc = param_dict["description"]
            table[idx] = [param, desc, val]
        return table

    def parameter_index(self, name):
        for key, val in self.parameter_table.items():
            if name.lower() in val["names"]:
                return val["ui pos"]
        else:
            return None

    def get_parameter_names(self, aliases=False):
        """Returns a list of parameter names, and optionally aliases"""
        param_names = [None] * self.nprop
        for param, pdict in self.parameter_table.items():
            idx = pdict["ui pos"]
            param_names[idx] = pdict["names"] if aliases else param
        return param_names

    def parse_parameters(self, *args):
        """ populate the materials ui array from the self.params dict """

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
            # %tjf: ignore units for now
            if param.lower() == "units":
                continue

            for key, val in self.parameter_table.items():
                if param.lower() in [param] + val["names"]:
                    param_name = key
                    break
                continue

            else:
                ignored.append(param)
                continue

            if not self.parameter_table[param_name]["parseable"]:
                not_parseable.append(param)
                continue

            ui_idx = self.parameter_table[param_name]["ui pos"]
            self.ui0[ui_idx] = param_val
            continue

        if ignored:
            pu.log_warning(
                "ignoring unregistered parameters: {0}"
                .format(", ".join(ignored)))
        if not_parseable:
            pu.log_warning("ignoring unparseable parameters: {0}"
                           .format(", ".join(not_parseable)))

        return

    def parse_user_params(self, user_params):
        """ read the user params and populate user_input_params """
        if not user_params:
            pu.report_and_raise_error("no parameters found")
            return 1

        errors = 0
        for line in user_params.split("\n"):
            line = re.sub(I_EQ, " ", line)

            # look for user specified materials from the material and matlabel
            # shortcuts
            matlabel = re.search(r"(?i)\bmaterial\s|\bmatlabel\s", line)
            if matlabel:
                if self.control_file is None:
                    pu.report_and_raise_error(
                        "Requested matlabel but {0} does not provide a "
                        "material data file".format(self.name))

                matlabel = line[matlabel.end():].strip()
                if not matlabel:
                    pu.report_and_raise_error("Empty matlabel encountered")

                # matlabel found, now parse the file for names and values
                matdat = self.parse_mtldb_file(material=matlabel)
                for name, val in matdat:
                    self.user_input_params[name.upper()] = val
                    continue

                continue

            line = line.split()
            name, val = "_".join(line[:-1]), line[-1]
            try:
                # Horrible band-aid for poorly formatted fortran output.
                # when it meant 1.0E+100, it spit out 1.0+100
                if val.endswith("+100"):
                    val = float(val.replace("+100", "E+100"))
                else:
                    val = float(val)
            except ValueError:
                errors += 1
                pu.log_warning(
                    "could not convert {0} for parameter {1} to float"
                    .format(val, name))
                continue

            self.user_input_params[name.upper()] = val
            continue

        if errors:
            pu.report_and_raise_error("stopping due to previous errors")

        return

    def initialize_state(self, material_data):
        """initialize the material state"""
        pass

    def compute_init_jacobian(self, simdat=None, matdat=None, isotropic=True):
        '''
        NAME
           compute_init_jacobian

        PURPOSE
           compute the initial material Jacobian J = dsig / deps, assuming
           isotropy
        '''
        if isotropic:
            j_0 = np.zeros((6, 6))
            threek, twog = 3. * self.bulk_modulus, 2. * self.shear_modulus
            poissons = (threek - twog) / (2. * threek + twog)
            const_1 = (1 - poissons) / (1 + poissons)
            const_2 = poissons / (1 + poissons)

            # set diagonal
            for i in range(3):
                j_0[i, i] = threek * const_1
            for i in range(3, 6):
                j_0[i, i] = twog

            # off diagonal
            (          j_0[0, 1], j_0[0, 2],
             j_0[1, 0],           j_0[1, 2],
             j_0[2, 0], j_0[2, 1]           ) = [threek * const_2] * 6
            self.J0 = np.array(j_0)

            return

        else:
            # material is not isotropic, numerically compute the jacobian
            matdat.stash_data("prescribed stress components")
            matdat.store_data("prescribed stress components",
                              [0, 1, 2, 3, 4, 5])
            self.J0 = self.jacobian(simdat, matdat)
            matdat.unstash_data("prescribed stress components")
            return

        return


    def jacobian(self, simdat, matdat):
        """Numerically compute and return a specified submatrix, j_sub, of the
        Jacobian matrix J = J_ij = dsigi / depsj.

        Parameters
        ----------
        simdat : object
          simulation data container
        matdat : object
          material data container

        Returns
        -------
        j_sub : array_like
          Jacobian of the deformation J = dsig / deps

        Notes
        -----
        The submatrix returned is the one formed by the intersections of the
        rows and columns specified in the vector subscript array, v. That is,
        j_sub = J[v, v]. The physical array containing this submatrix is
        assumed to be dimensioned j_sub[nv, nv], where nv is the number of
        elements in v. Note that in the special case v = [1,2,3,4,5,6], with
        nv = 6, the matrix that is returned is the full Jacobian matrix, J.

        The components of j_sub are computed numerically using a centered
        differencing scheme which requires two calls to the material model
        subroutine for each element of v. The centering is about the point eps
        = epsold + d * dt, where d is the rate-of-strain array.

        History
        -------
        This subroutine is a python implementation of a routine by the same
        name in Tom Pucick's MMD driver.

        Authors
        -------
        Tom Pucick, original fortran implementation in the MMD driver
        Tim Fuller, Sandial National Laboratories, tjfulle@sandia.gov
        """

        # local variables
        epsilon = np.finfo(np.float).eps
        nzc = matdat.get_data("prescribed stress components")
        l_nzc = len(nzc)
        deps, j_sub = math.sqrt(epsilon), np.zeros((l_nzc, l_nzc))

        delt = simdat.get_data("time step")
        delt = 1 if delt == 0. else delt
        sym_velgrad = matdat.get_data("rate of deformation")
        f_old = matdat.get_data("deformation gradient", form="Matrix")

        # stash the data
        simdat.stash_data("time step")
        matdat.stash_data("rate of deformation")
        matdat.stash_data("deformation gradient")

        for inum in range(l_nzc):
            # perturb forward
            sym_velgrad_p = np.array(sym_velgrad)
            sym_velgrad_p[nzc[inum]] = sym_velgrad[nzc[inum]] + (deps / delt) / 2.
            defgrad_p = (f_old +
                         np.dot(pt.to_matrix(sym_velgrad_p), f_old) * delt)
            matdat.store_data("rate of deformation", sym_velgrad_p, old=True)
            matdat.store_data("deformation gradient", defgrad_p, old=True)
            self.update_state(simdat, matdat)
            sigp = matdat.get_data("stress", cur=True)

            # perturb backward
            sym_velgrad_m = np.array(sym_velgrad)
            sym_velgrad_m[nzc[inum]] = sym_velgrad[nzc[inum]] - (deps / delt) / 2.
            defgrad_m = (f_old +
                         np.dot(pt.to_matrix(sym_velgrad_m), f_old) * delt)
            matdat.store_data("rate of deformation", sym_velgrad_m, old=True)
            matdat.store_data("deformation gradient", defgrad_m, old=True)
            self.update_state(simdat, matdat)
            sigm = matdat.get_data("stress", cur=True)

            j_sub[inum, :] = (sigp[nzc] - sigm[nzc]) / deps
            continue

        # restore data
        simdat.unstash_data("time step")
        matdat.unstash_data("deformation gradient")
        matdat.unstash_data("rate of deformation")

        return j_sub

    def parse_mtldb_file(self, material=None):
        """Parse the material database file

        Parameters
        ----------
        material : str, optional
          name of material

        Returns
        -------
        mtldat : list
          list of tuples of (name, val) pairs

        """
        import Source.Payette_xml_parser as px
        xml_obj = px.XMLParser(self.control_file)
        if material is None:
            try:
                mtldat = xml_obj.get_parameterized_materials()
            except XMLParserError as error:
                pu.report_and_raise_error(error.message)
        else:
            try:
                mtldat = xml_obj.get_material_parameterization(material)
            except XMLParserError as error:
                pu.report_and_raise_error(error.message)

        return mtldat

    def initial_density(self):
        """return the initial density"""
        for name, info in self.parameter_table.items():
            names = info["names"]
            if "density" in names and self.ui0[info["ui pos"]] > 0.:
                return self.ui0[info["ui pos"]]
            continue
        return 1.

    # The methods below are going to be depricated in favor of
    # under_score_separated names. We keep them here for compatibility.
    def parseParameters(self, *args):
        """docstring"""
        self.parse_parameters(*args)
        return
    def setUp(self, *args, **kwargs):
        """docstring"""
        self.set_up(*args, **kwargs)
        return
    def updateState(self, *args):
        """docstring"""
        self.update_state(*args)
        return
    def registerParameter(self, *args):
        """docstring"""
        self.register_parameter(*args)
        return
