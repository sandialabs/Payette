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

import sys
import os
import xml.dom as xmldom
import xml.dom.minidom as xdom
from textwrap import fill as textfill

import Payette_utils as pu

class NotTextNodeError:
    pass

class XMLParserError(Exception):
    def __init__(self, message):
        caller = pu.who_is_calling()
        self.message = message + " [reported by {0}]".format(caller)
        super(XMLParserError, self).__init__(self.message)


class XMLParser:
    """
    CLASS NAME
       XMLParser

    PURPOSE
       To parse material paramter databases and provide information about
       default material parameters and model parameterizations for different
       materials.

    AUTHORS
       Scot Swan, Sandia National Laboratories, mswan@sandia.gov
    """
    def __init__(self, file_name):
        """Parse an xml file that contains material parameter information.

        Parameters
        ----------
        file_name : str
          path to xml material database file

        Returns
        -------
        mm : dict
          The 'root' keys of this nested dictionary are:
            'Units' : str
              Unit system (eg. 'CGSEV' 'SI'...)
            'Materials': dict
              Contains all information about paramterized materials. The
              dictionary contains str(key):str(value) pairs of the possible
              parameters as well as the 'name' and 'description' keys.
            'Name' : str
              Model name for which the xml file is meant for.
            'Parameters' : list of dicts
              Eact dictionary in the list contains information about the
              parameters used in the material model. All keys and values are
              strings. Each point should have the following keys:
                'default': default parameter value
                'units'  : Unit type (such as DENSITY_UNITS)
                'type'   : data type (float, double, str)
                'name'   : paramter name
                'order'  : index of the parameter in the UI array.
            'Description' : str
              Description of the material model the file is meant for.
        """
        file_name = os.path.realpath(file_name)
        if not os.path.isfile(file_name):
            raise XMLParserError(
                "Cannot parse file '{0}' because it does not exist."
                .format(file_name))
        self.file = file_name
        self.fdir = os.path.dirname(self.file)

        dom = xdom.parse(self.file)

        # Get the root element (Should always be "MaterialModel")
        MaterialModel = dom.getElementsByTagName('MaterialModel')
        if not MaterialModel:
            raise XMLParserError("Expected Root Element 'MaterialModel'")
        self.MaterialModel = MaterialModel[0]

        # Get the Name and Description of the model.
        Name = self.MaterialModel.getElementsByTagName('Name')
        if not Name:
            raise XMLParserError("Expected a 'Name' for the 'MaterialModel'.")
        self.name = str(Name[0].firstChild.data).strip()

        Description = self.MaterialModel.getElementsByTagName('Description')
        if not Description:
            raise XMLParserError(
                "Expected a 'Description' for the 'MaterialModel'.")
        self.description = str(Description[0].firstChild.data).strip()

        #
        # Get the ModelParameters block.
        #
        ModelParameters = dom.getElementsByTagName('ModelParameters')
        if len(ModelParameters) != 1:
            raise XMLParserError("Expected Element 'ModelParameters'")
        self.ModelParameters = ModelParameters[0]

        tmp = self.ModelParameters.getElementsByTagName('Units')
        if len(tmp) != 1:
            raise XMLParserError(
                "Expected a 'Units' in the 'ModelParameters' element.")
        self.units_system = str(tmp[0].firstChild.data).strip()

        #
        # Get the Model Parameters (Default).
        #
        self.parameters = []
        Parameters = self.ModelParameters.getElementsByTagName('Parameter')
        for parameter in Parameters:
            # If you just want to loop through the attributes:
            tmp = {"aliases": None, "parseable": True}
            for idx in range(0, parameter.attributes.length):
                mname = str(parameter.attributes.item(idx).name).strip()
                mval = str(parameter.attributes.item(idx).value).strip()
                tmp[mname] = mval
            if tmp["aliases"] is not None:
                tmp["aliases"] = [x.strip() for x in tmp["aliases"].split(",")]
            self.parameters.append(tmp)

        #
        # Get the Material Parameterizations.
        #
        self.materials = []
        self.matnames_and_aliases = []
        material_list = self.ModelParameters.getElementsByTagName('Material')
        for material in material_list:
            # If you just want to loop through the attributes:
            tmp = {}
            try:
                tmp["description"] = str(material.firstChild.nodeValue).strip()
            except AttributeError:
                tmp["description"] = None

            for idx in range(0, material.attributes.length):
                mname = str(material.attributes.item(idx).name).strip()
                mval = str(material.attributes.item(idx).value).strip()
                tmp[mname] = mval
                continue

            if "name" not in tmp:
                raise XMLParserError(
                    "no name given for a material in {0}"
                    .format(os.path.basename(self.file_name)))

            self.materials.append(tmp)

            # save the name and aliases
            name = tmp.get("name")
            aliases = [name.lower()]
            aliases.extend([x.lower() for x in
                            tmp.get("aliases", "").split(",") if x])
            self.matnames_and_aliases.append((name, aliases))
            continue

        pass

    def get_parameterized_materials(self):
        """return the available materials as parsed by parse_xml()

        Parameters
        ----------

        Returns
        -------
        mtldat : list
          list of tuples of (name, description) pairs for parameterized materials
        """
        iam = "get_parameterized_materials"

        mtldat = []
        for mat in self.materials:
            mtldat.append([mat["name"], mat["description"]])
        return mtldat


    def get_material_parameterization(self, mat_name):
        """return the material parameterization for a given material name

        Parameters
        ----------
        mat_name : str
          name of the material to have the paramterization returned

        Returns
        -------
        mtldat : list
          list of tuples of (param_name, param_val) pairs for parameterized
          materials. The variable 'param_val' will be of whatever type the
          xml file says that it will be (likely all will be converted to float).
        """
        iam = "get_material_parameterizations"

        # Every set of inputs needs a unit system.
        mtldat = [("Units", self.units_system)]

        for name, aliases in self.matnames_and_aliases:
            mat_names = [x.lower()
                         for x in (mat_name, mat_name.replace(" ", "_"))]
            if any(x in mat_names for x in aliases):
                mat_name = name
                material = [x for x in self.materials if x["name"] == name][0]
                break
        else:
            raise XMLParserError(
                "Material '{0}' not in '{1}', available materials are:\n{2}"
                .format(mat_name, os.path.basename(self.file),
                        textfill(", ".join([x["name"] for x in self.materials])))
                )

        for param in self.parameters:
            param_name = param["name"]
            param_val = material.get(param_name)
            if param_val is None:
                param_val = param["default"]
            if param["type"] == "double":
                param_val = float(param_val)
            mtldat.append((param_name, param_val))

        return mtldat

    def get_parameters(self):
        """Return the parameters"""
        return self.parameters

    def get_sorted_parameters(self):
        return sorted(self.get_parameters(),
                      key=lambda x: int(x["order"]))

    def get_core_files(self):
        """Parse MaterialModel for "Core" files"""
        Files = self.MaterialModel.getElementsByTagName('Files')
        if not Files:
            raise XMLParserError(
                "Expected 'Files' for the 'MaterialModel'.")
        Core = Files[0].getElementsByTagName("Core")
        core_files = [
            os.path.realpath(os.path.join(self.fdir, x.strip()))
            for x in Core[0].firstChild.data.split()]
        return core_files

    def get_payette_interface_files(self):
        """Parse MaterialModel for "Interface" files"""
        Files = self.MaterialModel.getElementsByTagName('Files')
        if not Files:
            raise XMLParserError(
                "Expected 'Files' for the 'MaterialModel'.")
        Interface =  Files[0].getElementsByTagName("Interface")
        for idx, item in enumerate(Interface):
            if item.attributes.item(0).value.lower() == "payette":
                payette_interface = Interface[idx]
                break
            continue
        else:
            # not a Payette interface
            return None

        payette_interface_files = [
            os.path.realpath(os.path.join(self.fdir, x.strip()))
            for x in payette_interface.firstChild.data.split()]

        return payette_interface_files

    def get_payette_build_info(self):
        """Parse MaterialModel for information needed by Payette build system"""

        # Material model files
        payette_interface = self.get_payette_interface_files()
        if payette_interface is None:
            return None
        for item in payette_interface:
            if not os.path.isfile(item):
                raise XMLParserError(
                    "payette interface file {0} not found".format(item))
            continue

        key, aliases, mat_type, src_types = self.get_payette_info()

        return key, aliases, mat_type, payette_interface, src_types

    def get_payette_info(self):
        """Parse MaterialModel for information needed by Payette"""

        # Material type
        Type = self.MaterialModel.getElementsByTagName("Type")
        if not Type:
            material_type = None
#            raise XMLParserError(
#                "Expected a 'Type' for the 'MaterialModel'.")
        else:
            material_type = str(Type[0].firstChild.data).strip()

        # model keyword
        Key = self.ModelParameters.getElementsByTagName('Key')
        if not Key:
            raise XMLParserError(
                "Expected a 'Key' in the 'ModelParameters' element.")
        model_key = str(Key[0].firstChild.data).strip()

        # model keyword aliases
        Aliases = self.ModelParameters.getElementsByTagName('Aliases')
        if not Aliases:
            model_aliases = []
        elif Aliases[0].firstChild is None:
            model_aliases = []
        else:
            model_aliases = [x.strip().replace(" ", "_")
                             for x in Aliases[0].firstChild.data.split(",")]

        core_files = self.get_core_files()
        source_types = []
        for core_file in reversed(core_files):
            if core_file.endswith((".h", ".H", ".Blk")):
                continue
            elif core_file.endswith((".f90", ".f", ".F")):
                source_types.append("fortran")
            elif core_file.endswith(".py"):
                source_types.append("python")
            else:
                source_types.append(os.path.splitext(core_file)[1])

        return model_key, model_aliases, material_type, source_types

