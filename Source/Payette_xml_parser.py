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
import xml.dom.minidom as xdom


def get_parameterized_materials(mtldat_f):
    """return the available materials as parsed by parse_xml()

    Parameters
    ----------
    mtldat_f : str
      path to xml material database file

    Returns
    -------
    mtldat : list
      list of tuples of (name, description) pairs for parameterized materials
    """
    iam = "get_parameterized_materials"

    # Load the material database file
    mm = parse_xml(mtldat_f)

    mtldat = []
    for mat in mm["Materials"]:
        mtldat.append([mat["name"], mat["description"]])
    return mtldat


def get_material_parameterization(mtldat_f, mat_name):
    """return the material parameterization for a given material name

    Parameters
    ----------
    mtldat_f : str
      path to xml material database file
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

    # Load the material database file
    mm = parse_xml(mtldat_f)

    # Every set of inputs needs a unit system.
    mtldat = [("Units", mm["Units"])]

    for mat in mm["Materials"]:
        if mat["name"].lower() == mat_name.lower():
            break
    else:
        sys.exit("Material '{0}' not found in '{1}'".
                 format(material, mtldat_f))

    for param in mm["Parameters"]:
        param_name = param["name"]
        param_val = mat.get(param_name)
        if param_val is None:
            param_val = param["default"]
        if param["type"] == "double":
            param_val = float(param_val)
        mtldat.append((param_name, param_val))

    return mtldat


def parse_xml(file_name):
    """Parse an xml file that contains material parameter information.

    Parameters
    ----------
    mtldat_f : str
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
    iam = "get_material_parameterizations"


    dom = xdom.parse(file_name)
    mm = {}

    # Get the root element (Should always be "MaterialModel")
    MaterialModel = dom.getElementsByTagName('MaterialModel')
    if len(MaterialModel) != 1:
        sys.exit("Expected Root Element 'MaterialModel'")
    MaterialModel = MaterialModel[0]

    #
    # Get the Name and Description of the model.
    #
    tmp = MaterialModel.getElementsByTagName('Name')
    if len(tmp) != 1:
        sys.exit("Expected a 'Name' for the 'MaterialModel'.")
    mm["Name"] = str(tmp[0].firstChild.data).strip()

    tmp = MaterialModel.getElementsByTagName('Description')
    if len(tmp) != 1:
        sys.exit("Expected a 'Description' for the 'MaterialModel'.")
    mm["Description"] = str(tmp[0].firstChild.data).strip()

    #
    # Get the ModelParameters block.
    #
    ModelParameters = dom.getElementsByTagName('ModelParameters')
    if len(ModelParameters) != 1:
        sys.exit("Expected Element 'ModelParameters'")
    ModelParameters = ModelParameters[0]

    tmp = ModelParameters.getElementsByTagName('Units')
    if len(tmp) != 1:
        sys.exit("Expected a 'Units' in the 'ModelParameters' element.")
    mm["Units"] = str(tmp[0].firstChild.data).strip()

    #
    # Get the Model Parameters (Default).
    #
    mm["Parameters"] = []
    Parameters = ModelParameters.getElementsByTagName('Parameter')
    for parameter in Parameters:
        # If you just want to loop through the attributes:
        tmp = {}
        for idx in range(0, parameter.attributes.length):
            mname = str(parameter.attributes.item(idx).name).strip()
            mval = str(parameter.attributes.item(idx).value).strip()
            tmp[mname] = mval
        mm["Parameters"].append(tmp)

    #
    # Get the Material Parameterizations.
    #
    mm["Materials"] = []
    material_list = ModelParameters.getElementsByTagName('Material')
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
        mm["Materials"].append(tmp)

    return mm
