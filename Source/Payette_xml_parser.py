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

import xml.dom.minidom as xdom

def parse_xml_mtldb_file(mtldat_f, material=None):
    """Parse the xml material database file

    Parameters
    ----------
    mtldat_f : str
      path to xml material database file

    material : str
      name of material

    Returns
    -------
    mtldat : list
      if matlabel not None:
          list of tuples of (name, val) pairs for material parameter values
      else:
          list of tuples of (name, [aliase1, alias2,...]) pairs for valid
          materials
    """
    iam = "parse_xml_mtldb_file"


    # Load the material database file
    dom = xdom.parse(mtldat_f)

    if material is None:
        # Get all the xml data in 'Material' tags and return the names
        input_sets = []
        MatList = dom.getElementsByTagName('Material')
        for mat in MatList:
            # Here we get the name and description, just in case
            # someone wants the description at some point in time.
            try:
                description = str(mat.firstChild.nodeValue)
            except AttributeError:
                description = None
            input_sets.append((str(mat.getAttribute('name')), []))
        return input_sets
    else:
        # Every set of inputs needs a unit system.
        Unit_system = str(dom.getElementsByTagName('Units')[0].firstChild.data)
        mat_params = [("Units", Unit_system)]

        # Load in the defaults because only the non-default parameters
        # are given in the XML file.
        ParamList = dom.getElementsByTagName('Parameter')
        for param in ParamList:
            param_name = param.getAttribute('name')
            param_default = param.getAttribute('default')
            mat_params.append((str(param_name), float(param_default)))

        # Cycle through all the materials looking for the one requested.
        MatList = dom.getElementsByTagName('Material')
        for mat in MatList:
            mat_name = str(mat.getAttribute('name'))
            if mat_name.lower() != material.lower():
                continue

            # Go through all the default parameters looking for a given
            # value for the specific material.
            for idx in range(1, len(mat_params)):
                param_name, param_val = mat_params[idx]
                val = mat.getAttribute(param_name)
                if len(val) > 0:
                    mat_params[idx] = (param_name, float(val))
            break
        return mat_params

