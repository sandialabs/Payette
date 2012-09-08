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
"""Contains classes and functions for writing index files for permutation and
optimization simulations

"""
import os, sys
import imp

try:
    import cPickle as pickle
except ImportError:
    import pickle

import Payette_config as pc
import Source.Payette_utils as pu
import Source.runopts as ro


class ModelIndexError(Exception):
    """ModelIndex exception class"""
    def __init__(self, message):
        if not ro.DEBUG:
            sys.tracebacklimit = 0
        caller = pu.who_is_calling()
        self.message = message + " [reported by {0}]".format(caller)
        super(ModelIndexError, self).__init__(self.message)


class ModelIndex(object):
    """Class for indexing installed materials

    """
    def __init__(self, index_file=None):
        """Initialize the ModelIndex object

        Parameters
        ----------
        self : class instance

        """

        # index file name
        if index_file is None:
            index_file = pc.PC_MTLS_FILE

        index_dir = os.path.dirname(index_file)
        if not os.path.isdir(index_dir):
            raise ModelIndexError(
                "Directory {0} must first be created".format(index_dir))
        self.index_file = index_file

        # initialize class data
        self._installed_constitutive_models = {}

        # load the index file if it exists
        if os.path.isfile(self.index_file):
            self.load()

    def remove_model(self, model):
        """remove model from the index"""
        try:
            del self._installed_constitutive_models[model]
        except KeyError:
            pass
        return

    def constitutive_models(self):
        """return the installed constitutive models dict"""
        return self._installed_constitutive_models.keys()

    def store(self, name, libname, clsname, intrfc, cntrl, aliases,
              param_file=None, param_cls=None):
        """Store all kwargs in to the index dict

        Parameters
        ----------
        name : str
          material name
        libname : str
          library name
        clsname : str
          material class name
        intrfc : str
          absolute path to interface file
        cntrl : str
          absolute path to .xml control file
        aliases : list
          list of alternative names for the model
        """

        self._installed_constitutive_models[name] = {
            "libname": libname, "class name": clsname,
            "interface file": intrfc, "control file": cntrl,
            "aliases": aliases,
            "parameterization file": param_file,
            "parameterization class": param_cls,}

        return

    def dump(self):
        """Dump self.constitutive_models to a file"""
        # dup the index file
        if pc.PC_ROOT in self.index_file:
            stubf = "PAYETTE_ROOT" + self.index_file.split(pc.PC_ROOT)[1]
        else:
            stubf = self.index_file
        sys.stdout.write(
            "\nINFO: writing constitutive model information to: {0}\n"
            .format(stubf))
        with open(self.index_file, "wb") as fobj:
            pickle.dump(self._installed_constitutive_models, fobj)
        sys.stdout.write("INFO: constitutive model information written\n")
        return

    def load(self):
        """Load the index file"""
        # check existence of file
        if not os.path.isfile(self.index_file):
            raise ModelIndexError(
                "buildPayette must be executed to generate {0}"
                .format(self.index_file))
        # load it in
        self._installed_constitutive_models = pickle.load(
            open(self.index_file, "rb"))
        return self._installed_constitutive_models

    def constitutive_model(self, model_name):
        """ get the constitutive model dictionary of model_name """
        for key, val in self._installed_constitutive_models.items():
            if model_name.lower() == key.lower() or model_name in val["aliases"]:
                constitutive_model = val
                break
            continue

        else:
            pu.report_and_raise_error(
                "constitutive model {0} not found, installed models are: {1}"
                .format(model_name, ", ".join(self.constitutive_models())))

        return constitutive_model

    def control_file(self, model_name):
        """ get the control file for the material """
        constitutive_model = self.constitutive_model(model_name)
        return constitutive_model.get("control file")

    def constitutive_model_object(self, model_name):
        """ get the actual model object """
        constitutive_model = self.constitutive_model(model_name)
        py_mod, py_path = pu.get_module_name_and_path(
            constitutive_model["interface file"])
        cls_nam = constitutive_model["class name"]
        fobj, pathname, description = imp.find_module(py_mod, py_path)
        py_module = imp.load_module(py_mod, fobj, pathname, description)
        fobj.close()
        cmod = getattr(py_module, cls_nam)
        del py_module
        return cmod

    def parameterization_file(self, model_name):
        """ get the parameterize file for the material """
        constitutive_model = self.constitutive_model(model_name)
        return constitutive_model.get("parameterization file")

    def parameterizer(self, model_name):
        """ get the actual model object """
        constitutive_model = self.constitutive_model(model_name)
        if constitutive_model["parameterization file"] is None:
            return None
        py_mod, py_path = pu.get_module_name_and_path(
            constitutive_model["parameterization file"])
        cls_nam = constitutive_model["parameterization class"]
        fobj, pathname, description = imp.find_module(py_mod, py_path)
        py_module = imp.load_module(py_mod, fobj, pathname, description)
        fobj.close()
        _parameterizer = getattr(py_module, cls_nam)
        del py_module
        return _parameterizer


def remove_index_file():
    """remove the index file"""
    try:
        os.remove(pc.PC_MTLS_FILE)
    except OSError:
        pass
    return

