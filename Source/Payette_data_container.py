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

import os
import sys
import re
import imp
import math
import numpy as np
import scipy.linalg
import time
from copy import deepcopy

import Source.Payette_utils as pu
import Source.Payette_tensor as pt
import Source.Payette_unit_manager as um


BOOL_MAP = {True: 1, False: 0}


class DataContainer:
    """
    CLASS NAME
       DataContainer

    PURPOSE
       data container class
    """

    def __init__(self,name):
        self.name = name.replace(" ","_")
        self.tensor_vars = ["Tensor","SymTensor","Vector","Matrix"]
        self.data_types = self.tensor_vars + ["Scalar","Boolean","Array",
                                              "Integer Array","List"]
        self._container_idx = 0
        self._container = {}
        self.plot_key_map = {}
        self.plot_key_list = []
        self.static_data_container = {}
        self.extra_vars_map = {}
        self.extra_vars_registered = False
        self.num_extra = 0
        pass

    def register(self, name, typ, init_val=None, plot_key=None,
                      dim=None, constant=False, plot_idx=None, xtra=None,
                      units=None):

        """
            register data to the data container

            INPUT:
                 *args:
              **kwargs: name
                        plot_key
                        init_val
                        type
                        plotable
        """

        if (name in self._container or
            name.upper() in self._container or
            name.lower() in self._container):
            pu.report_and_raise_error(
                "variable {0} already registered".format(name))

        if typ not in self.data_types:
            pu.report_and_raise_error(
                "unrecognized data type: {0}".format(typ))

        # For the time being, we don't care if the data is "None" or some
        # valid unit name. However, if it isn't one of those two, exit.
        if units is not None and not um.UnitManager.is_valid_units(units):
            pu.report_and_raise_error(
                 "Units '{0}' for ({1},{2}) is not valid.".
                                         format(units, plot_key, name))

        shape = None
        if typ in self.tensor_vars:
            if ( init_val is not None and
                 init_val != "Identity" and
                 not isinstance(init_val,(list,np.ndarray)) ):
                msg = ("{0} data {1} must be a list "
                       "or numpy.ndarray, got {2}".format(typ,name,init_val))
                pu.report_and_raise_error(msg)

            if typ == "SymTensor":
                if init_val is None:
                    init_val = np.zeros(pt.NSYM)
                elif init_val == "Identity":
                    init_val = self.I6
                elif len(init_val) != 6:
                    msg = "length of SymTensor data {0} != 6".format(name)
                    pu.report_and_raise_error(msg)

            elif typ == "Tensor":
                if init_val is None:
                    init_val = np.zeros(pt.NTENS)
                elif init_val == "Identity":
                    init_val = pt.I9
                elif len(init_val) != 9:
                    msg = "length of Tensor data {0} != 9".format(name)
                    pu.report_and_raise_error(msg)

            elif typ == "Vector":
                if init_val is None:
                    init_val = np.zeros(pt.NVEC)
                elif init_val == "Identity":
                    init_val = pt.I3
                elif len(init_val) != 3:
                    msg = "length of Vector data {0} != 3".format(name)
                    pu.report_and_raise_error(msg)

            elif typ == "Matrix":
                if init_val is None:
                    if dim is None:
                        msg="no dim specified for {0}".format(name)
                        pu.report_and_raise_error(msg)

                    else:
                        if not isinstance(dim,(int,tuple)):
                            msg="bad dim {0} for {1}".format(dim,name)
                            pu.report_and_raise_error(msg)
                        else:
                            init_val = np.zeros(dim)

                else:
                    if init_val == "Identity":
                        if not isinstance(dim,int):
                            msg="bad dim {0} for {1}".format(dim,name)
                            pu.report_and_raise_error(msg)

                        init_val = np.eye(dim)

            value = np.array(init_val)
            old_value = np.array(init_val)
            stashed_value = np.array(init_val)
            shape = value.shape

        elif typ == "List":

            if init_val is None: init_val = []

            if not isinstance(init_val,(list,tuple)):
                msg = "List data {0} must be a list".format(name)
                pu.report_and_raise_error(msg)

            value = [x for x in init_val]
            old_value = [x for x in init_val]
            stashed_value = [x for x in init_val]
            shape = len(value)

        elif typ == "Integer Array":

            if init_val is None: init_val = []

            if not isinstance(init_val,(np.ndarray,list,tuple)):
                msg = "Integer Array data {0} must not be a np.ndarray".format(name)
                pu.report_and_raise_error(msg)

            value = np.array([x for x in init_val],dtype=int)
            old_value = np.array([x for x in init_val],dtype=int)
            stashed_value = np.array([x for x in init_val],dtype=int)
            shape = value.shape

        elif typ == "Array":

            if init_val is None: init_val = []

            if not isinstance(init_val,(np.ndarray,list,tuple)):
                msg = "Array data {0} must be a np.ndarray".format(name)
                pu.report_and_raise_error(msg)

            value = np.array(init_val)
            old_value = np.array(init_val)
            stashed_value = np.array(init_val)
            shape = value.shape

        elif typ == "Scalar":

            if init_val is None: init_val = 0.

            if isinstance(init_val,(list,tuple,np.ndarray)):
                msg = "Scalar data {0} must ba scalar".format(name)
                pu.report_and_raise_error(msg)

            value = init_val
            old_value = init_val
            stashed_value = init_val
            shape = 0

        elif typ == "Boolean":

            if init_val is None: init_val = False

            if not isinstance(init_val,bool):
                msg = ("Boolean data {0} must be boolean, got {1}"
                       .format(name,init_val))
                pu.report_and_raise_error(msg)

            value = init_val
            old_value = init_val
            stashed_value = init_val

        plotable = plot_key is not None
        if not plotable:
            plot_name = None

        if plotable:
            if not isinstance(plot_key, str):
                msg = ("plot_key for {0} must be a string, got {1}"
                       .format(name, plot_key))
                pu.report_and_raise_error( msg)

            # format the plot key
            plot_key = plot_key.replace(" ","_").upper()

            # format the plot key
            plot_name = name

            if typ == "Vector":
                plot_key = ["{0}{1}".format(plot_key,i+1) for i in range(pt.NVEC)]

            elif typ == "SymTensor":
                plot_key = ["{0}{1}".format(plot_key,self.mapping(i))
                            for i in range(pt.NSYM)]

            elif typ == "Tensor":
                plot_key = ["{0}{1}".format(plot_key,self.mapping(i,sym=False))
                            for i in range(pt.NTENS)]

            # format the plot name
            tmp = " component "
            if typ == "Vector":
                plot_name = ["{0}{1}{2}".format(i+1,tmp,name)
                             for i in range(pt.NVEC)]

            elif typ == "SymTensor":
                plot_name = ["{0}{1}{2}".format(self.mapping(i),tmp,name)
                             for i in range(pt.NSYM)]

            elif typ == "Tensor":
                plot_name = ["{0}{1}{2}".format(self.mapping(i,sym=False),tmp,name)
                             for i in range(pt.NTENS)]

            if not isinstance(plot_key, list):
                nam = name if xtra is None else xtra
                self.plot_key_map[plot_key] = {"name": nam,
                                               "idx": plot_idx,
                                               "plot name": plot_name}
                self.plot_key_list.append(plot_key)
            else:
                for idx, key in enumerate(plot_key):
                    self.plot_key_map[key] = {"name": name,
                                              "idx": idx,
                                              "plot name": plot_name[idx]}
                    self.plot_key_list.append(key)

        # register the data
        self._container[name] = {"name": name,
                                     "plot key": plot_key,
                                     "plot name": plot_name,
                                     "idx": self._container_idx,
                                     "type": typ,
                                     "shape": shape,
                                     "value": value,
                                     "units": units,
                                     "previous value": old_value,
                                     "old value": old_value,
                                     "stashed value": old_value,
                                     "constant": constant,
                                     "plotable": plotable}
        self._container_idx += 1
        setattr(self,name.replace(" ","_").upper(),old_value)
        return


    def ensure_valid_units(self):
        """Returns nothing if all registered data have valid units.
        Fails otherwise."""
        for data, data_dict in self._container.iteritems():
            if not um.UnitManager.is_valid_units(data_dict['units']):
                pu.report_and_raise_error(
      "Registered data does not have valid units set:\n"+
      "\n".join(["({0}:{1})".format(x, y) for x, y in data_dict.iteritems()]))
        return


    def unregister(self, name):
        """ unregister data with the data container """
        try:
            del self._container[name]
        except KeyError:
            pu.log_warning(
                "attempting to unregister non-registered data {0}".format(name))

    def register_xtra(self, nxtra, names, keys, values):
        """ register extra data with the data container """

        if self.extra_vars_registered:
            pu.report_and_raise_error(
                "extra variables can only be registered once")

        self.extra_vars_registered = True
        self.num_extra = nxtra

        for i in range(nxtra):
            name = names[i]
            key = keys[i]
            value = values[i]
            self.register(name, "Scalar",
                               init_val=np.float64(value),
                               plot_key=key, xtra="__xtra__",
                               plot_idx=i)
            self.extra_vars_map[i] = name
            continue

        return

    def register_static(self, name, val):
        """Register unchanging data

        Parameters
        ----------
        name : str
        val :
        """

        if name in self.static_data_container:
            pu.report_and_raise_error(
                "static data {0} already registered".format(name))

        self.static_data_container[name] = val
        setattr(self,name.replace(" ","_").upper(),val)
        return

    def get_static(self,name):

        """ return static_data[name] """

        if not self.static_data_container.has_key(name):
            msg = """\
{0} not in {1}.static_data_container, registered data are:
{2}.""".format(name, self.name, ", ".join(self.static_data_container.keys()))
            pu.report_and_raise_error(msg)

        return self.static_data_container.get(name)

    def units(self, name):
        """ return the units of the data 'name' """
        idx = None
        if name in self.plot_key_map:
            plot_key = name
            name = self.plot_key_map[plot_key]["name"]
            idx = self.plot_key_map[plot_key]["idx"]

        return self._container.get(name)['units']

    def get(self, name, stash=False, cur=False,
                 previous=False, form="Array"):
        """ return simulation_data[name][valtyp] """

        idx = None
        if name in self.plot_key_map:
            plot_key = name
            name = self.plot_key_map[plot_key]["name"]
            idx = self.plot_key_map[plot_key]["idx"]

        if stash and cur:
            pu.report_and_raise_error("cannot get stash and cur simultaneously")

        if stash:
            valtyp = "stashed value"
        elif cur:
            valtyp = "value"
        elif previous:
            valtyp = "previous value"
        else:
            valtyp = "old value"

        # handle __xtra__
        if name == "__xtra__":
            retval = np.zeros(self.num_extra)
            for ixtra, nam in self.extra_vars_map.items():
                retval[ixtra] = self._container[nam][valtyp]
                continue

        else:
            data = self._container.get(name)
            if data is None:
                # data not a key in the container, but data could be a plot key
                msg = (
                    "{0} not in {1}.data_container. registered data are:\n{2}."
                    .format(name, self.name,
                            ", ".join(self._container.keys())))
                pu.report_and_raise_error(msg)

            typ = data["type"]

            if typ in self.tensor_vars:

                if form == "Array":
                    retval = np.array(data[valtyp])

                elif form == "MIG":
                    retval = pt.to_mig(data[valtyp])

                elif form == "Matrix":
                    if typ == "Vector":
                        pu.report_and_raise_error(
                            "cannont return vector matrix")

                    retval = pt.to_matrix(data[valtyp])

                else:
                    pu.report_and_raise_error(
                        "unrecognized form {0}".format(form))

            elif typ == "List":
                retval = [x for x in data[valtyp]]

            elif typ == "Integer Array":
                retval = np.array([x for x in data[valtyp]],dtype=int)

            elif typ == "Array":
                retval = np.array(data[valtyp])

            else:
                retval = data[valtyp]

        if idx is None:
            return retval

        else:
            return retval[idx]

    def restore(self, name, newval):
        self.store(name, newval)
        self.store(name, newval,old=True)
        self.store(name, newval,stash=True)
        return

    def store(self, name, newval, stash=False, old=False):

        """ store the simulation data """

        if old and stash:
            pu.report_and_raise_error("can only store old or stash not both")

        if stash:
            valtyp = "stashed value"
        elif old:
            valtyp = "old value"
        else:
            valtyp = "value"

        # handle __xtra__
        if name == "__xtra__":
            if len(newval) != self.num_extra:
                pu.report_and_raise_error("wrong size for extra variable array")

            for ixv,xv in enumerate(newval):
                name = self._xtra_name(ixv)
                self._container[name][valtyp] = xv
                continue
            return

        data = self._container.get(name)

        if data is None:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name, self.name, ", ".join(self._container.keys())))
            pu.report_and_raise_error(msg)

        typ = data["type"]

        if data["constant"]:
            return

        if typ in self.tensor_vars:

            if newval.shape == (3,3):
                if typ == "Tensor":
                    newval = pt.to_array(newval,symmetric=False)
                elif typ == "SymTensor":
                    newval = pt.to_array(newval)
                else:
                    pu.report_and_raise_error(
                        "vector cannot be converted from matrix")

            # check lengths
            if typ == "Vector" and len(newval) != 3:
                pu.report_and_raise_error(
                    "len Vector data {0} != 3".format(name))
            elif typ == "SymTensor" and len(newval) != 6:
                pu.report_and_raise_error(
                    "len SymTensor data {0} != 6".format(name))
            elif typ == "Tensor" and len(newval) != 9:
                pu.report_and_raise_error(
                    "len Tensor data {0} != 9".format(name))

            # store the newval
            newval = np.array(newval)

        elif typ == "List":
            newval = [x for x in newval]

        elif typ == "Integer Array":
            newval = np.array([x for x in newval],dtype=int)

        elif typ == "Array":
            newval = np.array(newval)

        else:
            # store the scalar variable
            newval = float(newval)

        # transfer oldvalue to previous and save new
        data["previous value"] = deepcopy(data["value"])
        data[valtyp] = newval
        return

    def stash(self, name, cur=False):

        """ stash "old value" in "stashed value" """

        # handle __xtra__
        if name == "__xtra__":
            for idx,name in self.extra_vars_map:
                value = self.get(name, cur=cur)
                # stash the value
                self.store(name, value,stash=True)
                continue
            return

        value = self.get(name, cur=cur)
        # stash the value
        self.store(name, value, stash=True)

        return

    def get_stash(self,name):
        return self.get(name, stash=True)

    def unstash(self, name):

        """ unstash "value" from "stashed value" """

        # handle __xtra__
        if name == "__xtra__":

            for idx, name in self.extra_vars_map.items():
                value = self.get_stash(name)
                self.store(name, value, old=True)
                continue
            return

        if name not in self._container:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name, self.name, ", ".join(self._container.keys())))
            pu.report_and_raise_error(msg)

        value = self.get_stash(name)
        self.store(name, value, old=True)

        return

    def advance_all(self):

        """ advance "value" to "old value" """

        for name in self._container:
            self.advance(name)
            continue
        return

    def advance(self, name, value=None):
        """ advance "value" to "old value" """

        if name == "__xtra__":
            if value is not None:
                if len(value) != self.num_extra:
                    pu.report_and_raise_error(
                        "len(value [{0:d}]) != num_extra [{1:d}]"
                        .format(len(value), self.num_extra))

                for idx, exval in enumerate(value):
                    name = self.extra_vars_map[idx]
                    self.store(name, exval, old=True)
                    continue

            else:
                for idx, name in self.extra_vars_map.items():
                    value = self.get(name, cur=True)
                    self.store(name, value, old=True)
                    continue

            setattr(self,name.replace(" ","_").upper(),value)
            return

        if name not in self._container:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self._container.keys())))
            pu.report_and_raise_error(msg)

        if value is None:
            value = self.get(name, cur=True)

        self.store(name, value)
        self.store(name, value, old=True)
        setattr(self, name.replace(" ","_").upper(),value)

        return

    def _xtra_name(self, idx):
        name = self.extra_vars_map.get(idx)
        if name is None:
            msg = "{0:d} not in {1}.extra_vars_map.".format(idx,self.name)
            pu.report_and_raise_error(msg)
        return name

    def _plot_key(self, name):
        data = self._container.get(name)
        if data is None:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name, self.name,", ".join(self._container.keys())))
            pu.report_and_raise_error(msg)
        return data["plot key"]

    def _plot_name(self, name, idx=None):

        if name in self.plot_key_map:
            plot_key = name
            plot_name = self.plot_key_map[plot_key]["plot name"]

        else:
            data = self._container.get(name)
            if data is None:
                msg = ("{0} not in plotable data. plotable data are:\n{2}."
                       .format(name, self.name,", ".join(self.plot_key_list)))
                pu.report_and_raise_error(msg)
            plot_name = data["plot name"]
            if idx is not None:
                plot_name = plot_name[idx]
        return plot_name

    def plot_keys(self):
        """ return a list of plot keys in the order registered """
        return self.plot_key_list

    def plotable(self,name):
        return self._container[name]["plotable"]

    def dump(self,name):
        """ return self._container[name] """

        if "__xtra__" in name:
            ex_vars = {}
            for idx,name in self.extra_vars_map.items():
                ex_vars[name] = self._container[name]
                continue
            return ex_vars

        data = self._container.get(name)
        if data is None:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self._container.keys())))
            pu.report_and_raise_error(msg)

        return data


    def data_container(self):
        return self._container

    def mapping(self, ii, sym = True):
        # return the 2D cartesian component of 1D array
        comp = None
        if sym:
            if ii == 0: comp = "11"
            elif ii == 1: comp = "22"
            elif ii == 2: comp = "33"
            elif ii == 3: comp = "12"
            elif ii == 4: comp = "23"
            elif ii == 5: comp = "13"
        else:
            if ii == 0: comp = "11"
            elif ii == 1: comp = "12"
            elif ii == 2: comp = "13"
            elif ii == 3: comp = "21"
            elif ii == 4: comp = "22"
            elif ii == 5: comp = "23"
            elif ii == 6: comp = "31"
            elif ii == 7: comp = "32"
            elif ii == 8: comp = "33"

        if comp:
            return comp
        else:
            pu.report_and_raise_error("bad mapping")
