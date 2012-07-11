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
        self.data_container_idx = 0
        self.data_container = {}
        self.plot_key_map = {}
        self.plot_key_list = []
        self.static_data_container = {}
        self.extra_vars_map = {}
        self.extra_vars_registered = False
        self.num_extra = 0
        self.I3 = np.array([1.,1.,1.])
        self.I6 = np.array([1.,1.,1.,0.,0.,0.])
        self.I9 = np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.])
        self.nvec = 3
        self.nsym = 6
        self.ntens = 9
        pass

    def register_data(self, name, typ, init_val=None, plot_key=None,
                      dim=None, constant=False, plot_idx=None, xtra=None):

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

        if (name in self.data_container or
            name.upper() in self.data_container or
            name.lower() in self.data_container):
            pu.report_and_raise_error(
                "variable {0} already registered".format(name))

        if typ not in self.data_types:
            pu.report_and_raise_error(
                "unrecognized data type: {0}".format(typ))

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
                    init_val = np.zeros(self.nsym)
                elif init_val == "Identity":
                    init_val = self.I6
                elif len(init_val) != 6:
                    msg = "length of SymTensor data {0} != 6".format(name)
                    pu.report_and_raise_error(msg)

            elif typ == "Tensor":
                if init_val is None:
                    init_val = np.zeros(self.ntens)
                elif init_val == "Identity":
                    init_val = self.I9
                elif len(init_val) != 9:
                    msg = "length of Tensor data {0} != 9".format(name)
                    pu.report_and_raise_error(msg)

            elif typ == "Vector":
                if init_val is None:
                    init_val = np.zeros(self.nvec)
                elif init_val == "Identity":
                    init_val = self.I3
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
                plot_key = ["{0}{1}".format(plot_key,i+1) for i in range(self.nvec)]

            elif typ == "SymTensor":
                plot_key = ["{0}{1}".format(plot_key,self.mapping(i))
                            for i in range(self.nsym)]

            elif typ == "Tensor":
                plot_key = ["{0}{1}".format(plot_key,self.mapping(i,sym=False))
                            for i in range(self.ntens)]

            # format the plot name
            tmp = " component "
            if typ == "Vector":
                plot_name = ["{0}{1}{2}".format(i+1,tmp,name)
                             for i in range(self.nvec)]

            elif typ == "SymTensor":
                plot_name = ["{0}{1}{2}".format(self.mapping(i),tmp,name)
                             for i in range(self.nsym)]

            elif typ == "Tensor":
                plot_name = ["{0}{1}{2}".format(self.mapping(i,sym=False),tmp,name)
                             for i in range(self.ntens)]

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
        self.data_container[name] = {"name": name,
                                     "plot key": plot_key,
                                     "plot name": plot_name,
                                     "idx": self.data_container_idx,
                                     "type": typ,
                                     "shape": shape,
                                     "value": value,
                                     "previous value": old_value,
                                     "old value": old_value,
                                     "stashed value": old_value,
                                     "constant": constant,
                                     "plotable": plotable}
        self.data_container_idx += 1
        setattr(self,name.replace(" ","_").upper(),old_value)
        return

    def unregister_data(self, name):
        """ unregister data with the data container """
        try:
            del self.data_container[name]
        except KeyError:
            pu.log_warning(
                "attempting to unregister non-registered data {0}".format(name))

    def register_xtra_vars(self, nxtra, names, keys, values):
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
            self.register_data(name, "Scalar",
                               init_val=np.float64(value),
                               plot_key=key, xtra="extra variables",
                               plot_idx=i)
            self.extra_vars_map[i] = name
            continue

        return

    def register_static_data(self, name, val):
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

    def get_static_data(self,name):

        """ return static_data[name] """

        if not self.static_data_container.has_key(name):
            msg = """\
{0} not in {1}.static_data_container, registered data are:
{2}.""".format(name, self.name, ", ".join(self.static_data_container.keys()))
            pu.report_and_raise_error(msg)

        return self.static_data_container.get(name)

    def get_data(self, name, stash=False, cur=False,
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

        # handle extra variables
        if name == "extra variables":
            retval = np.zeros(self.num_extra)
            for ixtra, nam in self.extra_vars_map.items():
                retval[ixtra] = self.data_container[nam][valtyp]
                continue

        else:
            data = self.data_container.get(name)
            if data is None:
                # data not a key in the container, but data could be a plot key
                msg = (
                    "{0} not in {1}.data_container. registered data are:\n{2}."
                    .format(name, self.name,
                            ", ".join(self.data_container.keys())))
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

    def restore_data(self, name, newval):
        self.store_data(name, newval)
        self.store_data(name, newval,old=True)
        self.store_data(name, newval,stash=True)
        return

    def store_data(self, name, newval, stash=False, old=False):

        """ store the simulation data """

        if old and stash:
            pu.report_and_raise_error("can only store old or stash not both")

        if stash:
            valtyp = "stashed value"
        elif old:
            valtyp = "old value"
        else:
            valtyp = "value"

        # handle extra variables
        if name == "extra variables":
            if len(newval) != self.num_extra:
                pu.report_and_raise_error("wrong size for extra variable array")

            for ixv,xv in enumerate(newval):
                name = self.getExName(ixv)
                self.data_container[name][valtyp] = xv
                continue
            return

        data = self.data_container.get(name)

        if data is None:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name, self.name, ", ".join(self.data_container.keys())))
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

    def stash_data(self, name, cur=False):

        """ stash "old value" in "stashed value" """

        # handle extra variables
        if name == "extra variables":
            for idx,name in self.extra_vars_map:
                value = self.get_data(name, cur=cur)
                # stash the value
                self.store_data(name, value,stash=True)
                continue
            return

        value = self.get_data(name, cur=cur)
        # stash the value
        self.store_data(name, value, stash=True)

        return

    def get_stashed_data(self,name):
        return self.get_data(name, stash=True)

    def unstash_data(self, name):

        """ unstash "value" from "stashed value" """

        # handle extra variables
        if name == "extra variables":

            for idx, name in self.extra_vars_map.items():
                value = self.get_stashed_data(name)
                self.store_data(name, value, old=True)
                continue
            return

        if name not in self.data_container:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name, self.name, ", ".join(self.data_container.keys())))
            pu.report_and_raise_error(msg)

        value = self.get_stashed_data(name)
        self.store_data(name, value, old=True)

        return

    def advance_all_data(self):

        """ advance "value" to "old value" """

        for name in self.data_container:
            self.advance_data(name)
            continue
        return

    def advance_data(self, name, value=None):
        """ advance "value" to "old value" """

        if name == "extra variables":
            if value is not None:
                if len(value) != self.num_extra:
                    pu.report_and_raise_error(
                        "len(value [{0:d}]) != num_extra [{1:d}]"
                        .format(len(value), self.num_extra))

                for idx, exval in enumerate(value):
                    name = self.extra_vars_map[idx]
                    self.store_data(name, exval, old=True)
                    continue

            else:
                for idx, name in self.extra_vars_map.items():
                    value = self.get_data(name, cur=True)
                    self.store_data(name, value, old=True)
                    continue

            setattr(self,name.replace(" ","_").upper(),value)
            return

        if name not in self.data_container:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            pu.report_and_raise_error(msg)

        if value is None:
            value = self.get_data(name, cur=True)

        self.store_data(name, value)
        self.store_data(name, value, old=True)
        setattr(self, name.replace(" ","_").upper(),value)

        return

    def getExName(self, idx):
        name = self.extra_vars_map.get(idx)
        if name is None:
            msg = "{0:d} not in {1}.extra_vars_map.".format(idx,self.name)
            pu.report_and_raise_error(msg)
        return name

    def get_plot_key(self, name):
        data = self.data_container.get(name)
        if data is None:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name, self.name,", ".join(self.data_container.keys())))
            pu.report_and_raise_error(msg)
        return data["plot key"]

    def get_plot_name(self, name, idx=None):

        if name in self.plot_key_map:
            plot_key = name
            plot_name = self.plot_key_map[plot_key]["plot name"]

        else:
            data = self.data_container.get(name)
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
        return self.data_container[name]["plotable"]

    def dump_data(self,name):
        """ return self.data_container[name] """

        if "extra variables" in name:
            ex_vars = {}
            for idx,name in self.extra_vars_map.items():
                ex_vars[name] = self.data_container[name]
                continue
            return ex_vars

        data = self.data_container.get(name)
        if data is None:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            pu.report_and_raise_error(msg)

        return data


    def dataContainer(self):
        return self.data_container

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

