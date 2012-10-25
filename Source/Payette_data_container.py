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

import Source.__runopts__ as ro
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
    _tensors = ["Tensor", "SymTensor", "Vector", "Matrix"]
    _dtypes = _tensors + ["Scalar", "Boolean", "Array", "Integer Array", "List"]

    def __init__(self,name):

        self.name = "_".join(name.split())
        self._setup = False

        #
        self.data_container = {}
        self.data_container_idx = 0
        self.plot_key_map = {}
        self._plot_keys = []

        self._xtra_registered = False
        self._xtra_start, self._xtra_end = None, None
        self._xtra_names, self._xtra_keys = [], []
        self._nxtra = 0

        # new
        self._container = {}
        self._plotable_data = {}
        self._ireg = 0
        self._end = 0
        self._data = []
        self._ival = []
        self._names = []

        pass

#     def register(self, name, typ, iv=None, plot_key=None,
#                       dim=None, constant=False, plot_idx=None, xtra=None,
#                       units=None):

#         """
#             register data to the data container

#             INPUT:
#                  *args:
#               **kwargs: name
#                         plot_key
#                         iv
#                         type
#                         plotable
#         """
#         if (name in self.data_container or
#             name.upper() in self.data_container or
#             name.lower() in self.data_container):
#             pu.report_and_raise_error(
#                 "variable {0} already registered".format(name))

#         if typ not in self._dtypes:
#             pu.report_and_raise_error(
#                 "unrecognized data type: {0}".format(typ))

#         # For the time being, we don't care if the data is "None" or some
#         # valid unit name. However, if it isn't one of those two, exit.
#         if units is not None and not um.UnitManager.is_valid_units(units):
#             pu.report_and_raise_error(
#                  "Units '{0}' for ({1},{2}) is not valid.".
#                                          format(units, plot_key, name))

#         shape = None
#         if typ in self._tensors:
#             if ( iv is not None and
#                  iv != "Identity" and
#                  not isinstance(iv,(list,np.ndarray)) ):
#                 msg = ("{0} data {1} must be a list "
#                        "or numpy.ndarray, got {2}".format(typ,name,iv))
#                 pu.report_and_raise_error(msg)

#             if typ == "SymTensor":
#                 if iv is None:
#                     iv = np.zeros(pt.NSYM)
#                 elif iv == "Identity":
#                     iv = pt.I6
#                 elif len(iv) != 6:
#                     msg = "length of SymTensor data {0} != 6".format(name)
#                     pu.report_and_raise_error(msg)

#             elif typ == "Tensor":
#                 if iv is None:
#                     iv = np.zeros(pt.NTENS)
#                 elif iv == "Identity":
#                     iv = pt.I9
#                 elif len(iv) != 9:
#                     msg = "length of Tensor data {0} != 9".format(name)
#                     pu.report_and_raise_error(msg)

#             elif typ == "Vector":
#                 if iv is None:
#                     iv = np.zeros(pt.NVEC)
#                 elif iv == "Identity":
#                     iv = pt.I3
#                 elif len(iv) != 3:
#                     msg = "length of Vector data {0} != 3".format(name)
#                     pu.report_and_raise_error(msg)

#             elif typ == "Matrix":
#                 if iv is None:
#                     if dim is None:
#                         msg="no dim specified for {0}".format(name)
#                         pu.report_and_raise_error(msg)

#                     else:
#                         if not isinstance(dim,(int,tuple)):
#                             msg="bad dim {0} for {1}".format(dim,name)
#                             pu.report_and_raise_error(msg)
#                         else:
#                             iv = np.zeros(dim)

#                 else:
#                     if iv == "Identity":
#                         if not isinstance(dim,int):
#                             msg="bad dim {0} for {1}".format(dim,name)
#                             pu.report_and_raise_error(msg)

#                         iv = np.eye(dim)

#             value = np.array(iv)
#             old_value = np.array(iv)
#             stashed_value = np.array(iv)
#             shape = value.shape

#         elif typ == "List":

#             if iv is None: iv = []

#             if not isinstance(iv,(list,tuple)):
#                 msg = "List data {0} must be a list".format(name)
#                 pu.report_and_raise_error(msg)

#             value = [x for x in iv]
#             old_value = [x for x in iv]
#             stashed_value = [x for x in iv]
#             shape = len(value)

#         elif typ == "Integer Array":

#             if iv is None: iv = []

#             if not isinstance(iv,(np.ndarray,list,tuple)):
#                 msg = "Integer Array data {0} must not be a np.ndarray".format(name)
#                 pu.report_and_raise_error(msg)

#             value = np.array([x for x in iv],dtype=int)
#             old_value = np.array([x for x in iv],dtype=int)
#             stashed_value = np.array([x for x in iv],dtype=int)
#             shape = value.shape

#         elif typ == "Array":

#             if iv is None: iv = []

#             if not isinstance(iv,(np.ndarray,list,tuple)):
#                 msg = "Array data {0} must be a np.ndarray".format(name)
#                 pu.report_and_raise_error(msg)

#             value = np.array(iv)
#             old_value = np.array(iv)
#             stashed_value = np.array(iv)
#             shape = value.shape

#         elif typ == "Scalar":

#             if iv is None: iv = 0.

#             if isinstance(iv,(list,tuple,np.ndarray)):
#                 msg = "Scalar data {0} must ba scalar".format(name)
#                 pu.report_and_raise_error(msg)

#             value = iv
#             old_value = iv
#             stashed_value = iv
#             shape = 0

#         elif typ == "Boolean":

#             if iv is None: iv = False

#             if not isinstance(iv,bool):
#                 msg = ("Boolean data {0} must be boolean, got {1}"
#                        .format(name,iv))
#                 pu.report_and_raise_error(msg)

#             value = iv
#             old_value = iv
#             stashed_value = iv

#         plotable = plot_key is not None
#         if not plotable:
#             plot_name = None

#         if plotable:
#             if not isinstance(plot_key, str):
#                 msg = ("plot_key for {0} must be a string, got {1}"
#                        .format(name, plot_key))
#                 pu.report_and_raise_error( msg)

#             # format the plot key
#             plot_key = plot_key.replace(" ","_").upper()

#             # format the plot key
#             plot_name = name

#             if typ == "Vector":
#                 plot_key = ["{0}{1}".format(plot_key,i+1) for i in range(pt.NVEC)]

#             elif typ == "SymTensor":
#                 plot_key = ["{0}{1}".format(plot_key,self.mapping(i))
#                             for i in range(pt.NSYM)]

#             elif typ == "Tensor":
#                 plot_key = ["{0}{1}".format(plot_key,self.mapping(i,sym=False))
#                             for i in range(pt.NTENS)]

#             # format the plot name
#             tmp = " component "
#             if typ == "Vector":
#                 plot_name = ["{0}{1}{2}".format(i+1,tmp,name)
#                              for i in range(pt.NVEC)]

#             elif typ == "SymTensor":
#                 plot_name = ["{0}{1}{2}".format(self.mapping(i),tmp,name)
#                              for i in range(pt.NSYM)]

#             elif typ == "Tensor":
#                 plot_name = ["{0}{1}{2}".format(self.mapping(i,sym=False),tmp,name)
#                              for i in range(pt.NTENS)]

#             if not isinstance(plot_key, list):
#                 nam = name if xtra is None else xtra
#                 self.plot_key_map[plot_key] = {"name": nam,
#                                                "idx": plot_idx,
#                                                "plot name": plot_name}
#                 self._plot_keys.append(plot_key)
#             else:
#                 for idx, key in enumerate(plot_key):
#                     self.plot_key_map[key] = {"name": name,
#                                               "idx": idx,
#                                               "plot name": plot_name[idx]}
#                     self._plot_keys.append(key)

#         # register the data
#         self.data_container[name] = {"name": name,
#                                      "plot key": plot_key,
#                                      "plot name": plot_name,
#                                      "idx": self.data_container_idx,
#                                      "type": typ,
#                                      "shape": shape,
#                                      "value": value,
#                                      "units": units,
#                                      "previous value": old_value,
#                                      "old value": old_value,
#                                      "stashed value": old_value,
#                                      "constant": constant,
#                                      "plotable": plotable}
#         self.data_container_idx += 1
#         setattr(self,name.replace(" ","_").upper(),old_value)
#         return

#     def ensure_valid_units(self):
#         """Returns nothing if all registered data have valid units.
#         Fails otherwise."""
#         for data, data_dict in self.data_container.iteritems():
#             if not um.UnitManager.is_valid_units(data_dict['units']):
#                 pu.report_and_raise_error(
#       "Registered data does not have valid units set:\n"+
#       "\n".join(["({0}:{1})".format(x, y) for x, y in data_dict.iteritems()]))
#         return


#     def unregister(self, name):
#         """ unregister data with the data container """
#         try:
#             del self.data_container[name]
#         except KeyError:
#             pu.log_warning(
#                 "attempting to unregister non-registered data {0}".format(name))

#     def register_xtra(self, nxtra, names, keys, values):
#         """ register extra data with the data container """

#         self.register_xtra_vars(nxtra, names, keys, values)

#         if self.extra_vars_registered:
#             pu.report_and_raise_error(
#                 "__xtra__ can only be registered once")

#         self.extra_vars_registered = True
#         self.num_extra = nxtra

#         for i in range(nxtra):
#             name = names[i]
#             key = keys[i]
#             value = values[i]
#             self.register(name, "Scalar",
#                                iv=np.float64(value),
#                                plot_key=key, xtra="__xtra__",
#                                plot_idx=i)
#             self.extra_vars_map[i] = name
#             continue

#         return

#     def register_static(self, name, val):
#         """Register unchanging data

#         Parameters
#         ----------
#         name : str
#         val :
#         """

#         if name in self.static_data_container:
#             pu.report_and_raise_error(
#                 "static data {0} already registered".format(name))

#         self.static_data_container[name] = val
#         setattr(self,name.replace(" ","_").upper(),val)
#         return

#     def get_static(self,name):

#         """ return static_data[name] """

#         if not self.static_data_container.has_key(name):
#             msg = """\
# {0} not in {1}.static_data_container, registered data are:
# {2}.""".format(name, self.name, ", ".join(self.static_data_container.keys()))
#             pu.report_and_raise_error(msg)

#         return self.static_data_container.get(name)

#     def units(self, name):
#         """ return the units of the data 'name' """
#         idx = None
#         if name in self.plot_key_map:
#             plot_key = name
#             name = self.plot_key_map[plot_key]["name"]
#             idx = self.plot_key_map[plot_key]["idx"]

#         return self.data_container.get(name)['units']

#     def get(self, name, stash=False, cur=False,
#                  previous=False, form="Array", **kwargs):
#         """ return simulation_data[name][valtyp] """

#         idx = None
#         if name in self.plot_key_map:
#             plot_key = name
#             name = self.plot_key_map[plot_key]["name"]
#             idx = self.plot_key_map[plot_key]["idx"]

#         if stash and cur:
#             pu.report_and_raise_error("cannot get stash and cur simultaneously")

#         if stash:
#             valtyp = "stashed value"
#         elif cur:
#             valtyp = "value"
#         elif previous:
#             valtyp = "previous value"
#         else:
#             valtyp = "old value"

#         # handle __xtra__
#         if name == "__xtra__":
#             retval = np.zeros(self.num_extra)
#             for ixtra, nam in self.extra_vars_map.items():
#                 retval[ixtra] = self.data_container[nam][valtyp]
#                 continue

#         else:
#             data = self.data_container.get(name)
#             if data is None:
#                 # data not a key in the container, but data could be a plot key
#                 msg = (
#                     "{0} not in {1}.data_container. registered data are:\n{2}."
#                     .format(name, self.name,
#                             ", ".join(self.data_container.keys())))
#                 pu.report_and_raise_error(msg)

#             typ = data["type"]

#             if typ in self._tensors:

#                 if form == "Array":
#                     retval = np.array(data[valtyp])

#                 elif form == "MIG":
#                     retval = pt.to_mig(data[valtyp])

#                 elif form == "Matrix":
#                     if typ == "Vector":
#                         pu.report_and_raise_error(
#                             "cannont return vector matrix")

#                     retval = pt.to_matrix(data[valtyp])

#                 else:
#                     pu.report_and_raise_error(
#                         "unrecognized form {0}".format(form))

#             elif typ == "List":
#                 retval = [x for x in data[valtyp]]

#             elif typ == "Integer Array":
#                 retval = np.array([x for x in data[valtyp]],dtype=int)

#             elif typ == "Array":
#                 retval = np.array(data[valtyp])

#             else:
#                 retval = data[valtyp]

#         if idx is not None:
#             retval = retval[idx]

#         return retval

#     def store(self, name, newval, stash=False, old=False, **kwargs):

#         """ store the simulation data """

#         args = ("-",) if old else ()

#         if old and stash:
#             pu.report_and_raise_error("can only store old or stash not both")

#         if stash:
#             valtyp = "stashed value"
#         elif old:
#             valtyp = "old value"
#         else:
#             valtyp = "value"

#         # handle __xtra__
#         if name == "__xtra__":
#             if len(newval) != self.num_extra:
#                 pu.report_and_raise_error("wrong size for extra variable array")

#             for ixv,xv in enumerate(newval):
#                 name = self._xtra_name(ixv)
#                 self.data_container[name][valtyp] = xv
#                 continue
#             return

#         data = self.data_container.get(name)

#         if data is None:
#             msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
#                    .format(name, self.name, ", ".join(self.data_container.keys())))
#             pu.report_and_raise_error(msg)

#         typ = data["type"]

#         if data["constant"]:
#             return

#         if typ in self._tensors:

#             if newval.shape == (3,3):
#                 if typ == "Tensor":
#                     newval = pt.to_array(newval,symmetric=False)
#                 elif typ == "SymTensor":
#                     newval = pt.to_array(newval)
#                 else:
#                     pu.report_and_raise_error(
#                         "vector cannot be converted from matrix")

#             # check lengths
#             if typ == "Vector" and len(newval) != 3:
#                 pu.report_and_raise_error(
#                     "len Vector data {0} != 3".format(name))
#             elif typ == "SymTensor" and len(newval) != 6:
#                 pu.report_and_raise_error(
#                     "len SymTensor data {0} != 6".format(name))
#             elif typ == "Tensor" and len(newval) != 9:
#                 pu.report_and_raise_error(
#                     "len Tensor data {0} != 9".format(name))

#             # store the newval
#             newval = np.array(newval)

#         elif typ == "List":
#             newval = [x for x in newval]

#         elif typ == "Integer Array":
#             newval = np.array([x for x in newval],dtype=int)

#         elif typ == "Array":
#             newval = np.array(newval)

#         else:
#             # store the scalar variable
#             newval = float(newval)

#         # transfer oldvalue to previous and save new
#         data["previous value"] = deepcopy(data["value"])
#         data[valtyp] = newval
#         return

#     def stash(self, name, cur=False):

#         """ stash "old value" in "stashed value" """

#         # handle __xtra__
#         if name == "__xtra__":
#             for idx,name in self.extra_vars_map:
#                 value = self.get(name, cur=cur)
#                 # stash the value
#                 self.store(name, value,stash=True)
#                 continue
#             return

#         value = self.get(name, cur=cur)
#         # stash the value
#         self.store(name, value, stash=True)

#         return

#     def get_stash(self,name):
#         return self.get(name, stash=True)

#     def unstash(self, name):

#         """ unstash "value" from "stashed value" """

#         # handle __xtra__
#         if name == "__xtra__":

#             for idx, name in self.extra_vars_map.items():
#                 value = self.get_stash(name)
#                 self.store(name, value, old=True)
#                 continue
#             return

#         if name not in self.data_container:
#             msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
#                    .format(name, self.name, ", ".join(self.data_container.keys())))
#             pu.report_and_raise_error(msg)

#         value = self.get_stash(name)
#         self.store(name, value, old=True)

#         return

#     def advance_all(self):

#         """ advance "value" to "old value" """

#         for name in self.data_container:
#             self.advance(name)
#             continue
#         return

#     def advance(self, name, value=None):
#         """ advance "value" to "old value" """

#         if name == "__xtra__":
#             if value is not None:
#                 if len(value) != self.num_extra:
#                     pu.report_and_raise_error(
#                         "len(value [{0:d}]) != num_extra [{1:d}]"
#                         .format(len(value), self.num_extra))

#                 for idx, exval in enumerate(value):
#                     name = self.extra_vars_map[idx]
#                     self.store(name, exval, old=True)
#                     continue

#             else:
#                 for idx, name in self.extra_vars_map.items():
#                     value = self.get(name, cur=True)
#                     self.store(name, value, old=True)
#                     continue

#             setattr(self,name.replace(" ","_").upper(),value)
#             return

#         if name not in self.data_container:
#             msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
#                    .format(name,self.name,", ".join(self.data_container.keys())))
#             pu.report_and_raise_error(msg)

#         if value is None:
#             value = self.get(name, cur=True)

#         self.store(name, value)
#         self.store(name, value, old=True)
#         setattr(self, name.replace(" ","_").upper(),value)

#         return

#     def _xtra_name(self, idx):
#         name = self.extra_vars_map.get(idx)
#         if name is None:
#             msg = "{0:d} not in {1}.extra_vars_map.".format(idx,self.name)
#             pu.report_and_raise_error(msg)
#         return name

#     def _plot_key(self, name):
#         data = self.data_container.get(name)
#         if data is None:
#             msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
#                    .format(name, self.name,", ".join(self.data_container.keys())))
#             pu.report_and_raise_error(msg)
#         return data["plot key"]

#     def _plot_name(self, name, idx=None):

#         if name in self.plot_key_map:
#             plot_key = name
#             plot_name = self.plot_key_map[plot_key]["plot name"]

#         else:
#             data = self.data_container.get(name)
#             if data is None:
#                 msg = ("{0} not in plotable data. plotable data are:\n{2}."
#                        .format(name, self.name,", ".join(self._plot_keys)))
#                 pu.report_and_raise_error(msg)
#             plot_name = data["plot name"]
#             if idx is not None:
#                 plot_name = plot_name[idx]
#         return plot_name

#     def plotable(self,name):
#         return self.data_container[name]["plotable"]

#     def dump(self,name):
#         """ return self.data_container[name] """

#         if "__xtra__" in name:
#             ex_vars = {}
#             for idx,name in self.extra_vars_map.items():
#                 ex_vars[name] = self.data_container[name]
#                 continue
#             return ex_vars

#         data = self.data_container.get(name)
#         if data is None:
#             msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
#                    .format(name,self.name,", ".join(self.data_container.keys())))
#             pu.report_and_raise_error(msg)

#         return data

#     def data_container(self):
#         return self.data_container

#     def mapping(self, ii, sym = True):
#         # return the 2D cartesian component of 1D array
#         comp = None
#         if sym:
#             if ii == 0: comp = "11"
#             elif ii == 1: comp = "22"
#             elif ii == 2: comp = "33"
#             elif ii == 3: comp = "12"
#             elif ii == 4: comp = "23"
#             elif ii == 5: comp = "13"
#         else:
#             if ii == 0: comp = "11"
#             elif ii == 1: comp = "12"
#             elif ii == 2: comp = "13"
#             elif ii == 3: comp = "21"
#             elif ii == 4: comp = "22"
#             elif ii == 5: comp = "23"
#             elif ii == 6: comp = "31"
#             elif ii == 7: comp = "32"
#             elif ii == 8: comp = "33"

#         if comp:
#             return comp
#         else:
#             pu.report_and_raise_error("bad mapping")

#     def register_xtra_vars(self, nxtra, names, keys, values):
#         """ register extra data with the data container """

#         if self.extra_vars_registered:
#             pu.report_and_raise_error(
#                 "extra variables can only be registered once")

#         self.data_container["__xtra__"] = {}
#         self.extra_vars_registered = True
#         self.num_extra = nxtra
#         self.xtra_start = len(self._ival)
#         self.xtra_end = self.xtra_start + nxtra

#         for i in range(nxtra):
#             self._xtra_names.append(names[i])
#             self._xtra_keys.append(keys[i])
#             self.register(
#                 names[i], "Scalar", iv=float(values[i]), plot_key=keys[i])
#             continue

#         return



    # ------------------------------------------------ N E W  M E T H O D S ---

    def register(self, name, dtype, iv=None, plot_key=None,
                 units=None, constant=False):
        """Register data with the data container

        Parameters
        ----------
        name : str
            data name
        dtype : str
            data type
        iv : optional,
            initial value
        plot_key : optional,
            plot key
        units : optional,
            unit system
        dim : str
            dimension
        constant : bool


        """
        if any(name.lower() == x.lower() for x in self._container):
            pu.report_and_raise_error("{0} already registered".format(name))

        if self._setup:
            pu.report_and_raise_error("Cannot register after setup")

        # For the time being, we don't care if the data is "None" or some
        # valid unit name. However, if it isn't one of those two, exit.
        if units is not None and not um.UnitManager.is_valid_units(units):
            pu.report_and_raise_error("Units '{0}' for ({1},{2}) is not valid."
                                      .format(units, plot_key, name))

        # remove extra spaces from name
        name = " ".join(name.split())

        # get dtype
        dtype = " ".join(dtype.lower().split())
        try:
            idx = [x.lower() for x in self._dtypes].index(dtype)
        except ValueError:
            pu.report_and_raise_error("{0} not recognized".format(dtype))
        dtype = self._dtypes[idx]

        # Check validity of data
        if dtype in self._tensors:
            if (iv is not None
                and iv != "Identity"
                and not isinstance(iv, (list, tuple, np.ndarray))):
                pu.report_and_raise_error(
                    "{0} must be of array-like type".format(name))

        # get initial value
        iv = self._initial_value(dtype, iv)

        # register the data up to this point
        start, end = self._end, self._end + len(iv)
        self._names.append(name)

        # plotable?
        plotable = plot_key is not None

        self._container[name] = {
            "name": name, "initial value": iv, "dtype": dtype,
            "plot key": [None] * len(iv),
            "registration order": self._ireg, "constant": constant,
            "start": start, "end": end, "units": units}

        # register attribute with NAME if constant
        if constant:
            setattr(self, "_".join(name.split()).upper(), iv[0])

        # hold initial value
        self._ival.extend(iv)

        # increment the number of registered variables
        self._ireg += 1
        self._end = len(self._ival)

        if not plotable:
            return

        # determine if plotable
        namea, keya = self._plot_info(name, plot_key, dtype, len(iv))
        self._container[name]["plot key"] = keya
        self._container[name]["name"] = namea

        # some safety checks
        if end - start != len(keya):
            pu.report_and_raise_error("end - start != len({0})".format(name))

        idx = self._end - len(keya)
        for i, key in enumerate(keya):
            self._plotable_data[key] = idx + i
            continue

        return

    def register_xtra(self, nxtra, names, keys, values):
        """ register extra data with the data container """

        if self._xtra_registered:
            pu.report_and_raise_error(
                "extra variables can only be registered once")

        self._container["__xtra__"] = {}
        self._xtra_registered = True
        self._nxtra = nxtra
        self._xtra_start = len(self._ival)
        self._xtra_end = self._xtra_start + nxtra

        for i in range(nxtra):
            self._xtra_names.append(names[i])
            self._xtra_keys.append(keys[i])
            self.register(
                names[i], "Scalar", iv=float(values[i]), plot_key=keys[i])
            continue

        return

    def get(self, name, *args, **kwargs):
        """Get the data from the data container

        The name of the variable may either be the name that was registered,
        or the plot key. The value returned will depend on which name is sent
        in. For instance if name == "stress", where "stress" was the name of
        the registered variable, the entire stress array will be returned, but
        if name == "STRESS11" only the 11 component will be returned.

        Parameters
        ----------
        name : str
           name of variable
        args : tuple
           The value of args depends on context

        Returns
        -------
        val : depends on context

        Notes
        -----
        if args[0] == "all"
            args[1] : int, optional {0}
               step number
        if args[0] == "plot key"
            args[1] : int, optional {ro.ISTEP} [int, "+", "-"]
               step number, "+"/"-" indicates the previous or next step,
                            respectively
        if args[0] == "name"
        if args[0] == "description"
            args[1] : int, optional {0}
                component number
        default :
            args[0] : int, optional {ro.ISTEP} [int, "+", "-"]
               step number, "+"/"-" indicates the previous or next step,
                            respectively
            kwargs["copy"] : bool
                return a copy instead of reference
            kwargs["form"] : str, optional {array} [array, matrix]
                return form

        """
        if not self._setup:
            self.setup_data_container()

        # determine what to get and get it
        if name == "all":
            N = ro.ISTEP if not len(args) else args[0]
            if not isinstance(N, int):
                N = {"+": ro.ISTEP+1, "-": ro.ISTEP-1}[N]
            return self._get_all(N)

        if "plot key" in args:
            N = 0 if len(args) == 1 else args[1]
            return self._get_plot_key(name, N)

        if "name" in args:
            return self._get_name(name)

        if "description" in args:
            N = 0 if len(args) <= 1 else args[1]
            return self._get_description(name, N)

        # get the value
        N = ro.ISTEP if not len(args) or ro.ISTEP == 0 else args[0]
        if not isinstance(N, int):
            N = {"+": ro.ISTEP+1, "-": ro.ISTEP-1}[N]
        copy = kwargs.get("copy", False)
        form = kwargs.get("form", "Array").lower()
        return self._get_value(name, N, copy, form)

        return

    def setup_data_container(self):
        """Create the data container for entire simulation.

        The data container is a MxN numpy ndarray that where M is the total
        number of simulatin steps and N is the number of registered variables.

        """
        self._setup = True
        ldata = len(self._ival)
        self._data = np.empty((ro.NSTEPS + 1, ldata), dtype=float)
        self._data[0, :] = self._ival

        # initialize constant data
        for key, val in self._container.items():
            if key == "__xtra__":
                continue
            if val["constant"]:
                start, end = val["start"], val["end"]
                self._data[:, start:end] = val["initial value"]
            continue
        return

    def restore(self, name=None):
        self.advance(("-", ), name=name)
        return

    def store(self, name, v, *args, **kwargs):
        """Store the data to the data container

        Parameters
        ----------
        name : str
           name of variable
        v : [float, array-like]
           the value

        """
        inc, N = False, ro.ISTEP
        if "+=" in args:
            inc = True
            args = tuple([x for x in args if x != "+="])

        N = ro.ISTEP if not len(args) else args[0]
        if not isinstance(N, int):
            N = {"+": ro.ISTEP+1, "-": ro.ISTEP-1}.get(N)
            if N is None:
                pu.report_and_raise_error("Bad arg '{0}'".format(N))

        if _is_xtra(name):
            start, end = self._xtra_start, self._xtra_end
        else:
            data = self._container.get(name)

            # validity checks
            if data is None:
                pu.report_and_raise_error("{0} not found".format(name))
            elif data["constant"]:
                pu.report_and_raise_error(
                    "Attempting to update '{0}' which is constant".format(name))

            start, end = data["start"], data["end"]

        if kwargs.get("stash"):
            self.stash(name)

        # items need to be flattened in to arrays if sent in as a matrix
        try:
            shape = v.shape
        except AttributeError:
            try:
                shape = (len(v), )
            except TypeError:
                shape = (1, )

        if len(shape) > 1:
            if data["dtype"] == "SymTensor":
                shape = (6, )
                v = np.array(
                    [v[0, 0], v[1, 1], v[2, 2], v[0, 1], v[1, 2], v[0, 2]])

        if shape:
            try:
                r, c = shape
            except ValueError:
                r, c = shape[0], 1
        else:
            r, c = 1, 1

        if inc:
            v = self._data[N, start:end] + v

        self._data[N, start:end] = np.reshape(v, r * c)
        return

    def advance(self, args=(), name=None):
        """

        """
        # get start and end columns
        if name is not None:
            if _is_xtra(name):
                start, end = self._xtra_start, self._xtra_end
            else:
                data = self._container.get(name)

                # validity checks
                if data is None:
                    pu.report_and_raise_error("{0} not found".format(name))
                elif data["constant"]:
                    pu.report_and_raise_error(
                        "Attempting to update '{0}' which is constant".format(name))

                start, end = data["start"], data["end"]
        else:
            start, end = 0, self._end

        N = ro.ISTEP if "-" not in args else ro.ISTEP - 1
        self._data[N + 1, start:end] = self._data[N, start:end]
        return

    def plot_keys(self):
        """ return a list of plot keys in the order registered """
        return sorted(self._plotable_data, key=lambda x: self._plotable_data[x])

    # ---------------------------------------- P R I V A T E  M E T H O D S ---

    def _initial_value(self, dtype, iv):
        """ Determine the data type and get initial value

        """
        if dtype == "SymTensor":
            if iv is None:
                iv = [0.] * pt.NSYM
            elif iv == "Identity":
                iv = pt.I6
            elif len(iv) != pt.NSYM:
                pu.report_and_raise_error("len({0}) != {1}"
                                          .format(name, pt.NSYM))

        elif dtype == "Tensor":
            if iv is None:
                iv = [0.] * pt.NTENS
            elif iv == "Identity":
                iv = pt.I9
            elif len(iv) != pt.NTENS:
                pu.report_and_raise_error("len({0}) != {1}"
                                          .format(name, pt.NTENS))

        elif dtype == "Vector":
            if iv is None:
                iv = [0.] * pt.NVEC
            elif iv == "Identity":
                iv = pt.I3
            elif len(iv) != pt.NVEC:
                pu.report_and_raise_error("len({0}) != {1}"
                                          .format(name, pt.NVEC))

        elif dtype == "List":
            if iv is None:
                iv = [0]
            if not isinstance(iv, (list, tuple)):
                pu.report_and_raise_error("{0} must be a list".format(name))

        elif dtype == "Integer Array":
            if iv is None:
                iv = [0]
            if not isinstance(iv, (np.ndarray, list, tuple)):
                pu.report_and_raise_error(
                    "{0} must be of array-like type".format(name))

        elif dtype == "Array":
            if iv is None:
                iv = []
            if not isinstance(iv, (np.ndarray, list, tuple)):
                pu.report_and_raise_error(
                    "{0} must be of array-like type".format(name))

        elif dtype == "Matrix":
            if iv is None:
                pu.report_and_raise_error(
                    "Matrix data must provide initial value")
            if not isinstance(iv, (np.ndarray, list, tuple)):
                pu.report_and_raise_error(
                    "{0} must be of array-like type".format(name))
            try:
                r, c = np.array(iv).shape
            except ValueError:
                pu.report_and_raise_error("{0} not a square matrix")
            iv = np.reshape(iv, r * c)

        elif dtype == "Scalar":
            if iv is None:
                iv = 0.
            if isinstance(iv, (list, tuple, np.ndarray)):
                pu.report_and_raise_error(
                    "{0} must be of scalar type".format(name))
            iv = [iv]

        elif dtype == "Boolean":
            iv = [BOOL_MAP[bool(iv)]]

        else:
            pu.report_and_raise_error(
                "{0} type not recognized".format(dtype))

        return iv

    def _plot_info(self, name, key, dtype, len_ival):
        """Format the plot keys

        Parameters
        ----------
        name : str
            name of the data
        key : str
            plot key for the data
        dtype : str
            data type
        len_ival : int
            length of data

        Returns
        -------
        namea : list
            descriptions
        keya : list
            plot keys

        """
        # format plot key and name
        if key is None:
            return [None] * len_ival, [None] * len_ival

        key = "_".join(key.split()).upper()
        keya, namea = [], []
        for i in range(len_ival):
            if dtype == "Vector":
                iname = "{0} component of {1}".format(pt.VEC_MAP[i], name)
                ikey = "{0}{1}".format(key, pt.VEC_MAP[i])

            elif dtype == "SymTensor":
                iname = "{0} component of {1}".format(pt.SYM_MAP[i], name)
                ikey = "{0}{1}".format(key, pt.SYM_MAP[i])

            elif dtype == "Tensor":
                iname = "{0} component of {1}".format(pt.TENS_MAP[i], name)
                ikey = "{0}{1}".format(key, pt.TENS_MAP[i])

            elif "Array" in dtype or "List" in dtype:
                iname = "{0} component of {1}".format(i + 1, name)
                ikey = "{0}{1}".format(key, i + 1)

            else:
                iname = name
                ikey = key
            namea.append(iname)
            keya.append(ikey)
            continue

        return namea, keya

    def _get_value(self, name, N, copy, form):
        """Get the value for name

        Parameters
        ----------
        name : str
        N : int
        copy : bool

        Returns
        -------
        val : float or array like

        """
        # check for where name is
        if _is_xtra(name):
            start, end = self._xtra_start, self._xtra_end
        elif name in self._plotable_data:
            start = self._plotable_data[name]
            end = start + 1
        else:
            data = self._container.get(name)
            if data is None:
                pu.report_and_raise_error("{0} not found".format(name))
            start, end = data["start"], data["end"]
            if data["dtype"] == "Matrix":
                form = "matrix"

        # check for scalar values
        if end - start == 1:
            return float(self._data[N, start])
        val = self._data[N, start:end]

        # reshape matrices
        if form == "matrix":
            if self._container[name]["dtype"] == "SymTensor":
                val = np.reshape([val[0], val[3], val[5],
                                  val[3], val[1], val[4],
                                  val[5], val[4], val[2]], (3, 3))
            elif data["dtype"] == "Matrix":
                i = np.sqrt(len(val))
                val = np.reshape(val, (i, i))
            else:
                val = np.reshape(val, (3, 3))
        elif form == "integer array":
            val = np.array(val, dtype=int)

        if copy:
            return np.array(val)
        return val

    def _get_name(self, key):
        """Get the name from the plot key

        Parameters
        ----------
        key : str, optional
           the plot key
        idx : int, plot index

        Returns
        -------
        name : str
           the name

        """
        # get name from plot key
        for name, data in self._container.items():
            try:
                return data["key"].index(key)
            except ValueError:
                continue
        pu.report_and_raise_error(
            "{0} not found in plot keys".format(key))
        return

    def _get_units(self, name):
        """ return the units of the data 'name' """
        if name in self._plotable_data:
            name = self._get_name_from_plot_key(name)
        data = self._container.get(name)
        if data is None:
            pu.report_and_raise_error("{0} not found".format(name))

        return data["units"]

    def _get_description(self, name, N):
        """Get the description for name

        Parameters
        ----------
        key : str, optional
           the plot key
        idx : int, plot index

        Returns
        -------
        name : str
           the name

        """
        if name in self._plotable_data:
            name = self._get_name_from_plot_key(name)
        data = self._container.get(name)
        if data is None:
            pu.report_and_raise_error("{0} not found".format(name))
        return data["name"][N]

    def _get_plot_key(self, name, N):
        if name in self._xtra_names:
            return self._xtra_keys[self._xtra_names.index(name)]
        data = self._container.get(name)
        if data is None:
            pu.report_and_raise_error("{0} not not found".format(name))
        return data["plot key"][N]

    def _get_all(self, N):
        return self._data[N, :].copy()

    def _get_name_from_plot_key(self, key):
        """get name from plot key"""
        if key in self._xtra_keys:
            return self._xtra_names[self._xtra_keys.index(key)]
        for name, data in self._container.items():
            if name == "__xtra__":
                continue
            if key in data["plot key"]:
                return name
        else:
            pu.report_and_raise_error(
                "{0} not found in plot keys".format(key))
        return


def _fix_name(name):
    return " ".join(name.split()).lower()


def _is_xtra(name):
    return _fix_name(name) == "__xtra__"
