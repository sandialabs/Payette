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
    _dtypes = _tensors + ["Scalar", "Boolean", "Array",
                          "Integer Array", "List"]

    def __init__(self, name):

        self.name = "_".join(name.split())
        self._setup = False

        #
        self._plot_keys = []
        self._xtra_registered = False
        self._xtra_start, self._xtra_end = None, None
        self._xtra_names, self._xtra_keys = [], []
        self._nxtra = 0

        self._container = {}
        self._plotable_data = {}
        self._ireg = 0
        self._end = 0
        self._data = []
        self._ival = []
        self._names = []

        pass

    # ------------------------------------------ P U B L I C  M E T H O D S ---

    def register(self, name, dtype, iv=None, plot_key=None,
                 units=None, constant=False, attr=False):
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
        if constant or attr:
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
            pu.report_and_raise_error("cannot get data until setup")

        # determine what to get and get it
        if name == "all":
            N = ro.ISTEP if not len(args) else args[0]
            if not isinstance(N, int):
                N = {"+": ro.ISTEP + 1, "-": ro.ISTEP - 1}[N]
            return self._get_all(N)

        if "plot key" in args:
            N = 0 if len(args) == 1 else args[1]
            return self._get_plot_key(name, N)

        if "name" in args:
            return self._get_name(name)

        if "description" in args:
            N = 0 if len(args) <= 1 else args[1]
            return self._get_description(name, N)

        if "units" in args:
            return self._get_units(name)

        # get the value
        N = ro.ISTEP if not len(args) or ro.ISTEP == 0 else args[0]
        if not isinstance(N, int):
            N = {"+": ro.ISTEP + 1, "-": ro.ISTEP - 1}[N]
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
        self._data[0, :] = [x for x in self._ival]

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
            args = tuple(x for x in args if x not in ("+=",))

        N = ro.ISTEP if not len(args) else args[0]
        if not isinstance(N, int):
            N = {"+": ro.ISTEP + 1, "-": ro.ISTEP - 1}.get(N)
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
                v = np.array([v[pt.SYMTENS_MAP[0]], v[pt.SYMTENS_MAP[1]],
                              v[pt.SYMTENS_MAP[2]], v[pt.SYMTENS_MAP[3]],
                              v[pt.SYMTENS_MAP[4]], v[pt.SYMTENS_MAP[5]]])

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

        if name.lower() == "istep":
            setattr(self, "ISTEP", int(v))

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
                        "Attempting to update '{0}' which is constant"
                        .format(name))

                start, end = data["start"], data["end"]
        else:
            start, end = 0, self._end

        N = ro.ISTEP if "-" not in args else ro.ISTEP - 1
        if N == -1:
            # restoring on first step, restore from self._ival
            self._data[0, start:end] = np.array(self._ival)[start:end]
        else:
            self._data[N + 1, start:end] = self._data[N, start:end]
        return

    def plot_keys(self):
        """ return a list of plot keys in the order registered """
        return sorted(self._plotable_data, key=lambda x: self._plotable_data[x])

    def dump(self, N=None):
        if N is None:
            N = ro.NSTEPS
        return np.array(self._data[:N, :])

    def ensure_valid_units(self):
        """Returns nothing if all registered data have valid units.
        Fails otherwise."""
        for name, data in self._container.items():
            units = data["units"]
            if not um.UnitManager.is_valid_units(units):
                pu.report_and_raise_error(
                    "Units '{0}' for '{1}' not valid".format(units, name))
        return

    def xtra_keys(self):
        return [x for x in self._xtra_keys]

    def nxtra(self):
        return self._nxtra

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
        key = None
        if name in self._plotable_data:
            key, name = name, self._get_name_from_plot_key(name)
        data = self._container.get(name)
        if data is None:
            pu.report_and_raise_error("{0} not found".format(name))
        if key is not None:
            N = data["plot key"].index(key)
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


def _is_xtra(name):
    return name.strip().lower() == "__xtra__"
