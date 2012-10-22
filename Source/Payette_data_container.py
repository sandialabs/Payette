#!/usr/bin/env python

import numpy as np
from copy import deepcopy

import Source.__runopts__ as ro
import Source.Payette_utils as pu
import Source.Payette_unit_manager as um

NVEC, NSYM, NTENS = 3, 6, 9
VEC_MAP = {0: "1", 1: "2", 2: "3"}
I6 = [1, 1, 1, 0, 0, 0]
SYM_MAP = {0: "11", 3: "12", 5: "13",
                    1: "22", 4: "23",
                             2: "33"}
I9 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
TENS_MAP = {0: "11", 1: "12", 2: "13",
            3: "21", 4: "22", 5: "23",
            6: "31", 7: "32", 8: "33"}
BOOL_MAP = {True: 1, False: 0}


class DataContainer(object):
    _tensors = ["Tensor", "SymTensor", "Vector", "Matrix",]
    _dtypes = _tensors + ["Array", "List", "Scalar", "Boolean"]

    def __init__(self, name):
        self.name = "_".join(name.split())
        self._data_container = {}
        self._ireg = 0
        self._end = 0
        self._setup = False
        self._data = []
        self._ival = []
        self.extra_vars_registered = False
        self._plotable_data = {}
        self._names = []
        self._xtra_names, self._xtra_keys = [], []
        pass

    def register(self, name, dtype, ival=None, plot_key=None,
                 units=None, description=None, constant=False):
        """Register data with the data container

        Parameters
        ----------
        name : str
            data name
        dtype : str
            data type
        ival : optional,
            initial value
        plot_key : optional,
            plot key
        units : optional,
            unit system
        dim : str
            dimension
        constant : bool


        """
        if any(name.lower() == x.lower() for x in self._data_container):
            pu.report_and_raise_error("{0} already registered".format(name))

        if self._setup:
            pu.report_and_raise_error("Cannot register after setup")

        # For the time being, we don't care if the data is "None" or some
        # valid unit name. However, if it isn't one of those two, exit.
        if units is not None and not um.UnitManager.is_valid_units(units):
            pu.report_and_raise_error(
                "Units '{0}' for ({1},{2}) is not valid."
                .format(units, plot_key, name))

        # remove extra spaces from name
        name = " ".join(name.split())

        # get dtype
        dtype = " ".join(dtype.lower().split())
        try:
            idx = [x.lower() for x in self._dtypes].index(dtype)
        except ValueError:
            pu.report_and_raise_error(
                "{0} not a recognized dtype".format(dtype))
        dtype = self._dtypes[idx]

        # Check validity of data
        if dtype in self._tensors:
            if (ival is not None
                and ival != "Identity"
                and not isinstance(ival, (list, tuple, np.ndarray))):
                pu.report_and_raise_error(
                    "{0} must be of array-like type".format(name))

        ival = self._parse_initial_value(dtype, ival)

        # register the data up to this point
        start, end = self._end, self._end + len(ival)
        self._names.append(name)

        # default description
        if description is None:
            description = [name] * len(ival)

        # plotable?
        plotable = plot_key is not None

        self._data_container[name] = {
            "name": name, "initial value": ival, "dtype": dtype,
            "description": description, "plot key": [None] * len(ival),
            "registration order": self._ireg, "constant": constant,
            "start": start, "end": end, "units": units}

        # register attribute with NAME if constant
        if constant:
            setattr(self, "_".join(name.split()).upper(), ival[0])

        # hold initial value
        self._ival.extend(ival)

        # increment the number of registered variables
        self._ireg += 1
        self._end = len(self._ival)

        if not plotable:
            return

        # determine if plotable
        namea, keya = self._parse_plot_info(name, plot_key, dtype, len(ival))
        self._data_container[name]["plot key"] = keya
        self._data_container[name]["description"] = namea

        # some safety checks
        if end - start != len(keya):
            pu.report_and_raise_error("end - start != len({0})".format(name))

        idx = self._end - len(keya)
        for i, key in enumerate(keya):
            self._plotable_data[key] = idx + i
            continue

        return

    def register_xtra_vars(self, nxtra, names, keys, values):
        """ register extra data with the data container """

        if self.extra_vars_registered:
            pu.report_and_raise_error(
                "extra variables can only be registered once")

        self._data_container["__xtra__"] = {}
        self.extra_vars_registered = True
        self.num_extra = nxtra
        self.xtra_start = len(self._ival)
        self.xtra_end = self.xtra_start + nxtra

        for i in range(nxtra):
            self._xtra_names.append(names[i])
            self._xtra_keys.append(keys[i])
            self.register(
                names[i], "Scalar", ival=float(values[i]), plot_key=keys[i])
            continue

        return

    def setup_data_container(self):
        """Create the data container for entire simulation.

        The data container is a MxN numpy ndarray that where M is the total
        number of simulatin steps and N is the number of registered variables.

        """
        self._setup = True
        ldata = len(self._ival)
        self._data = np.empty((ro.NSTEPS + 2, ldata), dtype=float)
        self._data[0, :] = self._ival

        # initialize constant data
        for key, val in self._data_container.items():
            if key == "__xtra__":
                continue
            if val["constant"]:
                start, end = val["start"], val["end"]
                self._data[:, start:end] = val["initial value"]
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

        elif "plot key" in args:
            N = 0 if len(args) == 1 else args[1]
            return self._get_plot_key(name, N)

        elif "name" in args:
            return self._get_name(name)

        elif "description" in args:
            N = 0 if len(args) <= 1 else args[1]
            return self._get_description(name, N)

        else:
            # get the value
            N = ro.ISTEP if not len(args) else args[0]
            if not isinstance(N, int):
                N = {"+": ro.ISTEP+1, "-": ro.ISTEP-1}[N]
            copy = kwargs.get("copy", False)
            form = kwargs.get("form", "Array").lower()
            return self._get_value(name, N, copy, form)

        return

    def advance(self, args=(), name=None):
        """

        """
        # get start and end columns
        if name is not None:
            if _is_xtra(name):
                start, end = self.xtra_start, self.xtra_end
            else:
                data = self._data_container.get(name)

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

    def restore(self, name=None):
        self.advance(("-",), name=name)

    def save(self, name, v, *args, **kwargs):
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
            start, end = self.xtra_start, self.xtra_end
        else:
            data = self._data_container.get(name)

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

    def stash(self, name):
        """Stash the data

        Parameters
        ----------
        name : str
           name of variable

        """
        if _is_xtra(name):
            self._data_container["__xtra__"]["stash"] = self.get(name, copy=True)
        else:
            self._data_container[name]["stash"] = self.get(name, copy=True)
        return

    def unstash(self, name):
        """Unstash the data

        Parameters
        ----------
        name : str
           name of variable

        """
        if _is_xtra(name):
            self.save(name, self._data_container["__xtra__"]["stash"])
            del self._data_container["__xtra__"]["stash"]
        else:
            self.save(name, self._data_container[name]["stash"])
            del self._data_container[name]["stash"]
        return

    def remove(self, name):
        """ unregister data with the data container """
        try:
            del self._data_container[name]
        except KeyError:
            pu.log_warning(
                "attempting to unregister non-registered data {0}".format(name))

    def plot_keys(self):
        """ return a list of plot keys in the order registered """
        return sorted(self._plotable_data, key=lambda x: self._plotable_data[x])

    def plotable_data(self, N=None):
        cols = sorted(self._plotable_data.values())
        rows = {"all": [[x] for x in range(ro.NSTEPS+1)],
                None: ro.ISTEP, isinstance(N, int): N}[N]
        return self._data[rows, cols]

    def dump(self,name):
        """ return self.data_container[name] """
        if _is_xtra(name):
            xtra = {}
            for name in self._xtra_names:
                xtra[name] = self._data_container[name]
                continue
            return xtra
        data = self._data_container.get(name)
        if data is None:
            pu.report_and_raise_error(
                "{0} not in {1}.data_container. registered data are:\n{2}."
                .format(name,self.name,", ".join(self.data_container.keys())))
        return data

    def ensure_all_registered_data_have_valid_units(self):
        """Returns nothing if all registered data have valid units.
        Fails otherwise."""
        for name, data in self._data_container.items():
            if not um.UnitManager.is_valid_units(data["units"]):
                pu.report_and_raise_error(
                    "Registered data does not have valid units "
                    "set:\n\n".join(
                        ["({0}:{1})".format(x, y) for x, y in data.items()]))
            continue
        return

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
            start, end = self.xtra_start, self.xtra_end
        elif name in self._plotable_data:
            start = self._plotable_data[name]
            end = start + 1
        else:
            data = self._data_container.get(name)
            if data is None:
                pu.report_and_raise_error("{0} not found".format(name))
            start, end = data["start"], data["end"]
            if data["dtype"] == "Matrix":
                form = "matrix"

        # check for scalar values
        if end - start == 1:
            return self._data[N, start]

        val = self._data[N, start:end]

        # reshape matrices
        if form == "matrix":
            if self._data_container[name]["dtype"] == "SymTensor":
                return np.reshape([val[0], val[3], val[5],
                                   val[3], val[1], val[4],
                                   val[5], val[4], val[2]], (3, 3))

            if data["dtype"] == "Matrix":
                i = np.sqrt(len(val))
                return np.reshape(val, (i, i))

            return np.reshape(val, (3, 3))

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
        for name, data in self._data_container.items():
            try:
                return name, data["plot key"].index(key)
            except ValueError:
                continue
        pu.report_and_raise_error(
            "{0} not found in plot keys".format(key))
        return

    def _get_units(self, name):
        """ return the units of the data 'name' """
        if name in self._plotable_data:
            name = self._get_name_from_plot_key(name)
        data = self._data_container.get(name)
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
        data = self._data_container.get(name)
        if data is None:
            pu.report_and_raise_error("{0} not found".format(name))
        return data["description"][N]

    def _get_plot_key(self, name, N):
        if name in self._xtra_names:
            return self._xtra_keys[self._xtra_names.index(name)]
        data = self._data_container.get(name)
        if data is None:
            pu.report_and_raise_error("{0} not not found".format(name))
        return data["plot key"][N]

    def _get_all(self, N):
        return deepcopy(self._data[N, :])

    def _get_name_from_plot_key(self, key):
        """get name from plot key"""
        for name, data in self._data_container.items():
            if key in data["plot key"]:
                return name
        else:
            pu.report_and_raise_error(
                "{0} not found in plot keys".format(key))
        return

    def _parse_initial_value(self, dtype, ival):
        """ Determine the data type and get initial value

        """
        if dtype == "SymTensor":
            if ival is None:
                ival = [0.] * NSYM
            elif ival == "Identity":
                ival = I6
            elif len(ival) != NSYM:
                pu.report_and_raise_error("len({0}) != {1}".format(name, NSYM))

        elif dtype == "Tensor":
            if ival is None:
                ival = [0.] * NTENS
            elif ival == "Identity":
                ival = I9
            elif len(ival) != NTENS:
                pu.report_and_raise_error("len({0}) != {1}".format(name, NTENS))

        elif dtype == "Vector":
            if ival is None:
                ival = [0.] * NVEC
            elif ival == "Identity":
                ival = I3
            elif len(ival) != NVEC:
                pu.report_and_raise_error("len({0}) != {1}".format(name, NVEC))

        elif dtype == "List":
            if ival is None:
                ival = [0]
            if not isinstance(ival, (list, tuple)):
                pu.report_and_raise_error("{0} must be a list".format(name))

        elif dtype == "Array":
            if ival is None:
                ival = []
            if not isinstance(ival, (np.ndarray, list, tuple)):
                pu.report_and_raise_error(
                    "{0} must be of array-like type".format(name))

        elif dtype == "Matrix":
            if ival is None:
                pu.report_and_raise_error(
                    "Matrix data must provide initial value")
            if not isinstance(ival, (np.ndarray, list, tuple)):
                pu.report_and_raise_error(
                    "{0} must be of array-like type".format(name))
            try:
                r, c = np.array(ival).shape
            except ValueError:
                pu.report_and_raise_error("{0} not a square matrix")
            ival = np.reshape(ival, r * c)

        elif dtype == "Scalar":
            if ival is None:
                ival = 0.
            if isinstance(ival, (list, tuple, np.ndarray)):
                pu.report_and_raise_error(
                    "{0} must be of scalar type".format(name))
            ival = [ival]

        elif dtype == "Boolean":
            ival = [BOOL_MAP[bool(ival)]]

        else:
            pu.report_and_raise_error(
                "{0} type not recognized".format(dtype))

        return ival

    def _parse_plot_info(self, name, key, dtype, len_ival):
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
                iname = "{0} component of {1}".format(VEC_MAP[i], name)
                ikey = "{0}{1}".format(key, VEC_MAP[i])

            elif dtype == "SymTensor":
                iname = "{0} component of {1}".format(SYM_MAP[i], name)
                ikey = "{0}{1}".format(key, SYM_MAP[i])

            elif dtype == "Tensor":
                iname = "{0} component of {1}".format(TENS_MAP[i], name)
                ikey = "{0}{1}".format(key, TENS_MAP[i])

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


def _fix_name(name):
    return " ".join(name.split()).lower()


def _is_xtra(name):
    return _fix_name(name) == "extra variables"


if __name__ == "__main__":
    import sys
    ro.NSTEPS = 7
    ro.ISTEP = 0
    dc = DataContainer("a name")
    dc.register("time", "Scalar", plot_key="time")
    dc.register("nprints", "Scalar", ival=2, constant=True)
    dc.register("stress", "SymTensor", plot_key="sig")
    dc.register("strain", "SymTensor", ival=[1,1,1,1,1,1], plot_key="EPS")
    dc.register("boolean", "Boolean", ival=True, constant=True)
    dc.register("false boolean", "Boolean", ival=False, constant=True)
    dc.register("deformation gradient", "Tensor", ival="Identity", plot_key="F")
    dc.register("matrix", "Matrix", ival=[[1,2],[3,4]], constant=True)
    names = ["xtra var {0}".format(i) for i in range(10)]
    keys = ["xtra_{0}".format(i) for i in range(10)]
    xtra = np.array([20,21,22,23,24,25,26,27,28,29])
    dc.register_xtra_vars(10, names, keys, xtra)
    dc.setup_data_container()

    DT = 1.
    dc.advance()

    for I in range(0, ro.NSTEPS):

        ro.ISTEP += 1

        dc.save("time", DT, "+=")
        a = dc.get("stress")
        dc.save("stress", np.array([4,4,4,4,4,4], dtype=float))
        a = a + np.ones(len(a))
        dc.save("stress", a)
        dc.save("extra variables", dc.get("extra variables"))
        dc.save("deformation gradient", dc.get("deformation gradient"))
        dc.advance()

    print
    sys.stdout.write(" ".join("{0:12s}".format(x) for x in dc.plot_keys()) + "\n")
    for row in dc.plotable_data("all"):
        sys.stdout.write(" ".join("{0:12.6E}".format(x) for x in row) + "\n")
        continue
    print
    print dc.get("deformation gradient")
    print dc.get("deformation gradient", form="matrix")
    print dc.get("F11", "description")
    print dc.get("stress")

    print
    print dc.get("all").tolist()
