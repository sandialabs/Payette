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


from __future__ import print_function
import os
import sys
import re
import imp
import math
import numpy as np
import scipy.linalg
import time

from Source.Payette_utils import *
from Source.Payette_tensor import *

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
        self.option_container = {}
        self.extra_vars_map = {}
        self.extra_vars_registered = False
        self.I3 = np.array([1.,1.,1.])
        self.I6 = np.array([1.,1.,1.,0.,0.,0.])
        self.I9 = np.array([1.,0.,0.,0.,1.,0.,0.,0.,1.])
        self.nvec = 3
        self.nsym = 6
        self.ntens = 9
        pass

    def registerData(self, name, typ, init_val=None, plot_key=None,
                     dim=None, constant=False):

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

        iam = "{0}.registerData(self,name,**kwargs)".format(self.name)

        if name in self.data_container:
            reportError(iam,"variable {0} already registered".format(name))
            pass

        if typ not in self.data_types:
            reportError(iam,"unrecognized data type: {0}".format(typ))
            pass

        shape = None
        if typ in self.tensor_vars:
            if ( init_val is not None and
                 init_val != "Identity" and
                 not isinstance(init_val,(list,np.ndarray)) ):
                msg = ("{0} data {1} must be a list "
                       "or numpy.ndarray, got {2}".format(typ,name,init_val))
                reportError(iam,msg)
                pass

            if typ == "SymTensor":
                if init_val is None:
                    init_val = np.zeros(self.nsym)
                elif init_val == "Identity":
                    init_val = self.I6
                elif len(init_val) != 6:
                    msg = "length of SymTensor data {0} != 6".format(name)
                    reportError(iam,msg)
                    pass
                pass

            elif typ == "Tensor":
                if init_val is None:
                    init_val = np.zeros(self.ntens)
                elif init_val == "Identity":
                    init_val = self.I9
                elif len(init_val) != 9:
                    msg = "length of Tensor data {0} != 9".format(name)
                    reportError(iam,msg)
                    pass
                pass

            elif typ == "Vector":
                if init_val is None:
                    init_val = np.zeros(self.nvec)
                elif init_val == "Identity":
                    init_val = self.I3
                elif len(init_val) != 3:
                    msg = "length of Vector data {0} != 3".format(name)
                    reportError(iam,msg)
                    pass
                pass

            elif typ == "Matrix":
                if init_val is None:
                    if dim is None:
                        msg="no dim specified for {0}".format(name)
                        reportError(iam,msg)
                        pass
                    else:
                        if not isinstance(dim,(int,tuple)):
                            msg="bad dim {0} for {1}".format(dim,name)
                            reportError(iam,msg)
                        else:
                            init_val = np.zeros(dim)
                            pass
                else:
                    if init_val == "Identity":
                        if not isinstance(dim,int):
                            msg="bad dim {0} for {1}".format(dim,name)
                            reportError(iam,msg)
                            pass
                        init_val = np.eye(dim)
                        pass
                    pass
                pass

            value = np.array(init_val)
            old_value = np.array(init_val)
            stashed_value = np.array(init_val)
            shape = value.shape
            pass

        elif typ == "List":

            if init_val is None: init_val = []

            if not isinstance(init_val,(list,tuple)):
                msg = "List data {0} must be a list".format(name)
                reportError(iam,msg)
                pass
            value = [x for x in init_val]
            old_value = [x for x in init_val]
            stashed_value = [x for x in init_val]
            shape = len(value)

        elif typ == "Integer Array":

            if init_val is None: init_val = []

            if not isinstance(init_val,(np.ndarray,list,tuple)):
                msg = "Integer Array data {0} must not be a np.ndarray".format(name)
                reportError(iam,msg)
                pass
            value = np.array([x for x in init_val],dtype=int)
            old_value = np.array([x for x in init_val],dtype=int)
            stashed_value = np.array([x for x in init_val],dtype=int)
            shape = value.shape

        elif typ == "Array":

            if init_val is None: init_val = []

            if not isinstance(init_val,(np.ndarray,list,tuple)):
                msg = "Array data {0} must be a np.ndarray".format(name)
                reportError(iam,msg)
                pass
            value = np.array(init_val)
            old_value = np.array(init_val)
            stashed_value = np.array(init_val)
            shape = value.shape

        elif typ == "Scalar":

            if init_val is None: init_val = 0.

            if isinstance(init_val,(list,tuple,np.ndarray)):
                msg = "Scalar data {0} must ba scalar".format(name)
                reportError(iam,msg)
                pass
            value = init_val
            old_value = init_val
            stashed_value = init_val
            shape = 0

        elif typ == "Boolean":

            if init_val is None: init_val = False

            if not isinstance(init_val,bool):
                msg = ("Boolean data {0} must be boolean, got {1}"
                       .format(name,init_val))
                reportError(iam,msg)
                pass

            value = init_val
            old_value = init_val
            stashed_value = init_val
            pass

        plotable = plot_key is not None
        if not plotable: plot_name = None
        if plotable:
            if not isinstance(plot_key,str):
                msg = ("plot_key for {0} must be a string, got {1}"
                       .format(name,plot_key))
                reportError(iam,msg)
                pass

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
                pass

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

        # register the data
        self.data_container[name] = { "name": name,
                                      "plot key": plot_key,
                                      "plot name": plot_name,
                                      "idx": self.data_container_idx,
                                      "type": typ,
                                      "shape": shape,
                                      "value": value,
                                      "old value": old_value,
                                      "stashed value": old_value,
                                      "constant": constant,
                                      "plotable": plotable }
        self.data_container_idx += 1
        setattr(self,name.replace(" ","_").upper(),old_value)
        pass

    def unregisterData(self,name):
        """ unregister data with the data container """
        try: del self.data_container[name]
        except: reportWarning(iam,"attempting to unregistered non-registered data")

    def registerExtraVariables(self,nextra,names,keys,values):
        """ register extra data with the data container """

        iam = ("{0}.registerExtraVariables(self,nextra,names,keys,values)"
               .format(self.name))
        if self.extra_vars_registered:
            reporteError(iam,"extra variables can only be registered once")

        self.extra_vars_registered = True
        self.num_extra = nextra

        for i in range(nextra):
            name = names[i]
            key = keys[i]
            value = values[i]
            self.registerData(name,"Scalar",
                              init_val = value,
                              plot_key = key)
            self.extra_vars_map[i] = name

            continue

        pass

    def registerOption(self,name,val):

        """
            register data to the option container

            INPUT
              name: option name
              val: option value
        """

        iam = "{0}.registerOption(self,name,val)".format(self.name)

        if name in self.option_container:
            reportError(iam,"option {0} already registered".format(name))
            pass

        self.option_container[name] = val
        setattr(self,name.replace(" ","_").upper(),val)
        pass

    def getAllOptions(self):
        return self.option_container

    def getOption(self,name):

        """ return option[name] """

        iam = "{0}.registerOption(self,name,val)".format(self.name)

        try:
            return self.option_container[name]

        except KeyError:
            msg = ("{0} not in {1}.option_container. registered options are:\n{2}."
                   .format(name,self.name,", ".join(self.option_container.keys())))
            reportError(iam,msg)

        except:
            raise

        pass

    def getData(self,name,stash=False,cur=False,form="Array"):

        """ return simulation_data[name][valtyp] """

        iam = "{0}.getData(self,name)".format(self.name)

        if stash and cur:
            reportError(iam,"cannot get stash and cur simultaneously")
            pass

        if stash: valtyp = "stashed value"
        elif cur: valtyp = "value"
        else: valtyp = "old value"

        # handle extra variables
        if name == "extra variables":
            ex = [None]*self.num_extra
            for idx, name in self.extra_vars_map.items():
                ex[idx] = self.data_container[name][valtyp]
                continue
            return np.array(ex)

        try: data = self.data_container[name]
        except KeyError:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            reportError(iam,msg)
        except:
            raise

        typ = data["type"]

        if typ in self.tensor_vars:

            if form == "Array":
                return np.array(data[valtyp])

            elif form == "MIG":
                return toMig(data[valtyp])

            elif form == "Matrix":
                if typ == "Vector":
                    reportError(iam,"cannont return vector matrix")
                    pass

                return toMatrix(data[valtyp])

            else:
                reportError(iam,"unrecognized form {0}".format(form))
                pass

        elif typ == "List":
            return [x for x in data[valtyp]]

        elif typ == "Integer Array":
            return np.array([x for x in data[valtyp]],dtype=int)

        elif typ == "Array":
            return np.array(data[valtyp])

        else:
            return data[valtyp]

        return

    def restoreData(self,name,newval):
        self.storeData(name,newval)
        self.storeData(name,newval,old=True)
        self.storeData(name,newval,stash=True)
        pass

    def storeData(self,name,newval,stash=False,old=False):

        """ store the simulation data """

        iam = "{0}.storeData(self,name,newval)".format(self.name)

        if old and stash:
            reportError(iam,"can only store old or stash not both")

        if stash: valtyp = "stashed value"
        elif old: valtyp = "old value"
        else: valtyp = "value"

        # handle extra variables
        if name == "extra variables":
            if len(newval) != self.num_extra:
                reportError(iam,"wrong size for extra variable array")
                pass

            for ixv,xv in enumerate(newval):
                name = self.getExName(ixv)
                self.data_container[name][valtyp] = xv
                continue
            return

        try: data = self.data_container[name]
        except KeyError:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            reportError(iam,msg)
        except:
            raise

        typ = data["type"]

        if data["constant"]:
            return

        if typ in self.tensor_vars:

            if newval.shape == (3,3):
                if typ == "Tensor":
                    newval = toArray(newval,symmetric=False)
                elif typ == "SymTensor":
                    newval = toArray(newval)
                else:
                    reportError(iam,"vector cannot be converted from matrix")
                    pass
                pass

            # check lengths
            if typ == "Vector" and len(newval) != 3:
                reportError(iam,"len Vector data {0} != 3".format(name))
            elif typ == "SymTensor" and len(newval) != 6:
                reportError(iam,"len SymTensor data {0} != 6".format(name))
            elif typ == "Tensor" and len(newval) != 9:
                reportError(iam,"len Tensor data {0} != 9".format(name))
            else: pass

            # store the newval
            data[valtyp] = np.array(newval)

        elif typ == "List":
            data[valtyp] = [x for x in newval]

        elif typ == "Integer Array":
            data[valtyp] = np.array([x for x in newval],dtype=int)

        elif typ == "Array":
            data[valtyp] = np.array(newval)

        else:
            # store the scalar variable
            data[valtyp] = newval
            pass

        return

    def stashData(self,name,cur=False):

        """ stash "old value" in "stashed value" """

        iam = "{0}.stashData(self,name,newval)".format(self.name)

        # handle extra variables
        if name == "extra variables":
            for idx,name in self.extra_vars_map:
                value = self.getData(name,cur=cur)
                # stash the value
                self.storeData(name,value,stash=True)
                continue
            return

        value = self.getData(name,cur=cur)
        # stash the value
        self.storeData(name,value,stash=True)

        return

    def getStashedData(self,name):
        return self.getData(name,stash=True)

    def unstashData(self,name):

        """ unstash "value" from "stashed value" """

        iam = "{0}.unstashData(self,name,newval)".format(self.name)

        # handle extra variables
        if name == "extra variables":

            for idx, name in self.extra_vars_map.items():
                value = self.getStashedData(name)
                self.storeData(name,value,old=True)
                continue
            return

        if name not in self.data_container:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            reportError(iam,msg)
            pass

        value = self.getStashedData(name)
        self.storeData(name,value,old=True)

        return

    def advanceAllData(self):

        """ advance "value" to "old value" """

        for name in self.data_container:
            self.advanceData(name)
            continue
        return

    def advanceData(self,name,value=None):

        """ advance "value" to "old value" """

        iam = "{0}.advanceData(self,name)".format(self.name)

        if name == "extra variables":
            if value is not None:
                if len(value) != self.num_extra:
                    reportError(iam,"len(value [{0:d}]) != num_extra [{1:d}]"
                                .format(len(value),self.num_extra))
                    pass
                for idx, exval in enumerate(value):
                    name = self.extra_vars_map[idx]
                    self.storeData(name,exval,old=True)
                    continue
                pass
            else:

                for idx, name in self.extra_vars_map.items():
                    value = self.getData(name,cur=True)
                    self.storeData(name,value,old=True)
                    continue
                pass
            setattr(self,name.replace(" ","_").upper(),value)
            return

        if name not in self.data_container:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            reportError(iam,msg)
            pass

        if value is None: value = self.getData(name,cur=True)
        self.storeData(name,value,old=True)
        setattr(self,name.replace(" ","_").upper(),value)

        return

    def getExName(self,idx):
        try: return self.extra_vars_map[idx]
        except KeyError:
            msg = "{0:d} not in {1}.extra_vars_map.".format(idx,self.name)
            reportError(iam,msg)
        except:
            raise
        pass

    def getPlotKey(self,name):
        try: data = self.data_container[name]
        except KeyError:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            reportError(iam,msg)
        except:
            raise
        return data["plot key"]

    def getPlotName(self,name):
        try: data = self.data_container[name]
        except KeyError:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            reportError(iam,msg)
        except:
            raise
        return data["plot name"]

    def plotKeys(self):

        """ return a list of plot keys in the order registered """

        iam = "{0}.plotKeys(self)".format(self.name)

        plot_keys = [None]*len(self.data_container)

        for key, val in self.data_container.items():

            # skip non plotable output
            if not val["plotable"]: continue

            plot_keys[val["idx"]] = val["plot key"]

            continue

        plot_keys = [x for x in plot_keys if x is not None]

        # return flattened plot keys
        return flatten(plot_keys)

    def plotData(self):

        """
            return list of current values of all plotable data in order registered
        """

        iam = "{0}.plotableData(self)".format(self.name)

        plot_data = [None]*len(self.data_container)

        for name, dic in self.data_container.items():

            # skip non plotable output
            if not dic["plotable"]:
                continue

            idx = dic["idx"]
            value = self.getData(name)
            plot_data[idx] = value

            continue

        plot_data = [x for x in plot_data if x is not None]

        # return flattened plot keys
        return flatten(plot_data)

    def plotable(self,name):
        return self.data_container[name]["plotable"]

    def dumpData(self,name):

        """
            return self.data_container[name]
        """

        iam = "{0}.dumpData(self)".format(self.name)

        if "extra variables" in name:
            ex_vars = {}
            for idx,name in self.extra_vars_map.items():
                ex_vars[name] = self.data_container[name]
                continue
            return ex_vars

        try: return self.data_container[name]
        except KeyError:
            msg = ("{0} not in {1}.data_container. registered data are:\n{2}."
                   .format(name,self.name,", ".join(self.data_container.keys())))
            reportError(iam,msg)
        except:
            raise

        return


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
            pass

        if comp: return comp
        else: reportError(__file__,"bad mapping")
        pass

