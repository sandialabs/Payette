#
#    Copyright (2011) Sandia Corporation.
#    Use and export of this program may require a license from
#    the United States Government.
#
from __future__ import print_function
import sys
import numpy as np

from Source.Payette_utils import *
from Source.Materials.SNL_kayenta import Kayenta as Parent
try:
    from Source.Materials.SNL_kayenta import mtllib
    imported = True
except:
    imported = False
    pass

attributes = {
    "payette material":True,
    "name":'kayenta_qsfail',
    "fortran source":False,
    "build script":"Not_Needed",
    "depends":"kayenta",
    "aliases":["kayenta quasistatic failure"],
    "material type":["mechanical"]
    }

class KayentaQSFail(Parent):
    """
    CLASS NAME
       KayentaQSFail


    METHODS
       updateState
       _check_props
       _set_field

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    def __init__(self):

        Parent.__init__(self)

        self.name = attributes["name"]
        self.aliases = attributes["aliases"]

        # this model is setup for multi-level failure
        self.multi_level_fail = True

        pass

    # Public methods
    def setUp(self,simdat,matdat,user_params,f_params):

        iam = self.name + ".setUp(self,material,props)"

        Parent.setUp(self,simdat,matdat,user_params,f_params)

        # request extra data to be stored
        matdat.registerData("crack flag","Array",
                            init_val = np.zeros(1),
                            plot_key = "CFLG")
        matdat.registerData("failure ratio","Array",
                            init_val = np.zeros(1),
                            plot_key = "FRATIO")
        matdat.registerData("decay","Array",
                            init_val = np.zeros(1),
                            plot_key = "DECAY")
        return

    def updateState(self,simdat,matdat):
        '''
           update the material state based on current state and strain increment
        '''

        iam = self.name + ".setUp(self,material,props)"

        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")
        crkflg = np.array([matdat.getData("crack flag")])

        signew = matdat.getData("stress")
        svnew = matdat.getData("extra variables")

        a = [dt,self.ui,self.dc,d,sigold,svold,crkflg,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        updated_state = mtllib.kayenta_update_state(*a)

        signew,svnew,crkflg,decay,fratio,usm = updated_state

        matdat.storeData("stress",signew)
        matdat.storeData("extra variables",svnew)
        matdat.storeData("failure ratio",fratio)
        matdat.storeData("crack flag",crkflg)
        matdat.storeData("decay",decay)

        return

    # Private methods
    def _check_props(self):
        props = np.array(self.ui0)
        a = [props,props,self.dc,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        return mtllib.kayenta_chk(*a)

    def _set_field(self,*args,**kwargs):
        a = [self.ui,self.ui,self.dc,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        return mtllib.kayenta_rxv(*a)

