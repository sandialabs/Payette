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


#from __future__ import print_function
import os
import sys
import re
import imp
import math
import numpy as np
import time

from Source.Payette_utils import *
from Source.Payette_material import Material
from Source.Payette_data_container import DataContainer
import Source.Payette_installed_materials as pim
import Source.Payette_driver as pd


PARSE_ERRORS = 0


@CountCalls
def parser_error(msg):
    print("ERROR: {0:s}".format(msg))
    sys.exit(101)
    return


class Payette:
    """
    CLASS NAME
       Payette

    PURPOSE
       main container class for a Payette single element simulation, instantiated in
       the runPayette script. The documentation is currently sparse, but may be
       expanded if time allows.

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    """

    def __init__(self, simname, user_input, opts):

        iam = simname + ".__init__"

        if opts.debug:
            opts.verbosity = 4
        loglevel = opts.verbosity

        delete = not opts.keep
        basedir = os.getcwd()

        self.name = simname
        self.disp = opts.disp

        # check user input for required blocks
        req_blocks = ("material", )
        for block in req_blocks:
            if block not in user_input:
                parser_error("{0} block not found in input file".format(block))

        outfile = os.path.join(basedir, simname + ".out")
        if delete and os.path.isfile(outfile):
            os.remove(outfile)

        elif os.path.isfile(outfile):
            i = 0
            while True:
                outfile = os.path.join(basedir, "%s.%i.out" % (simname, i))
                if os.path.isfile(outfile):
                    i += 1
                    if i > 100:
                        parser_error("Come on!  Really, over 100 output files???")

                    else:
                        continue
                else:
                    break
                continue

        # logfile
        logfile = "{0}.log".format(os.path.splitext(outfile)[0])
        try:
            os.remove(logfile)
        except OSError:
            pass

        setupLogger(logfile,loglevel)

        msg = "setting up simulation {0}".format(simname)
        reportMessage(iam, msg)

        # file name for the Payette restart file
        self.is_restart = False
        rfile = "{0}.prf".format(os.path.splitext(outfile)[0])

        # set up the material
        self.material = get_material(user_input.get("material"))
        self.matdat = self.material.material_data()

        # set up boundary and leg blocks
        bargs = [user_input.get("boundary"), user_input.get("legs")]
        get_boundary = parse_boundary_block
        if self.material.eos_model:
            get_boundary = parse_eos_boundary_block

        boundary = get_boundary(*bargs)
        t0 = boundary["initial time"]
        tf = boundary["termination time"]
        bcontrol = boundary["bcontrol"]
        lcontrol = boundary["lcontrol"]

        # set up the simulation data container and register obligatory data
        self.simdat = DataContainer(simname)
        self.simdat.register_data("time", "Scalar",
                                 init_val=0., plot_key="time")
        self.simdat.register_data("time step","Scalar",
                                 init_val=0., plot_key="timestep")
        self.simdat.register_data("number of steps","Scalar", init_val=0)
        self.simdat.register_data("leg number","Scalar", init_val=0 )
        self.simdat.register_data("leg data", "List", init_val=lcontrol)


        # get mathplot
        mathplot = parse_mathplot_block(user_input.get("mathplot"))
        self.simdat.register_option("mathplot vars", mathplot)
        if mathplot:
            math1 = os.path.join(basedir, simname + ".math1")
            math2 = os.path.join(basedir, simname + ".math2")
            self.simdat.register_option("math1", math1)
            self.simdat.register_option("math2", math2)

        # get extraction
        extract = parse_extraction_block(user_input.get("extraction"))
        self.simdat.register_option("extraction", extract)

        # check if user has specified simulation options
        for item in user_input["content"]:
            for pat, repl in ((",", " "), (";", " "), (":", " "), ):
                item = item.replace(pat, repl)
                continue
            item = item.split()

            if len(item) == 1:
                item.append("True")

            try:
                val = eval(item[1])
            except:
                val = str(item[1])

            self.simdat.register_option(item[0], val)
            continue

        # register default options
        self.simdat.register_option("simname", simname)
        self.simdat.register_option("outfile", outfile)
        self.simdat.register_option("logfile", logfile)
        self.simdat.register_option("loglevel", loglevel)
        self.simdat.register_option("verbosity", opts.verbosity)
        self.simdat.register_option("sqa", opts.sqa)
        self.simdat.register_option("debug", opts.debug)
        self.simdat.register_option("material", self.material)

        if "strict" not in self.simdat.get_all_options():
            self.simdat.register_option("strict",False)
            pass

        if "nowriteprops" not in self.simdat.get_all_options():
            self.simdat.register_option("nowriteprops", opts.nowriteprops)

        if "norestart" in self.simdat.get_all_options():
            self.simdat.register_option("write restart",False)

        else:
            self.simdat.register_option("write restart", not opts.norestart)

        # write out properties
        if not self.simdat.NOWRITEPROPS:
            self.write_mtl_params()

        if not self.material.eos_model:
            # register data not needed by the eos models
            self.simdat.register_option("emit", bcontrol["emit"])
            self.simdat.register_option("screenout", bcontrol["screenout"])
            self.simdat.register_option("nprints", bcontrol["nprints"])
            self.simdat.register_option("kappa", bcontrol["kappa"])
            self.simdat.register_option("legs", lcontrol)
            self.simdat.register_option("write vandd table",
                                       opts.write_vandd_table)
            self.simdat.register_option("initial time", t0)
            self.simdat.register_option("termination time", tf)
            self.simdat.register_option("test restart", opts.testrestart)
            self.simdat.register_option("use table", opts.use_table)
            self.simdat.register_option("restart file", rfile)

            # Below are obligatory options that may have been specified in the
            # input file.
            if "proportional" not in self.simdat.get_all_options():
                self.simdat.register_option("proportional",False)
        else:
            # register data that is needed by the EOS models
            for dict_key in bcontrol:
                self.simdat.registerOption(dict_key, bcontrol[dict_key])

        if parser_error.count():
            sys.exit("Stopping due to {0} previous parsing errors"
                     .format(parser_error.count()))

        pass

    def run_job(self, *args, **kwargs):
        if not self.material.eos_model:
            retcode = pd.solid_driver(self, restart=self.is_restart)

        else:
            retcode = pd.eos_driver(self)

        reportMessage("run_job()", "{0} Payette simulation ran to completion"
                  .format(self.name))

        if not self.disp:
            return retcode

        else:
            return {"retcode": retcode}

    def finish(self):

        # close the log and output files
        closeFiles()

        # write the mathematica files
        if self.simdat.MATHPLOT_VARS:
            writeMathPlot(self.simdat, self.matdat)

        # extract requested variables
        if self.simdat.EXTRACTION:
            write_extraction(self.simdat, self.matdat)

        return

    def simulation_data(self):
        return self.simdat

    def write_mtl_params(self):
        with open(self.simdat.SIMNAME + ".props", "w" ) as f:
            for item in self.matdat.PARAMETER_TABLE:
                key = item["name"]
                val = item["adjusted value"]
                f.write("{0:s} = {1:12.5E}\n".format(key,val))
                continue
            pass
        return

    def setup_restart(self):
        iam = self.name + "setup_restart"
        setupLogger(self.simdat.LOGFILE,self.simdat.LOGLEVEL,mode="a")
        msg = "setting up simulation %s"%self.simdat.SIMNAME
        self.is_restart = True
        reportMessage(iam, msg)


def parse_first_leg(leg):
    """Parse the first leg of the legs block.

    The first leg of the legs block may be in one of two forms.  The usual

              <leg_no>, <leg_t>, <leg_steps>, <leg_cntrl>, <c[ij]>

    or, if the user is prescribing the legs through a table

              using <time, dt>, <deformation type> [from columns ...]

    here, we determine what kind of legs the user is prescrbing.

    Parameters
    ----------
    leg: list
        First leg in the legs block of the user input

    Returns
    -------
    leg_inf: dict

    Raises
    ------

    See also
    --------

    Notes
    -----

    Examples
    --------
    >>> parse_first_leg([0, 0., 0, 222222, 0, 0, 0, 0, 0, 0])
    {"table": False}

    >>> parse_first_leg(["using", "dt", "strain"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> parse_first_leg(["using", "dt", "strain", "from", "columns", "1:7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> parse_first_leg(["using", "dt", "stress", "from",
                         "columns", "1,2,3,4,5,6,7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> parse_first_leg(["using", "dt", "strain", "from", "columns", "1-7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> parse_first_leg(["using", "dt", "strain", "from", "columns", "1,5-10"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": [0,4,5,6,7,8,9,20]}

    >>> parse_first_leg(["using", "dt", "strain", "from", "columns", "1,5-7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": [0,4,5,6]}

    """

    errors = 0
    allowed_legs = {
        "strain rate": {"num": 1, "len": 6},
        "strain": {"num": 2, "len": 6},
        "stress rate": {"num": 3, "len": 6},
        "stress": {"num": 4, "len": 6},
        "deformation gradient": {"num": 5, "len": 9},
        "electric field": {"num": 6, "len": 3},
        "displacement": {"num": 8, "len": 3},
        "vstrain": {"num": 2, "len": 1}}
    allowed_t = ["time", "dt"]

    if "using" not in leg[0]:
        return {"table": False}

    t_typ = leg[1]

    if t_typ not in allowed_t:
        errors += 1
        msg = ("requested bad time type {0} in {1}, expected one of [{2}]"
               .format(t_typ, leg, ", ".join(allowed_t)))
        parser_error(msg)

    col_spec =[x for x in leg[2:] if "from" in x or "column" in x]

    if not col_spec:
        # default value for col_idxs
        use_typ = " ".join(leg[2:])
        if use_typ not in allowed_legs:
            parser_error("requested bad control type {0}".format(use_typ))

        col_idxs = range(allowed_legs[use_typ]["len"] + 1)

    elif col_spec and len(col_spec) != 2:
        # user specified a line of the form
        # using <dt,time> <deftyp> from ...
        # or
        # using <dt,time> <deftyp> columns ...
        msg = ("expected {0} <deftyp> from columns ..., got {1}"
               .format(t_typ, leg))
        parser_error(msg)

    else:
        use_typ = " ".join(leg[2:leg.index(col_spec[0])])
        if use_typ not in allowed_legs:
            parser_error("requested bad control type {0}".format(use_typ))

        # now we need to find the column indexes
        col_idxs = " ".join(leg[leg.index(col_spec[-1]) + 1:])
        col_idxs = col_idxs.replace("-", ":")

        if ":" in col_idxs.split():
            tmpl = col_idxs.split()

            # user may have specified something like 1 - 6 which would now be
            # 1 : 6, which is a bad range specifier, we need to fix it
            idx = tmpl.index(":")

            if len(tmpl) == 3 and idx == 1:
                # of form: from columns 1:7
                col_idxs = "".join(tmpl)

            elif len(tmpl) == 4 and idx != 2:
                # of form: from columns 1:6, 7 -> not allowed
                parser_error("bad column range specifier in: '{0}'"
                             .format(" ".join(leg)))

            elif len(tmpl) == 4:
                # of form: from columns 1, 2:7
                col_idxs = tmpl[0] + " " + "".join(tmpl[1:])

        if col_idxs.count(":") > 1:
            # only one range allowed
            parser_error("only one column range supported".format(use_typ))

        col_idxs = col_idxs.split()
        if len(col_idxs) == 1 and not [x for x in col_idxs if ":" in x]:
            # of form: from columns 8 -> not allowed
            parser_error("not enough columns specified in: '{0}'"
                         .format(" ".join(leg)))

        elif len(col_idxs) == 1 and [x for x in col_idxs if ":" in x]:
            # of form: from columns 2:8
            col_idxs = col_idxs[0].split(":")
            col_idxs = range(int(col_idxs[0]) - 1, int(col_idxs[1]))

        elif len(col_idxs) == 2 and [x for x in col_idxs if ":" in x]:
            # specified a single index and range
            if col_idxs.index([x for x in col_idxs if ":" in x][0]) != 1:
                # of form: from columns 2:8, 1 -> not allowed
                parser_error("bad column range specifier in: '{0}'"
                             .format(" ".join(leg)))
            else:
                # of form: from columns 1, 2:8
                tmp = col_idxs[1].split(":")
                col_idxs = [int(col_idxs[0]) - 1]
                col_idxs.extend(range(int(tmp[0]) - 1, int(tmp[1])))
        else:
            # specified all columns individually, convert to 0 index
            col_idxs = [int(x) - 1 for x in col_idxs]

        # we have now parsed the first line, assemble leg_ing
        if len(col_idxs) > allowed_legs[use_typ]["len"] + 1:
            parser_error("too many columns specified")

    # we have exhausted all ways of specifying columns that I can think of,
    # save the info and return
    leg_inf = {
        "table": True,
        "col_idxs": col_idxs,
        "ttyp": t_typ,
        "deftyp": use_typ,
        "len": allowed_legs[use_typ]["len"],
        "control":("{0}".format(allowed_legs[use_typ]["num"])
                   * allowed_legs[use_typ]["len"])
        }

    return leg_inf


def get_material(material_inp=None):
    """ Read material block from input file, parse it, and create material
    object

    Parameters
    ----------
    material_inp : dict
        input material block, split on lines

    Returns
    -------
    material : object
        the instantiated material object

    """

    if material_inp is None:
        return None

    material = material_inp["content"]

    model_nam = None
    user_params = []
    user_options = {"code": "python"}

    # check for required input
    for item in material:
        if "constitutive model" in item:
            # line defines the constitutive model, get the name of the model
            # to be used
            model_nam = item[len("constitutive model"):].strip()
            continue

        if "options" in item:
            options = item[len("options"):].strip().split()
            if "fortran" in options:
                user_options["code"] = "fortran"
            continue

        user_params.append(item)
        continue

    if model_nam is None:
        # constitutive model not given, exit
        parser_error("no constitutive model in material block")

    # constitutive model given, now see if it is available, here we replace
    # spaces with _ in all model names and convert to lower case
    model_nam = model_nam.lower().replace(" ","_")
    constitutive_model = None
    for key, val in pim.PAYETTE_CONSTITUTIVE_MODELS.items():
        if model_nam == key or model_nam in val["aliases"]:
            constitutive_model = key
        continue

    if constitutive_model is None:
        parser_error("constitutive model {0} not found".format(model_nam))

    # instantiate the material object
    material = Material(constitutive_model, user_params, **user_options)

    return material


def parse_boundary_block(*args, **kwargs):
    """Scan the user input for a begin boundary .. end boundary block and
    parse it

    Parameters
    ----------
    args : list
        args[0]
          boundary_inp : dict
            the boundary block
        args[1]
          legs_inp : dict
            the legs block

    Returns
    -------
    boundary : dict
      keys
        t0 : float
          initial time
        tf : float
          final time
        bcontrol : dict
          parsed boundary block
        lcontrol : dict
          parsed leg block

    """

    iam = "parse_boundary_block"

    boundary_inp, legs_inp = args[0:2]
    if boundary_inp is None:
        parser_error(iam, "boundary block not found")

    if legs_inp is None:
        parser_error(iam, "legs block not found")

    boundary = boundary_inp["content"]
    legs = legs_inp["content"]

    # parse boundary
    bcontrol = {"kappa":0.,"estar":1.,"tstar":1.,"ampl":1.,"efstar":1.,
                "ratfac":1.,"sstar":1.,"fstar":1.,"dstar":1.,
                "emit":"all","nprints":0,"stepstar":1,"screenout":False}
    for item in boundary:
        item = item.replace("="," ")
        try:
            kwd, val = item.split()
        except:
            parser_error("boundary control items must be key = val pairs")

        try:
            kwd, val = kwd.strip(),float(eval(val))
        except:
            kwd, val = kwd.strip(),str(val)

        if kwd in bcontrol.keys():
            bcontrol[kwd] = val

        else:
            continue

        continue

    if bcontrol["emit"] not in ("all", "sparse"):
        parser_error("emit must be one of [all,sparse]")

    if bcontrol["screenout"]:
        bcontrol["screenout"] = True

    if not isinstance(bcontrol["nprints"], int):
        bcontrol["nprints"] = int(bcontrol["nprints"])

    if not isinstance(bcontrol["stepstar"], (float,int)):
        bcontrol["stepstar"] = max(0.,float(bcontrol["stepstar"]))

    if bcontrol["stepstar"] <= 0:
        parser_error("stepstar must be > 0.")

    # the following are from Brannon's MED driver
    # estar is the "unit" of strain
    # sstar is the "unit" of stress
    # fstar is the "unit" of deformation gradient
    # efstar is the "unit" of electric field
    # dstar is the "unit" of electric field
    # tstar is the "unit" of time
    # All strains are multiplied by efac=ampl*estar
    # All stresses are multiplied by sfac=ampl*sstar
    # All deformation gradients are multiplied by ffac=ampl*fstar
    # All electric fields are multiplied by effac=ampl*efstar
    # All displacements are multiplied by dfac=ampl*dstar
    # All times are multiplied by tfac=abs(ampl)*tstar/ratfac
    # From these formulas, note that AMPL may be used to increase or decrease
    # the peak strain without changing the strain rate. ratfac is the
    # multiplier on strain rate and stress rate.
    bcontrol["tfac"] = (abs(bcontrol["ampl"]) *
                        bcontrol["tstar"] / bcontrol["ratfac"])
    bcontrol["efac"] = bcontrol["ampl"] * bcontrol["estar"]
    bcontrol["sfac"] = bcontrol["ampl"] * bcontrol["sstar"]
    bcontrol["ffac"] = bcontrol["ampl"] * bcontrol["fstar"]
    bcontrol["effac"] = bcontrol["ampl"] * bcontrol["efstar"]
    bcontrol["dfac"] = bcontrol["ampl"] * bcontrol["dstar"]

    # parse legs
    lcontrol = []
    kappa = bcontrol["kappa"]
    sigc = False

    # control should be a group of letters describing what type of
    # control type the leg is. valid options are:
    #  1: strain rate control
    #  2: strain control
    #  3: stress rate control
    #  4: stress control
    #  5: deformation gradient control
    #  6: electric field
    #  8: displacement
    for ileg, leg in enumerate(legs):

        leg = [x.strip() for x in leg.replace(","," ").split() if x]

        if ileg == 0:
            g_inf = parse_first_leg(leg)

            if g_inf["table"]:
                # the first leg let us know the user specified a table,
                # continu on to the actual table
                g_leg_no = 0
                g_time = 0.
                continue

        if g_inf["table"]:
            if leg[0].lower() == "time":
                # table could have
                continue

            control = g_inf["control"]
            leg_steps = int(bcontrol["stepstar"])
            leg_no = g_leg_no
            g_leg_no += 1
            if g_inf["ttyp"] == "dt":
                g_time += float(leg[g_inf["col_idxs"][0]])
            else:
                g_time = float(leg[g_inf["col_idxs"][0]])

            leg_t = bcontrol["tfac"] * g_time
            try:
                cij = [float(eval(leg[x])) for x in g_inf["col_idxs"][1:]]
            except:
                parser_error("syntax error in leg {0}".format(leg[0]))

        else:
            if len(leg) < 5:
                parser_error(
                    "leg {0} input must be of form:".format(leg[0]) +
                    "\n       leg number, time, steps, type, c[ij]")
            leg_no = int(float(leg[0]))
            leg_t = bcontrol["tfac"] * float(leg[1])
            leg_steps = int(bcontrol["stepstar"] * float(leg[2]))
            if ileg != 0 and leg_steps == 0:
                parser_error("leg number {0} has no steps".format(leg_no))

            control = leg[3].strip()
            try:
                cij = [float(eval(y)) for y in leg[4:]]
            except:
                parser_error("syntax error in leg {0}".format(leg[0]))

        # control should be a group of letters describing what type of
        # control type the leg is. valid options are:
        #  1: strain rate control
        #  2: strain control
        #  3: stress rate control
        #  4: stress control
        #  5: deformation gradient control
        #  6: electric field
        #  8: displacement
        allwd_cntrl = "1234568"
        if [x for x in control if x not in allwd_cntrl]:
            parser_error("leg control parameters can only be one of "
                         "[{0}] got {1} for leg number {2:d}"
                         .format(allwd_cntrl, control, leg_no))

        if not sigc:
            sigc = bool([x for x in control if x == "3" or x == "4"])

        lcntrl = [int(x) for x in list(control)]

        if len(lcntrl) != len(cij):
            parser_error("length of leg control != number of control "
                         "items in leg {0:d}".format(leg_no))

        # separate out electric fields from deformations
        ef, hold, efcntrl = [], [], []
        for i, j in enumerate(lcntrl):
            if j == 6:
                ef.append(cij[i])
                hold.append(i)
                efcntrl.append(j)
            continue

        ef.extend([0.] * (3 - len(ef)))
        efcntrl.extend([6] * (3 - len(efcntrl)))
        cij = [i for j, i in enumerate(cij) if j not in hold]
        lcntrl = [i for j, i in enumerate(lcntrl) if j not in hold]

        if len(lcntrl) != len(cij):
            parser_error("final length of leg control != number of "
                         "control items in leg {0:d}".format(leg_no))

        reduced_lcntrl = list(set(lcntrl))
        if 5 in reduced_lcntrl:
            # deformation gradient control check
            if len(reduced_lcntrl) != 1:
                parser_error("only components of deformation gradient "
                             "are allowed with deformation gradient "
                             "control in leg {0:d}, got {1}"
                             .format(leg_no, control))

            elif len(cij) != 9:
                parser_error("all 9 components of deformation gradient "
                             "must be specified for leg {0:d}"
                             .format(leg_no))

            else:
                # check for valid deformation
                F = np.array([[cij[0], cij[1], cij[2]],
                              [cij[3], cij[4], cij[5]],
                              [cij[6], cij[7], cij[8]]])
                J = np.linalg.det(F)
                if J <= 0:
                    parser_error("inadmissible deformation gradient in leg "
                                 "{0:d} gave a Jacobian of {1:f}"
                                 .format(leg_no, J))

                # convert F to strain E with associated rotation given by axis
                # of rotation x and angle of rotation theta
                R,V = np.linalg.qr(F)
                U = np.dot(R.T,F)
                if np.max(np.abs(R - np.eye(3))) > epsilon():
                    parser_error(
                        "rotation encountered in leg {0}. ".format(leg_no) +
                        "rotations are not yet supported")

        elif 8 in reduced_lcntrl:
            # displacement control check
            if len(reduced_lcntrl) != 1:
                parser_error("only components of displacment are allowed "
                             "with displacment control in leg {0:d}, got {1}"
                             .format(leg_no, control))

            elif len(cij) != 3:
                parser_error("all 3 components of displacement must "
                             "be specified for leg {0:d}".format(leg_no))

            # convert displacments to strains
            dfac = bcontrol["dfac"]
            # Seth-Hill generalized strain is defined
            # strain = (1/kappa)*[(stretch)^kappa - 1]
            # and
            # stretch = displacement + 1

            # In the limit as kappa->0, the Seth-Hill strain becomes
            # strain = ln(stretch).
            for j in range(3):
                stretch = dfac * cij[j] + 1
                if kappa != 0:
                    cij[j] = 1 / kappa * (stretch ** kappa - 1.)
                else:
                    cij[j] = math.log(stretch)
                continue

             # displacements now converted to strains
            lcntrl = [2, 2, 2]

        if lcntrl == [2]:

            # only one strain value given -> volumetric strain
            ev = cij[0] * bcontrol["efac"]
            if kappa * ev + 1. < 0.:
                parser_error("1 + kappa*ev must be positive")

            if kappa == 0.:
                ev = ev / 3.

            else:
               ev = ((kappa * ev + 1.) ** (1. / 3.) - 1.) / kappa

            lcntrl = [2, 2, 2]
            cij = [ev, ev, ev]
            efac_hold = bcontrol["efac"]
            bcontrol["efac"] = 1.0

        for i, j in enumerate(lcntrl):
            if j == 1 or j == 3:
                cij[i] = bcontrol["ratfac"] * cij[i]

            elif j == 2:
                cij[i] = bcontrol["efac"] * cij[i]
                if kappa * cij[i] + 1. < 0.:
                    parser_error("1 + kappa*c[{0:d}] must be positive"
                                 .format(i))

            elif j == 4:
                cij[i] = bcontrol["sfac"] * cij[i]

            elif j == 5:
                cij[i] = bcontrol["ffac"] * cij[i]

            elif j == 6:
                cij[i] = bcontrol["effac"] * cij[i]

            continue

        try:
            bcontrol["efac"] = efac_hold
        except:
            pass

        # fill in cij and lcntrl so that their lengths are always 9
        cij.extend([0.] * (9 - len(cij)))
        lcntrl.extend([0] * (9 - len(lcntrl)))

        # append leg control
        # the electric field control is added to the end of lcntrl
        lcontrol.append([leg_no,
                         leg_t,
                         leg_steps,
                         lcntrl + efcntrl,
                         np.array(cij + ef)])
        continue

    if sigc:
        # stress and or stress rate is used to control this leg. For
        # these cases, kappa is set to 0. globally.
        if kappa != 0.:
            reportWarning(
                iam,
                "WARNING: stress control boundary conditions "
                "only compatible with kappa=0. kappa is being "
                "reset to 0. from %f\n"%kappa)
            bcontrol["kappa"] = 0.

    # check that time is monotonic in lcontrol
    i = -0.001
    t0 = None
    for leg in lcontrol:
        if t0 is None:
            t0 = leg[1]
        if leg[1] < i:
            print(leg[1],i)
            print(leg)
            parser_error("time must be monotonic in from {0:d} to {1:d}"
                         .format(int(leg[0] - 1), int(leg[0])))

        i = leg[1]
        tf = leg[1]
        continue

    return {"initial time": t0, "termination time": tf,
            "bcontrol": bcontrol, "lcontrol": lcontrol}

def parse_eos_boundary_block(*args, **kwargs):
    # @msw: I have put None for all of the entries, but that might break
    # something down stream?
    boundary_inp, legs_inp = args[:2]

    # Initialize bcontrol
    bcontrol = {}
    lcontrol = []


    # parse parameters. If multiple declarations, use the last.


    #
    #                     LEGS
    #
    if legs_inp:
        for tok in legs_inp["content"]:
            vals = [float(x) for x in tok.split()]
            if len(vals) != 2:
                parse_error("unacceptable entry in legs:\n" + tok)
            lcontrol.append(vals)

    #
    #                     BOUNDARY
    #

    bcontrol["nprints"] = 4
    for tok in boundary_inp["content"]:
        if tok.startswith("nprints"):
            nprints = int("".join(tok.split()[1:2]))
            bcontrol["nprints"] = nprints

    bcontrol["input units"] = None
    recognized_unit_systems = ["MKSK","CGSEV"]
    for tok in boundary_inp["content"]:
        if tok.startswith("input units"):
            input_units = tok.split()[2]
            if input_units.upper() not in recognized_unit_systems:
                parser_error("Unrecognized input unit system.")
            bcontrol["input units"] = input_units
    if bcontrol["input units"] == None:
        parser_error("Missing 'input units XYZ' keyword in boundary block.\n"
                     "Please include that line with one of the following\n"
                     "unit systems:\n" + "\n".join(recognized_unit_systems))

    bcontrol["output units"] = None
    for tok in boundary_inp["content"]:
        if tok.startswith("output units"):
            output_units = tok.split()[2]
            if output_units.upper() not in recognized_unit_systems:
                parser_error("Unrecognized output unit system.")
            bcontrol["output units"] = output_units
    if bcontrol["output units"] == None:
        parser_error("Missing 'output units XYZ' keyword in boundary block.\n"
                     "Please include that line with one of the following\n"
                     "unit systems:\n" + "\n".join(recognized_unit_systems))

    bcontrol["density range"] = [0.0, 0.0]
    for tok in boundary_inp["content"]:
        if tok.startswith("density range"):
            bounds = [float(x) for x in tok.split()[2:4]]
            if len(bounds) != 2 or bounds[0] == bounds[1]:
                parser_error("Unacceptable density range in boundary block.")
            bcontrol["density range"] = sorted(bounds)

    bcontrol["temperature range"] = [0.0, 0.0]
    for tok in boundary_inp["content"]:
        if tok.startswith("temperature range"):
            bounds = [float(x) for x in tok.split()[2:4]]
            if len(bounds) != 2 or bounds[0] == bounds[1]:
                parser_error("Unacceptable temperature range in boundary block.")
            bcontrol["temperature range"] = sorted(bounds)

    bcontrol["surface increments"] = 10
    for tok in boundary_inp["content"]:
        if tok.startswith("surface increments"):
            n_incr = int("".join(tok.split()[2:3]))
            if n_incr <= 0:
                parser_error("Number of surface increments must be positive non-zero.")
            bcontrol["surface increments"] = int("".join(tok.split()[2:]))

    bcontrol["path increments"] = 100
    for tok in boundary_inp["content"]:
        if tok.startswith("path increments"):
            bcontrol["path increments"] = int("".join(tok.split()[2]))

    bcontrol["path isotherm"] = None
    for tok in boundary_inp["content"]:
        if tok.startswith("path isotherm"):
            # isotherm expects [density, temperature]
            isotherm = [float(x) for x in tok.split()[2:4]]
            bad_rho = not (bcontrol["density range"][0] <=
                           isotherm[0] <=
                           bcontrol["density range"][1])
            bad_temp = not (bcontrol["temperature range"][0] <=
                                isotherm[1] <=
                                bcontrol["temperature range"][1])
            if len(bounds) != 2 or bad_rho or bad_temp:
                parser_error("Bad initial state for isotherm.")
            bcontrol["path isotherm"] = isotherm

    bcontrol["path hugoniot"] = None
    for tok in boundary_inp["content"]:
        if tok.startswith("path hugoniot"):
            # isotherm expects [density, temperature]
            hugoniot = [float(x) for x in tok.split()[2:4]]
            bad_rho = not (bcontrol["density range"][0] <=
                           hugoniot[0] <=
                           bcontrol["density range"][1])
            bad_temp = not (bcontrol["temperature range"][0] <=
                                hugoniot[1] <=
                                bcontrol["temperature range"][1])
            if len(bounds) != 2 or bad_rho or bad_temp:
                parser_error("Bad initial state for hugoniot.")
            bcontrol["path hugoniot"] = hugoniot


    return {"initial time": None, "termination time": None,
            "bcontrol": bcontrol, "lcontrol": lcontrol}

def parse_mathplot_block(mathplot):
    plotable = []
    if mathplot is None:
        return plotable

    for item in mathplot["content"]:
        for pat, repl in ((",", " "), (";", " "), (":", " "), ):
            item = item.replace(pat, repl)
            continue
        plotable.extend(item.split())
        continue
    return [x.upper() for x in plotable]

def parse_extraction_block(extraction):
    extract = []
    if extraction is None:
        return extract
    for item in extraction["content"]:
        for pat, repl in ((",", " "), (";", " "), (":", " "), ):
            item = item.replace(pat, repl)
            continue
        item = item.split()
        for x in item:
            if x[0] not in ("%", "@"):
                msg = "unrecognized extraction request {0}".format(x)
                reportWarning(iam, msg)
                continue
            extract.append(x)
        continue
    return [x.upper() for x in extract]

if __name__ == "__main__":
    sys.exit("Payette_container.py must be called by runPayette")
