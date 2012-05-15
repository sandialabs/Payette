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

from Source.Payette_utils import CountCalls as CountCalls
from Source.Payette_material import Material
from Source.Payette_data_container import DataContainer
import Source.Payette_installed_materials as pim
import Source.Payette_driver as pd
import Source.Payette_utils as pu
import Source.Payette_extract as pe


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
        self.loglevel = opts.verbosity

        delete = not opts.keep
        self.simdir = os.getcwd()

        self.name = simname
        self.disp = opts.disp
        self.write_vandd_table = opts.write_vandd_table
        self._open_files = {}

        # check user input for required blocks
        req_blocks = ("material", )
        for block in req_blocks:
            if block not in user_input:
                parser_error("{0} block not found in input file".format(block))

        tmpnam = os.path.join(self.simdir, simname + ".out")
        if delete and os.path.isfile(tmpnam):
            os.remove(tmpnam)

        elif os.path.isfile(tmpnam):
            i = 0
            while True:
                tmpnam = os.path.join(self.simdir,
                                      "{0}.{1:d}.out".format(simname, i))
                if os.path.isfile(tmpnam):
                    i += 1
                    if i > 100:
                        parser_error("max number of output files exceeded")

                    else:
                        continue
                else:
                    break
                continue
        self.outfile = tmpnam

        # logfile
        self.logfile = "{0}.log".format(os.path.splitext(self.outfile)[0])
        try:
            os.remove(self.logfile)
        except OSError:
            pass

        pu.setup_logger(self.logfile, self.loglevel)

        msg = "setting up simulation {0}".format(simname)
        pu.reportMessage(iam, msg)

        # file name for the Payette restart file
        self.is_restart = False
        self.test_restart = opts.testrestart

        # set up the material
        self.material = _parse_mtl_block(user_input.get("material"))
        self.matdat = self.material.material_data()

        # set up boundary and leg blocks
        bargs = [user_input.get("boundary"), user_input.get("legs")]
        get_boundary = _parse_boundary_block
        if self.material.eos_model:
            get_boundary = _parse_eos_boundary_block

        boundary = get_boundary(*bargs)
        self.t0 = boundary["initial time"]
        self.tf = boundary["termination time"]
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
        self.simdat.register_option("verbosity", opts.verbosity)
        self.simdat.register_option("sqa", opts.sqa)
        self.simdat.register_option("debug", opts.debug)
        self.simdat.register_option("material", self.material)

        if "strict" not in self.simdat.get_all_options():
            self.simdat.register_option("strict", False)

        if "nowriteprops" not in self.simdat.get_all_options():
            self.simdat.register_option("nowriteprops", opts.nowriteprops)

        if "norestart" in self.simdat.get_all_options():
            self.write_restart = False
        else:
            self.write_restart = not opts.norestart
        if self.write_restart:
            self.restart_file = os.path.splitext(self.outfile)[0] + ".prf"

        # write out properties
        if not self.simdat.NOWRITEPROPS:
            self._write_mtl_params()

        if not self.material.eos_model:
            # register data not needed by the eos models
            self.simdat.register_option("emit", bcontrol["emit"])
            self.simdat.register_option("kappa", bcontrol["kappa"])
            self.simdat.register_option("screenout", bcontrol["screenout"])
            self.simdat.register_option("nprints", bcontrol["nprints"])
            self.simdat.register_option("legs", lcontrol)
            self.simdat.register_option("use table", opts.use_table)

            # Below are obligatory options that may have been specified in the
            # input file.
            if "proportional" not in self.simdat.get_all_options():
                self.simdat.register_option("proportional",False)
        else:
            # register data that is needed by the EOS models
            for dict_key in bcontrol:
                self.simdat.register_option(dict_key, bcontrol[dict_key])

        if parser_error.count():
            sys.exit("Stopping due to {0} previous parsing errors"
                     .format(parser_error.count()))

        # list of plot keys for all plotable data
        self.plot_keys = [x for x in self.simdat.plot_keys()]
        self.plot_keys.extend(self.matdat.plot_keys())

        # get mathplot
        self.mathplot_vars = _parse_mathplot_block(
            user_input.get("mathplot"), self.plot_keys)

        # get extraction
        self.extraction_vars = _parse_extraction_block(
            user_input.get("extraction"), self.plot_keys)

        # get output block
        self.out_vars, self.out_format, self.out_nam = _parse_output_block(
            user_input.get("output"), self.plot_keys)

        self._setup_files()

        pass

    # private methods
    def _write_extraction(self):
        """ write out the requested extraction """
        exargs = [self.outfile, "--silent", "--xout"] + self.extraction_vars
        error = pe.extract(exargs)
        return

    def _write_mathplot(self):
        """ Write the $SIMNAME.math1 file for mathematica post processing """

        iam = "_write_mathplot"

        math1 = os.path.join(self.simdir, self.name + ".math1")
        math2 = os.path.join(self.simdir, self.name + ".math2")

        # math1 is a file containing user inputs, and locations of simulation
        # output for mathematica to use
        param_table = self.material.constitutive_model.parameter_table_idx_map
        with open( math1, "w" ) as f:
            # write out user given input
            for idx, nam in sorted(param_table.iteritems()):
                val = self.material.constitutive_model.ui0[idx]
                val = "{0:12.5E}".format(val).replace("E","*^")
                f.write("{0:s}U={1:s}\n".format(nam, val))
                continue

            # write out checked, possibly modified, input
            for idx, nam in sorted(param_table.iteritems()):
                val = self.material.constitutive_model.ui[idx]
                val = "{0:12.5E}".format(val).replace("E","*^")
                f.write("{0:s}M={1:s}\n".format(nam, val))

            # write out user requested plotable output
            f.write('simdat = Delete[Import["{0:s}", "Table"],-1];\n'
                    .format(self.outfile))
            sig_idx = None

            for i, item in enumerate(self.out_vars):
                if item == "SIG11": sig_idx = i + 1
                f.write('{0:s}=simdat[[2;;,{1:d}]];\n'.format(item,i+1))
                continue

            # a few last ones...
            if sig_idx != None:
                pres=("-(simdat[[2;;,{0:d}]]".format(sig_idx) +
                      "+simdat[[2;;,{0:d}]]".format(sig_idx + 1) +
                      "+simdat[[2;;,{0:d}]])/3;".format(sig_idx+2))
                f.write('PRES={0}\n'.format(pres))
            f.write("lastep=Length[{0}]\n".format(self.simdat.get_plot_key("time")))

        # math2 is a file containing mathematica directives to setup default
        # plots that the user requested
        lowhead = [x.lower() for x in self.out_vars]
        lowplotable = [x.lower() for x in self.mathplot_vars]
        with open( math2, "w" ) as f:
            f.write('showcy[{0},{{"cycle","time"}}]\n'
                    .format(self.simdat.get_plot_key("time")))

            for item in self.mathplot_vars:
                name = self.plot_keys[lowhead.index(item.lower())]
                f.write('grafhis[{0:s},"{0:s}"]\n'.format(name))
                continue

        return

    def _close_open_files(self):
        for fnam, fobj in self._open_files.items():
            fobj.flush()
            fobj.close()
            continue
        return

    def _write_mtl_params(self):
        with open(self.name + ".props", "w" ) as f:
            for item in self.matdat.PARAMETER_TABLE:
                key = item["name"]
                val = item["adjusted value"]
                f.write("{0:s} = {1:12.5E}\n".format(key,val))
                continue
        return


    def _setup_out_file(self, file_name):

        if self._open_files.get(file_name) is not None:
            self._open_files[file_name].close()
            del self._open_files[file_name]

        if self.is_restart:
            self.outfile_obj = open(file_name, "a")

        else:
            self.outfile_obj = open(file_name, "w")

            for key in self.out_vars:
                self.outfile_obj.write(pu.textformat(key))
                continue

            self.outfile_obj.write("\n")
            self.outfile_obj.flush()
        self._open_files[file_name] = self.outfile_obj
        return

    def _setup_files(self):

        self._setup_out_file(self.outfile)

        self._write_avail_dat_to_log()

        if self.write_vandd_table:
            # set up files
            vname = os.path.splitext(self.outfile)[0] + ".vtable"
            dname = os.path.splitext(self.outfile)[0] + ".dtable"

            # set up velocity and displacement table files
            if self.is_restart:
                self.vtable_fobj = open(vname, "a")

            else:
                self.vtable_fobj = open(vname, "w")
                default = ["time", "v1", "v2", "v3"]
                for key in default:
                    self.vtable_fobj.write(pu.textformat(key))
                self.vtable_fobj.write("\n")
            self._open_files[vname] = self.vtable_fobj

            if self.is_restart:
                self.dtable_fobj = open(dname, "a")

            else:
                self.dtable_fobj = open(dname, "w")
                default = ["time", "d1", "d2", "d3"]
                for key in default:
                    self.dtable_fobj.write(pu.textformat(key))
                self.dtable_fobj.write("\n")
                self.dtable_fobj.write(pu.textformat(0.))
                for init_disp in [0.] * 3:
                    self.dtable_fobj.write(pu.textformat(init_disp))
                self.dtable_fobj.write("\n")
            self._open_files[dname] = self.dtable_fobj

        return

    def _write_avail_dat_to_log(self):
        """ write to the log file a list of all available data and whether or
        not it was requested for being plotted """

        def _write_plotable(idx, key, name, val):
            """ write to the logfile the available variables """
            tok = "plotable" if key in self.out_vars else "no request"
            pu.write_to_simlog("{0:<3d} {1:<10s}: {2:<10s} = {3:<50s} = {4:12.5E}"
                               .format(idx, tok, key, name, val))
            return

        # write to the log file what is plotable and not requested, along with
        # inital value
        pu.write_to_simlog("Summary of available output")

        for plot_idx, plot_key in enumerate(self.plot_keys):
            if plot_key in self.simdat.plot_keys():
                dat = self.simdat
            else:
                dat = self.matdat
            plot_name = dat.get_plot_name(plot_key)
            val = dat.get_data(plot_key)
            _write_plotable(plot_idx + 1, plot_key, plot_name, val)
            continue

        return

    def write_state(self):
        """ write the simulation and material data to the output file """
        data = []
        for plot_key in self.out_vars:
            if plot_key in self.simdat.plot_keys():
                dat = self.simdat
            else:
                dat = self.matdat
            data.append(dat.get_data(plot_key))
            continue

        for dat in data:
            self.outfile_obj.write(pu.textformat(dat))
            continue
        self.outfile_obj.write('\n')
        return

    def setup_restart(self):
        iam = self.name + "setup_restart"
        self.is_restart = True
        pu.setup_logger(self.logfile, self.loglevel, mode="a")
        msg = "setting up simulation {0}".format(self.name)
        pu.reportMessage(iam, msg)
        self._setup_files()
        return

    def write_vel_and_disp(self, tbeg, tend, epsbeg, epsend):
        """For each strain component, make a velocity and a displacement table
        and write it to a file. Useful for setting up simulations in other
        host codes.

        Parameters
        ----------
        tbeg : float
          time at beginning of step
        tend : float
          time at end of step
        epsbeg : float
          strain at beginning of step
        epsend : float
          strain at end of step

        Background
        ----------
          python implementation of similar function in Rebecca Brannon's MED
          driver

        """
        if not self.write_vandd_table:
            return

        dt = tend - tbeg
        disp, vlcty = np.zeros(3), np.zeros(3)


        psbeg, psend = epsbeg, epsend

        # determine average velocities that will ensure passing through the
        # exact stretches at each interval boundary.
        for j in range(3):
            if self.simdat.KAPPA != 0.:
                # Seth-Hill generalized strain is defined
                # strain = (1/kappa)*[(stretch)^kappa - 1]
                lam0 = (psbeg[j] * kappa + 1.) ** (1. / kappa)
                lam = (psend[j] * kappa + 1.) ** (1. / kappa)

            else:
                # In the limit as kappa->0, the Seth-Hill strain becomes
                # strain = ln(stretch).
                lam0 = math.exp(psbeg[j])
                lam = math.exp(psend[j])

            disp[j] = lam - 1

            # Except for logarithmic strain, a constant strain rate does NOT
            # imply a constant boundary velocity. We will here determine a
            # constant boundary velocity that will lead to the correct value of
            # stretch at the beginning and end of the interval.
            vlcty[j] = (lam - lam0) / dt
            continue

        # displacement
        self.dtable_fobj.write(pu.textformat(tend))

        for item in disp:
            self.dtable_fobj.write(pu.textformat(item))
        self.dtable_fobj.write("\n")

        # jump discontinuity in velocity will be specified by a VERY sharp change
        # occuring from time = tjump - delt to time = tjump + delt
        delt = self.tf * 1.e-9
        if tbeg > self.t0:
            tbeg += delt

        if tend < self.tf:
            tend -= delt

        self.vtable_fobj.write(pu.textformat(tbeg))
        for item in vlcty:
            self.vtable_fobj.write(pu.textformat(item))
        self.vtable_fobj.write("\n")
        self.vtable_fobj.write(pu.textformat(tend))
        for item in vlcty:
            self.vtable_fobj.write(pu.textformat(item))
        self.vtable_fobj.write("\n")
        return

    # public methods
    def run_job(self, *args, **kwargs):
        if not self.material.eos_model:
            retcode = pd.solid_driver(self, restart=self.is_restart)

        else:
            retcode = pd.eos_driver(self)

        pu.reportMessage("run_job()", "{0} Payette simulation ran to completion"
                         .format(self.name))

        if not self.disp:
            return retcode

        else:
            return {"retcode": retcode}

    def finish(self):

        # close the files
        pu.close_aux_files()
        self._close_open_files()

        # write the mathematica files
        if self.mathplot_vars:
            self._write_mathplot()

        # extract requested variables
        if self.extraction_vars:
            self._write_extraction()

        return

    def simulation_data(self):
        return self.simdat


def _parse_first_leg(leg):
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
    >>> _parse_first_leg([0, 0., 0, 222222, 0, 0, 0, 0, 0, 0])
    {"table": False}

    >>> _parse_first_leg(["using", "dt", "strain"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> _parse_first_leg(["using", "dt", "strain", "from", "columns", "1:7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> _parse_first_leg(["using", "dt", "stress", "from",
                         "columns", "1,2,3,4,5,6,7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> _parse_first_leg(["using", "dt", "strain", "from", "columns", "1-7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": range(7)}

    >>> _parse_first_leg(["using", "dt", "strain", "from", "columns", "1,5-10"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": [0,4,5,6,7,8,9,20]}

    >>> _parse_first_leg(["using", "dt", "strain", "from", "columns", "1,5-7"])
    {"table": True, "ttyp": "dt", "deftyp": "strain", "len": 6,
     "colidx": [0,4,5,6]}

    """

    iam = "_parse_first_leg"

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


def _parse_mtl_block(material_inp=None):
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

    iam = "_parse_mtl_block"

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


def _parse_boundary_block(*args, **kwargs):
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

    iam = "_parse_boundary_block"

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
            g_inf = _parse_first_leg(leg)

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
                if np.max(np.abs(R - np.eye(3))) > pu.EPSILON:
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
            pu.reportWarning(
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


def _parse_eos_boundary_block(*args, **kwargs):
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
                parser_error("unacceptable entry in legs:\n" + tok)
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


def _parse_extraction_block(extraction, avail_keys):

    iam = "_parse_extraction_block"

    extraction_vars = []
    if extraction is None:
        return extraction_vars

    for items in extraction["content"]:
        for pat, repl in ((",", " "), (";", " "), (":", " "), ):
            items = items.replace(pat, repl)
            continue
        items = items.split()
        for item in items:
            if item[0] not in ("%", "@") and not item[0].isdigit():
                msg = "unrecognized extraction request {0}".format(item)
                pu.reportWarning(iam, msg)
                continue

            elif item[1:].lower() not in [x.lower() for x in avail_keys]:
                msg = ("requested extraction variable {0} not found"
                       .format(item))
                pu.reportWarning(iam, msg)
                continue

            extraction_vars.append(item)

        continue

    return [x.upper() for x in extraction_vars]

def _parse_mathplot_block(mathplot, avail_keys):

    iam = "_parse_mathplot_block"

    mathplot_vars = []
    if mathplot is None:
        return mathplot_vars

    for item in mathplot["content"]:
        for pat, repl in ((",", " "), (";", " "), (":", " "), ):
            item = item.replace(pat, repl)
            continue
        mathplot_vars.extend(item.split())
        continue

    bad_keys = [x for x in mathplot_vars if x.lower() not in
                [y.lower() for y in avail_keys]]
    if bad_keys:
        msg = (
            "requested mathplot variable{0:s} {1:s} not found"
            .format("s" if len(bad_keys) > 1 else "", ", ".join(bad_keys)))
        pu.reportWarning(iam, msg)

    return [x.upper() for x in mathplot_vars if x not in bad_keys]


def _parse_output_block(output, avail_keys):

    iam = "_parse_output_block"

    supported_formats = ("ascii", )
    out_format = "ascii"
    out_vars = []
    if output is None:
        return avail_keys, out_format, None

    for item in output["content"]:
        for pat, repl in ((",", " "), (";", " "), (":", " "), ):
            item = item.replace(pat, repl)
            continue

        _vars = [x.upper() for x in item.split()]
        if "FORMAT" in _vars:
            try:
                idx = _vars.index("FORMAT")
                out_format = _vars[idx+1].lower()
            except IndexError:
                pu.reportWarning(iam, "format keyword found, but no format given")
            continue

        out_vars.extend(_vars)
        continue

    if out_format not in supported_formats:
        pu.reportError(iam, "output format {0} not supported, choose from {1}"
                       .format(out_format, ", ".join(supported_formats)))

    if "ALL" in out_vars:
        out_vars = avail_keys

    bad_keys = [x for x in out_vars if x.lower() not in
                [y.lower() for y in avail_keys]]
    if bad_keys:
        msg = (
            "requested output variable{0:s} {1:s} not found"
            .format("s" if len(bad_keys) > 1 else "", ", ".join(bad_keys)))
        pu.reportWarning(iam, msg)

    out_vars = [x.upper() for x in out_vars if x not in bad_keys]

    if not out_vars:
        pu.reportError(iam, "no output variables found")

    if "TIME" not in out_vars:
        out_vars.insert(0, "TIME")

    elif out_vars.index("TIME") != 0:
        out_vars.remove("TIME")
        out_vars.insert(0, "TIME")

    return out_vars, out_format, output["name"]


if __name__ == "__main__":
    sys.exit("Payette_container.py must be called by runPayette")
