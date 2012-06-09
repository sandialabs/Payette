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

"""Main Payette class definition"""

#from __future__ import print_function
import os
import sys
import math
import numpy as np
from copy import deepcopy

import Source.Payette_driver as pd
import Source.Payette_utils as pu
import Source.Payette_extract as pe
import Source.Payette_boundary as pb
import Source.runopts as ro
from Source.Payette_material import Material
from Source.Payette_data_container import DataContainer


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

    def __init__(self, simname, user_input):

        # save the user input
        self.user_input = deepcopy(user_input)

        delete = not ro.KEEP
        self.simdir = os.getcwd()

        self.name = simname
        self._open_files = {}

        # default variables
        self.dtable_fobj = None
        self.outfile_obj = None
        self.vtable_fobj = None

        # check user input for required blocks
        req_blocks = ("material", )
        for block in req_blocks:
            if block not in user_input:
                pu.report_and_raise_error(
                    "{0} block not found in input file".format(block),
                    tracebacklimit=0)

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
                        pu.report_and_raise_error(
                            "max number of output files exceeded",
                            tracebacklimit=0)

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

        # set up the logger for the simulation
        pu.setup_logger(self.logfile)
        pu.log_message("setting up simulation {0}".format(simname))

        # file name for the Payette restart file
        self.is_restart = False

        # set up the material
        self.material = _parse_mtl_block(user_input.get("material"))
        self.matdat = self.material.material_data()

        # set up boundary and leg blocks
        if not self.material.eos_model:
            boundary = pb.Boundary(user_input.get("boundary")["content"],
                                   user_input.get("legs")["content"])
        else:
            boundary = pb.EOSBoundary(user_input.get("boundary")["content"],
                                   user_input.get("legs")["content"])

        self.t0 = boundary.initial_time
        self.tf = boundary.termination_time
        bcontrol = boundary.get_boundary_control_params()
        lcontrol = boundary.get_leg_control_params()

        # set up the simulation data container and register obligatory data
        self.simdat = DataContainer(simname)
        self.simdat.register_data("time", "Scalar",
                                 init_val=0., plot_key="time")
        self.simdat.register_data("time step", "Scalar",
                                 init_val=0., plot_key="timestep")
        self.simdat.register_data("number of steps", "Scalar", init_val=0)
        self.simdat.register_data("leg number", "Scalar", init_val=0 )
        self.simdat.register_data("leg data", "List", init_val=lcontrol)

        # check if user has specified simulation options directly in the input
        # file
        for item in user_input["content"]:
            self.user_input["content"].remove(item)
            for pat in ",;:":
                item = item.replace(pat, " ")
                continue
            item = item.split()

            if len(item) == 1:
                item.append("True")

            attr = item[0]
            val = "_".join(item[1:])

            try:
                val = eval(val)

            except (NameError, TypeError):
                val = str(val)

            ro.set_global_option(attr, val)
            continue

        if ro.CHECK_SETUP:
            exit("EXITING to check setup")

        if not ro.NORESTART:
            self.restart_file = os.path.splitext(self.outfile)[0] + ".prf"

        # write out properties
        if not ro.NOWRITEPROPS:
            self._write_mtl_params()

        if not self.material.eos_model:
            # register data not needed by the eos models
            self.simdat.register_static_data("emit", boundary.emit())
            self.simdat.register_static_data("kappa", boundary.kappa())
            self.simdat.register_static_data("screenout", boundary.screenout())
            self.simdat.register_static_data("nprints", boundary.nprints())
            self.simdat.register_static_data("legs", lcontrol)

        else:
            # register data that is needed by the EOS models
            self.simdat.register_static_data("nprints", boundary.nprints())
            self.simdat.register_static_data("input units",
                                             boundary.input_units())
            self.simdat.register_static_data("output_units",
                                             boundary.output_units())
            self.simdat.register_static_data("density_range",
                                             boundary.density_range())
            self.simdat.register_static_data("temperature_range",
                                             boundary.temperature_range())
            self.simdat.register_static_data("surface_increments",
                                             boundary.surface_increments())
            self.simdat.register_static_data("path_increments",
                                             boundary.path_increments())
            self.simdat.register_static_data("path_isotherm",
                                             boundary.path_isotherm())
            self.simdat.register_static_data("path_hugoniot",
                                             boundary.path_hugoniot())

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
        pe.extract(exargs)
        return

    def _write_mathplot(self):
        """ Write the $SIMNAME.math1 file for mathematica post processing """

        math1 = os.path.join(self.simdir, self.name + ".math1")
        math2 = os.path.join(self.simdir, self.name + ".math2")

        # math1 is a file containing user inputs, and locations of simulation
        # output for mathematica to use
        cmod = self.material.constitutive_model
        user_params = cmod.get_parameter_names_and_values()
        adjusted_params = cmod.get_parameter_names_and_values(default=False)
        with open(math1, "w") as fobj:
            # write out user given input
            for name, desc, val in user_params:
                val = "{0:12.5E}".format(val).replace("E", "*^")
                fobj.write("{0:s}U={1:s}\n".format(name, val))
                continue

            # write out checked, possibly modified, input
            for name, desc, val in adjusted_params:
                val = "{0:12.5E}".format(val).replace("E", "*^")
                fobj.write("{0:s}M={1:s}\n".format(name, val))

            # write out user requested plotable output
            fobj.write('simdat = Delete[Import["{0:s}", "Table"],-1];\n'
                    .format(self.outfile))

            sig_idx = None
            for i, item in enumerate(self.out_vars):
                if item == "SIG11":
                    sig_idx = i + 1
                fobj.write('{0:s}=simdat[[2;;,{1:d}]];\n'.format(item, i + 1))
                continue

            # a few last ones...
            if sig_idx is not None:
                pres = ("-(simdat[[2;;,{0:d}]]".format(sig_idx) +
                        "+simdat[[2;;,{0:d}]]".format(sig_idx + 1) +
                        "+simdat[[2;;,{0:d}]])/3;".format(sig_idx+2))
                fobj.write('PRES={0}\n'.format(pres))
            fobj.write(
                "lastep=Length[{0}]\n".format(self.simdat.get_plot_key("time")))

        # math2 is a file containing mathematica directives to setup default
        # plots that the user requested
        lowhead = [x.lower() for x in self.out_vars]
        with open( math2, "w" ) as fobj:
            fobj.write('showcy[{0},{{"cycle", "time"}}]\n'
                    .format(self.simdat.get_plot_key("time")))

            for item in self.mathplot_vars:
                name = self.plot_keys[lowhead.index(item.lower())]
                fobj.write('grafhis[{0:s}, "{0:s}"]\n'.format(name))
                continue

        return

    def _close_open_files(self):
        """close files opened by Payette"""
        for fnam, fobj in self._open_files.items():
            fobj.flush()
            fobj.close()
            continue
        return

    def _write_mtl_params(self):
        """write the material parameters to a file"""
        cmod = self.material.constitutive_model
        params = cmod.get_parameter_names_and_values(default=False)
        with open(self.name + ".props", "w" ) as fobj:
            for param, desc, val in params:
                fobj.write("{0:s} ={1:12.5E}\n".format(param, val))
                continue
        return


    def _setup_out_file(self, file_name):
        """set up the ouput files"""
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
        """set up files"""
        self._setup_out_file(self.outfile)

        self._write_avail_dat_to_log()

        if ro.WRITE_VANDD_TABLE:
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
            pu.write_to_simlog(
                "{0:<3d} {1:<10s}: {2:<10s} = {3:<50s} = {4:12.5E}"
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
        """set up the restart files"""
        self.is_restart = True
        pu.setup_logger(self.logfile, mode="a")
        pu.log_message("setting up simulation {0}".format(self.name))
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
        if not ro.WRITE_VANDD_TABLE:
            return

        delt = tend - tbeg
        disp, vlcty = np.zeros(3), np.zeros(3)


        psbeg, psend = epsbeg, epsend

        # determine average velocities that will ensure passing through the
        # exact stretches at each interval boundary.
        kappa = self.simdat.KAPPA
        for j in range(3):
            if kappa != 0.:
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
            vlcty[j] = (lam - lam0) / delt
            continue

        # displacement
        self.dtable_fobj.write(pu.textformat(tend))

        for item in disp:
            self.dtable_fobj.write(pu.textformat(item))
        self.dtable_fobj.write("\n")

        # jump discontinuity in velocity will be specified by a VERY sharp
        # change occuring from time = tjump - delt to time = tjump + delt
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
        """run the job"""
        if not self.material.eos_model:
            retcode = pd.solid_driver(self, restart=self.is_restart)

        else:
            retcode = pd.eos_driver(self)

        pu.log_message(
            "{0} Payette simulation ran to completion\n\n".format(self.name))

        if not ro.DISP:
            return retcode

        else:
            return {"retcode": retcode}

    def finish(self):
        """finish up"""

        # restore runopts
        ro.restore_default_options()
        pu.reset_error_and_warnings()

        # close the files
        pu.close_aux_files()
        self._close_open_files()

        # write the mathematica files
        if self.mathplot_vars:
            self._write_mathplot()

        # extract requested variables
        if self.extraction_vars:
            self._write_extraction()

        if ro.WRITE_INPUT:
            pu.write_input_file(
                self.name, self.user_input,
                os.path.join(self.simdir, self.name + ".inp"))

        return

    def simulation_data(self):
        """return the simulation simdat object"""
        return self.simdat


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

    if material_inp is None:
        return None

    material = material_inp["content"]

    model_name = None
    user_params = []
    user_options = {}

    # check for required input
    for item in material:
        item = " ".join(item.split())
        if "constitutive model" in item:
            # line defines the constitutive model, get the name of the model
            # to be used
            model_name = item[len("constitutive model"):].strip()
            continue

        if "options" in item:
            options = item[len("options"):].strip().split()
            if "fortran" in options:
                user_options["code"] = "fortran"
            continue

        if "strength model" in item:
            strength_model = item[len("strength model"):].strip().split()
            user_options["strength model"] = strength_model
            continue

        user_params.append(item)
        continue

    if model_name is None:
        # constitutive model not given, exit
        pu.report_and_raise_error("no constitutive model in material block",
                                  tracebacklimit=0)

    # constitutive model given, now see if it is available, here we replace
    # spaces with _ in all model names and convert to lower case
    model_name = model_name.lower().replace(" ", "_")

    # instantiate the material object
    material = Material(model_name, user_params, **user_options)

    return material


def _parse_extraction_block(extraction, avail_keys):
    """parse the extraction block of the input file"""
    extraction_vars = []
    if extraction is None:
        return extraction_vars

    for items in extraction["content"]:
        for pat in ",;:":
            items = items.replace(pat, " ")
            continue
        items = items.split()
        for item in items:
            if item[0] not in ("%", "@") and not item[0].isdigit():
                msg = "unrecognized extraction request {0}".format(item)
                pu.log_warning(msg)
                continue

            elif item[1:].lower() not in [x.lower() for x in avail_keys]:
                msg = ("requested extraction variable {0} not found"
                       .format(item))
                pu.log_warning(msg)
                continue

            extraction_vars.append(item)

        continue

    return [x.upper() for x in extraction_vars]


def _parse_mathplot_block(mathplot, avail_keys):
    """parse the mathplot block of the input file"""

    mathplot_vars = []
    if mathplot is None:
        return mathplot_vars

    for item in mathplot["content"]:
        for pat in ",;:":
            item = item.replace(pat, " ")
            continue
        mathplot_vars.extend(item.split())
        continue

    bad_keys = [x for x in mathplot_vars if x.lower() not in
                [y.lower() for y in avail_keys]]
    if bad_keys:
        msg = (
            "requested mathplot variable{0:s} {1:s} not found"
            .format("s" if len(bad_keys) > 1 else "", ", ".join(bad_keys)))
        pu.log_warning(msg)

    return [x.upper() for x in mathplot_vars if x not in bad_keys]


def _parse_output_block(output, avail_keys):
    """parse the output block of the input file"""

    supported_formats = ("ascii", )
    out_format = "ascii"
    out_vars = []
    if output is None:
        return avail_keys, out_format, None

    for item in output["content"]:
        for pat in ",;:":
            item = item.replace(pat, " ")
            continue

        _vars = [x.upper() for x in item.split()]
        if "FORMAT" in _vars:
            try:
                idx = _vars.index("FORMAT")
                out_format = _vars[idx+1].lower()
            except IndexError:
                pu.log_warning("format keyword found, but no format given")
            continue

        out_vars.extend(_vars)
        continue

    if out_format not in supported_formats:
        pu.report_and_raise_error(
            "output format {0} not supported, choose from {1}"
            .format(out_format, ", ".join(supported_formats)))

    if "ALL" in out_vars:
        out_vars = avail_keys

    bad_keys = [x for x in out_vars if x.lower() not in
                [y.lower() for y in avail_keys]]
    if bad_keys:
        msg = (
            "requested output variable{0:s} {1:s} not found"
            .format("s" if len(bad_keys) > 1 else "", ", ".join(bad_keys)))
        pu.log_warning(msg)

    out_vars = [x.upper() for x in out_vars if x not in bad_keys]

    if not out_vars:
        pu.report_and_raise_error("no output variables found")

    # remove duplicates
    uniq = set(out_vars)
    out_vars = [x for x in out_vars if x in uniq and not uniq.remove(x)]

    if "TIME" not in out_vars:
        out_vars.insert(0, "TIME")

    elif out_vars.index("TIME") != 0:
        out_vars.remove("TIME")
        out_vars.insert(0, "TIME")

    return out_vars, out_format, output["name"]

