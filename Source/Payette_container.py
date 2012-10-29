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

"""Main Payette class definition"""

import os, sys, math, re
import numpy as np
import datetime
import shutil
from copy import deepcopy
from textwrap import fill as textfill

import Source.__config__ as cfg
import Source.__runopts__ as ro
import Source.Payette_driver as pd
import Source.Payette_utils as pu
import Source.Payette_extract as pe
import Source.Payette_boundary as pb
import Source.Payette_input_parser as pip
import Source.Payette_unit_manager as um
from Source.Payette_material import Material
from Source.Payette_data_container import DataContainer
from Source.Payette_boundary import BoundaryError as BoundaryError


class PayetteError(Exception):
    def __init__(self, message, caller=None):

        if caller is None:
            caller = pu.who_is_calling()

        if not ro.DEBUG:
            sys.tracebacklimit = 0
        else:
            sys.tracebacklimit = 10

        # do not report who is calling. Sometimes other functions act as
        # intermediaries to PayetteError and it is nice to leave them out of
        # the "reported by" section
        if caller == "anonymous":
            reported_by = ""
        else:
            reported_by = "[reported by: {0}]".format(caller)

        self.message = (
            "ERROR: {0} {1}"
            .format(" ".join(x for x in message.split() if x), reported_by))

        super(PayetteError, self).__init__(self.message)


class Payette(object):
    """Main container class for a Payette single element simulation,
    instantiated in the payette script. The documentation is currently sparse,
    but may be expanded if time allows.

    """

    def __init__(self, ilines):

        # instantiate the user input object
        self.ilines = ilines
        self.ui = pip.InputParser(ilines)

        delete = not ro.KEEP

        # check if user has specified simulation options directly in the input
        # file
        for attr, val in self.ui.options().items():
            ro.set_global_option(attr, val)
            continue

        # directory to run simulation
        self.simdir = os.getcwd()
        if ro.SIMDIR is not None:
            if ro.SIMDIR[0] == "/":
                # user gave full path to simdir, use it
                self.simdir = ro.SIMDIR
            else:
                self.simdir = os.path.join(self.simdir, ro.SIMDIR)

        if not os.path.isdir(self.simdir):
            try:
                os.makedirs(self.simdir)
            except OSError:
                raise PayetteError(
                    "cannot create simulation directory: {0}"
                    .format(self.simdir))

        self.name = self.ui.name
        self._open_files = {}

        # default variables
        self.dtable_fobj = None
        self.outfile_obj = None
        self.vtable_fobj = None

        tmpnam = os.path.join(self.simdir, self.name + ".out")
        if delete and os.path.isfile(tmpnam):
            os.remove(tmpnam)

        elif os.path.isfile(tmpnam):
            i = 0
            while True:
                tmpnam = os.path.join(self.simdir,
                                      "{0}.{1:d}.out".format(self.name, i))
                if os.path.isfile(tmpnam):
                    i += 1
                    if i > 100:
                        raise PayetteError(
                            "max number of output files exceeded")

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
        pu.log_message("setting up simulation {0}".format(self.name))

        # write some info to the log file
        now = datetime.datetime.now()
        pu.write_to_simlog("Date: {0}".format(now.strftime("%A %B %d, %Y")))
        pu.write_to_simlog("Time: {0}".format(now.strftime("%H:%M:%S")))
        pu.write_to_simlog("Platform: {0}".format(cfg.OSTYPE))
        pu.write_to_simlog("Python interpreter: {0}".format(cfg.PYINT))

        desc = self.ui.find_block("description", "  None")
        desc = textfill(desc, initial_indent="  ", subsequent_indent="  ")
        pu.write_to_simlog(desc)

        # write input to log file
        pu.write_to_simlog("User input:\nbegin input")
        ns = 0
        for line in self.ui.inp.split("\n"):
            if "end" in line.split()[0]:
                ns -= 2
            pu.write_to_simlog(" " * ns + line)
            if "begin" in line:
                ns += 2
            continue
        pu.write_to_simlog("end input")

        # file name for the Payette restart file
        self.is_restart = 0

        # instantiate the material object
        material = self.ui.find_block("material")
        if material is None:
            pu.report_and_raise_error(
                "Material block not found for {0}".format(self.name))
        self.material = Material(material)
        self.matdat = self.material.material_data()

        # get the boundary and legs blocks
        boundary, legs = self.ui.find_nested_blocks("boundary", ("legs", ))
        if boundary is None:
            pu.report_and_raise_error(
                "Boundary block not found for {0}".format(self.name))
        if not self.material.eos_model and legs is None:
            pu.report_and_raise_error(
                "Legs block not found for {0}".format(self.name))

        # solid and eos materials have different boundary classes
        if not self.material.eos_model:
            self.boundary = pb.Boundary(boundary, legs)
        else:
            self.boundary = pb.EOSBoundary(boundary, legs)

        self.t0 = self.boundary.initial_time()
        self.tf = self.boundary.termination_time()
        ro.set_number_of_steps(N=self.boundary.nsteps(), I=0)

        # set up the simulation data container and register obligatory data
        self.simdat = DataContainer(self.name)
        self.simdat.register("istep", "Scalar", iv=0, attr=True,
                             units="NO_UNITS")
        self.simdat.register("nsteps", "Scalar", iv=self.boundary.nsteps(),
                             constant=True, units="NO_UNITS")
        self.simdat.register("time", "Scalar", iv=0., plot_key="time",
                             units="TIME_UNITS")
        self.simdat.register("time step", "Scalar", iv=0.,
                             plot_key="timestep", units="TIME_UNITS")
        self.simdat.register("leg number", "Scalar", iv=0, units="NO_UNITS")
        self.simdat.register("payette density", "Scalar",
                             iv=self.material.initial_density(),
                             plot_key="PRHO", units="DENSITY_UNITS")
        self.simdat.register("volume fraction", "Scalar", iv=1.,
                             plot_key="VFRAC", units="NO_UNITS")

        if ro.CHECK_SETUP:
            exit("EXITING to check setup")

        if ro.WRITERESTART:
            self.restart_file = os.path.splitext(self.outfile)[0] + ".prf"

        # write out properties
        if not ro.NOWRITEPROPS:
            self._write_mtl_params()
        self.write_input = ro.WRITE_INPUT or self.simdir != os.getcwd()
        self.simdat.register("nprints", "Scalar",
                             self.boundary.nprints(), constant=True,
                             units="NO_UNITS")
        if not self.material.eos_model:
            # register data not needed by the eos models
            self.simdat.register("emit", "Scalar", self.boundary.emit(),
                                 constant=True, units="NO_UNITS")
            self.simdat.register("kappa", "Scalar", self.boundary.kappa(),
                                 constant=True, units="NO_UNITS")
            self.simdat.register("screenout", "Scalar", self.boundary.screenout(),
                                 constant=True, units="NO_UNITS")

        # --- optional information ------------------------------------------ #
        # list of plot keys for all plotable data
        output = self.ui.find_block("output")
        output, oformat = pip.parse_output(output)
        self.plot_keys = [x for x in self.simdat.plot_keys()]
        self.plot_keys.extend(self.matdat.plot_keys())
        self.plot_keys = [x.upper() for x in self.plot_keys]
        if "ALL" in output:
            self.out_vars = [x for x in self.plot_keys]
        else:
            self.out_vars = [x for x in output if x.upper() in self.plot_keys]

        mathplot = self.ui.find_block("mathplot", [])
        if mathplot:
            mathplot = pip.parse_mathplot(mathplot)
        self.mathplot_vars = mathplot

        # get extraction
        extraction, eopts = self.ui.find_block("extraction", []), {}
        if extraction:
            extraction, eopts = pip.parse_extraction(extraction)
            extraction = [x for x in extraction
                          if x[1:] in self.plot_keys or x[1] == "%"]
        self.extraction_vars = extraction
        self.eopts = eopts

        # set up the data containers and initialize material models
        self.simdat.setup_data_container()
        self.matdat.setup_data_container()
        self.material.constitutive_model.initialize_state(self.matdat)

        self._setup_files()

        pass

    # private methods
    def _write_extraction(self):
        """ write out the requested extraction """
        exargs = [self.outfile] + self.extraction_vars
        step = int(self.eopts.get("step", 1))
        pe.extract(exargs, silent=True, write_xout=True, step=step)
        return

    def _write_mathplot(self):
        """ Write the $SIMNAME.math1 file for mathematica post processing """

        math1 = os.path.join(self.simdir, self.name + ".math1")
        math2 = os.path.join(self.simdir, self.name + ".math2")

        # math1 is a file containing user inputs, and locations of simulation
        # output for mathematica to use
        cmod = self.material.constitutive_model
        user_params = cmod.get_parameter_names_and_values(version="unmodified")
        adjusted_params = cmod.get_parameter_names_and_values(version="modified")
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
                "lastep=Length[{0}]\n".format(self.simdat.get("time", "plot key")))

        # math2 is a file containing mathematica directives to setup default
        # plots that the user requested
        lowhead = [x.lower() for x in self.out_vars]
        with open( math2, "w" ) as fobj:
            fobj.write('showcy[{0},{{"cycle", "time"}}]\n'
                    .format(self.simdat.get("time", "plot key")))

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
        props_f = os.path.join(self.simdir, self.name + ".props")
        with open(props_f, "w" ) as fobj:
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
            if not os.path.isfile(file_name):
                pu.report_and_raise_error(
                    "Original output file {0} not found during "
                    "setup of restart".format(file_name))
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
            plot_name = dat.get(plot_key, "description")
            val = dat.get(plot_key)
            _write_plotable(plot_idx + 1, plot_key, plot_name, val)
            continue

        return

    def write_state(self, iu=None, ou=None):
        """ write the simulation and material data to the output file. If
        iu and ou are given and valid, convert
        from the input system to the output system first.
        """
        UNITCONVERT = False
        if all(x is not None for x in (iu, ou,)):
            UNITCONVERT = True
        for out_var in self.out_vars:
            if out_var in self.simdat.plot_keys():
                dat = self.simdat
            else:
                dat = self.matdat
            val = dat.get(out_var)

            if UNITCONVERT:
                units = dat.get(out_var, "units")
                umgr = um.UnitManager(val, iu, units)
                umgr.convert(ou)
                val = umgr.get()
            self.outfile_obj.write(pu.textformat(val))
            continue
        self.outfile_obj.write('\n')
        return

    def setup_restart(self):
        """set up the restart files"""
        nsteps, istep = self.simdat.NSTEPS, self.simdat.ISTEP
        if self.is_restart:
            pu.report_and_raise_error("Restart file already run")
        self.is_restart = istep
        ro.set_number_of_steps(N=nsteps, I=istep)
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

        # solid and eos materials have different drivers
        if not self.material.eos_model:
            driver = pd.solid_driver

        else:
            driver = pd.eos_driver

        # jrh: This dictionary to get extra info from eos_driver
        extra_files = {}
        try:
            retcode = driver(self, restart=self.is_restart, extra_files=extra_files)

        except PayetteError as error:

            if ro.DEBUG:
                self.finish()
                raise

            if ro.ERROR.lower() == "ignore":
                retcode = 0
                sys.stderr.write("WARNING: Payette simulation "
                                 "{0} failed with the following message:\n{1}\n"
                                 .format(self.name,
                                         re.sub("ERROR:\s*", "", error.message)))

            else:
                retcode = 66
                l = 79 # should be odd number
                stars = "*" * (l + 2) + '\n'
                stars_spaces = "*" + " " * (l) + '*\n'
                psf = ("Payette simulation {0} failed"
                       .format(self.name).center(l - 2))
                ll = (l - len(psf)) / 2
                psa = "*" + " " * ll + psf + " " * ll + "*\n"
                head = stars + stars_spaces + psa + stars_spaces + stars
                sys.stderr.write(
                    "{0} Payette simulation {1} failed with the following "
                    "message:\n{2}\n".format(head, self.name, error.message))

        if retcode == 0:
            pu.log_message("Payette simulation {0} ran to completion"
                           .format(self.name))

        if not ro.DISP:
            return retcode

        else:
            return {"retcode": retcode,
                    "output file": self.outfile,
                    "extra files": extra_files,
                    "simulation name": self.name,
                    "simulation directory": self.simdir}

    def finish(self, wipe=False, wipeall=False):
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

        if self.write_input:
            fpath = os.path.join(self.simdir, self.name + ".inp")
            self.ui.write_input_file(fpath)

        if wipe or wipeall:
            # remove cruft
            for ext in (".log", ".props", ".math1", ".math2", ".prf", ".xout"):
                try: os.remove(self.name + ext)
                except OSError: pass
                continue
        if wipeall:
            try: os.remove(self.name + ".out")
            except OSError: pass
        return

    def simulation_data(self):
        """return the simulation simdat object"""
        return self.simdat
