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
"""Payette_optimize.py module. Contains classes and functions for optimizing
parameters using Payette.

"""

import os
import sys
import shutil
import numpy as np
import scipy
import scipy.optimize
import math
from copy import deepcopy

import Source.Payette_utils as pu
import Source.Payette_container as pc
import Source.Payette_extract as pe
import Source.Payette_input_parser as pip
import Source.runopts as ro
import Source.Payette_sim_index as psi
import Toolset.KayentaParamConv as kpc
from Source.Payette_extract import ExtractError as ExtractError

# Module level variables
IOPT = 0

class ParameterizeError(Exception):
    def __init__(self, message):
        super(ParameterizeError, self).__init__(message)

class ParameterizeLogger(object):
    loggers = {"root": None, }
    def __init__(self, name, fpath, mode="w"):
        self.name = name
        self.fpath = fpath
        self.fobj = open(fpath, mode)
        self.loggers[self.name] = {"object": self, "file object": self.fobj}
        pass

    def log(self, message, level=1, end="\n", cout=False, root_write=False):
        """Log a message

        Parameters
        ----------
        message : str
            the message to write
        level : int [optional, 1]
            log level
        end : str [optional, \n]
            ending character
        cout : bool [optional, False]
            write message to console
        """
        pre = {0: "", 1: "INFO: ", 2: "WARNING: ", 3: "ERROR: "}[level]
        message = message.split("\n")
        message = "".join(["{0}{1}{2}".format(pre, x, end)
                           for x in message if x])

        self.fobj.write(message)

        if cout or ro.VERBOSITY > 1:
            sys.stdout.write(message)

        root = self.loggers["root"].get("object")
        if self == root:
            return

        if root is not None and root_write:
            self.loggers["root"]["file object"].write(message)

        return

    def close(self):
        self.fobj.flush()
        self.fobj.close()
        del self.loggers[self.name]
        return

    def closeall(self):
        for logger in self.loggers:
            try:
                logger["file object"].flush()
                logger["file object"].close()
            except ValueError:
                pass

def get_logger(name, fpath=None):
    if fpath is not None:
        return ParameterizeLogger(name, fpath)
    logger = ParameterizeLogger.loggers.get(name)
    if logger is None:
        raise ParameterizeLogger("Logger {0} not found".format(name))
    return logger["object"]


def parameterizer(input_lines, material_index):
    r"""docstring -> needs to be completed """

    # get the optimization block
    job_inp = pip.InputParser(input_lines)

    # check for required directives
    req_directives = ("constitutive model", "material", )
    job_directives = job_inp.input_options()
    for directive in req_directives:
        if directive not in job_directives:
            raise ParameterizeError(
                "Required directive {0} not found".format(directive))

    # check for compatible constitutive model
    constitutive_models = {"kayenta": Kayenta}
    constitutive_model = job_inp.user_options["constitutive model"]
    if constitutive_model not in constitutive_models:
        raise ParameterizeError("Constitutive model {0} not recognized"
                                .format(constitutive_model))

    return constitutive_models[constitutive_model](job_inp, material_index)


class Parameterize(object):
    r"""docstring -> needs to be completed """

    def __init__(self, job, material_index, *args, **kwargs):
        r""" Initialization """

        # the name is the name of the material
        self.job_directives = job.input_options()
        self.name = self.job_directives["material"]
        self.fext = ".opt"
        self.verbosity = ro.VERBOSITY
        self.job = job
        self.material_index = material_index

        pass

    def run_job(self):
        sys.exit("run_job method not provided")

    def finish(self):
        sys.exit("finish method not provided")

    def parse_block(self, block):
        """Parse the Kayenta parameterization blocks

        """

        # defaults
        data_f = None
        optimize = {}
        fix = {}
        options = {}

        # parse the shearfit block
        for item in block:

            for pat in ",=()":
                item = item.replace(pat, " ")
            item = item.split()

            if not item:
                continue

            token = item[0].lower()

            if " ".join(item[0:2]).lower() == "data file":
                data_f = os.path.realpath(item[2])
                if not data_f:
                    pu.report_error("data file {0} not found".format(data_f))

            elif "optimize" in token:
                # set up this parameter to optimize
                key = item[1]
                vals = item[2:]

                bounds = (None, None)
                init_val = None

                # upper bound
                if "bounds" in vals:
                    try:
                        idx = vals.index("bounds") + 1
                        bounds = [float(x) for x in vals[idx:idx+2]]
                    except ValueError:
                        pu.report_error("bounds requires 2 arguments")

                    if bounds[0] > bounds[1]:
                        pu.report_error(
                            "lower bound {0} > upper bound {1} for {2}"
                            .format(bounds[0], bounds[1], key))

                if "initial" in vals:
                    idx = vals.index("initial")
                    if vals[idx + 1] == "value":
                        idx = idx + 1
                        try:
                            init_val = float(vals[idx+1])
                        except ValueError:
                            pu.report_error("excpected float for initial value")

                optimize[key] = {"bounds": bounds, "initial value": init_val,}

            elif "fix" in token:
                # set up this parameter to fix
                key = item[1]
                vals = item[2:]
                init_val = None
                if "initial" in vals:
                    idx = vals.index("initial")
                    if vals[idx + 1] == "value":
                        idx = idx + 1
                        try:
                            init_val = float(vals[idx+1])
                        except ValueError:
                            pu.report_error("excpected float for initial value")

                fix[key] = {"initial value": init_val}

            else:
                options[token.upper()] = item[1]

            continue

        if data_f is None:
            pu.report_error("No data file given for shearfit problem")

        for key, val in fix.items():
            if key.lower() in [x.lower() for x in optimize]:
                pu.report_error("cannot fix and optimize {0}".format(key))

        if pu.error_count():
            pu.report_and_raise_error("stopping due to previous errors")

        return data_f, optimize, fix, options


# parameters to be optimized by shearfit and hydrofit
def A_map(token=None):
    mapping = {"A1": 0, "A2": 1, "A3": 2, "A4": 3,}
    if token is None:
        return sorted(mapping.keys())
    if isinstance(token, (int, float)):
        token = int(token)
        for key, val in mapping.items():
            if val == token:
                return key
            continue
        return None

    return mapping.get(token)

def B_map(token):
    mapping = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4,}
    if token is None:
        return sorted(mapping.keys())
    if isinstance(token, (int, float)):
        token = int(token)
        for key, val in mapping.items():
            if val == token:
                return key
            continue
        return None

    return mapping[token]

class Kayenta(Parameterize):

    def __init__(self, job, material_index, *args, **kwargs):
        """Initialize the Kayenta parameterization class

        """

        super(Kayenta, self).__init__(job, material_index, *args, **kwargs)

        self.data = {"shearfit": None, "hydrofit": None}

        # get the shearfit block
        shearfit_block = job.get_block("shearfit")
        hydrofit_block = job.get_block("hydrofit")

        self.shearfit = shearfit_block is not None
        self.hydrofit = hydrofit_block is not None

        if self.shearfit:
            block_data = self.parse_block(shearfit_block)
            self.init_shearfit(*block_data)

        if self.hydrofit:
            block_data = self.parse_block(hydrofit_block)
            self.init_hydrofit(*block_data)


        pass

    def init_shearfit(self, *args):
        """ Initialize data needed by shearfit

        To run shearfit, we need values for A1 - A4, in that order. The user
        may only want to optimize some, not all, of the A parameters. Here we
        initialize all A parameters to zero, and then populate the initial
        array with input from the user.

        """

        data_f, optimize, fix, options = args

        # pass arguments to class data
        self.SF_maxiter = int(options.get("MAXITER", 20))
        self.SF_tolerance = float(options.get("TOLERANCE", 1.e-6))
        self.SF_method = options.get("METHOD", "SIMPLEX")

        # check input vars
        for key, val in optimize.items():
            if A_map(key.upper()) is None:
                raise ParameterizeError(
                    "unrecognized optimization variable {0}".format(key))
        for key, val in fix.items():
            if A_map(key.upper()) is None:
                raise ParameterizeError(
                    "unrecognized fixed variable {0}".format(key))

        # -------------------------------------------------------------------- #
        # check the data file. the user can specify either columns of I1 and
        # ROOTJ2 or SIG11 -> SIG13 from which I1 and ROOTJ2 will be extracted.
        min_vars = ["@I1", "@ROOTJ2"]
        exargs = [data_f, "--silent", "--disp=1"] + min_vars
        try:
            i1_rtj2 = pe.extract(exargs)[1]

        except ExtractError:
            head = list(set([x.lower() for x in sorted(pu.get_header(data_f))]))
            sigcnt = 0
            for item in head:
                if item.lower()[0:3] == "sig":
                    sigcnt += 1
                continue
            if sigcnt < 3:
                pu.report_and_raise_error(
                    "{0} does not contain sufficient information to "
                    "compute I1 and ROOTJ2".format(data_f))

            # user gave file with sig11, sig22, ..., it needs to be
            # converted to i1 and rootj2
            i1_rtj2 = get_rtj2_vs_i1(data_f)

        # save the i1 and rtj2 data
        self.SF_data = np.array(i1_rtj2)

        # -------------------------------------------------------------------- #
        ai = [None] * 4
        bounds = [[None, None] for i in range(4)]
        v = [None] * 4
        fixed = [None] * 4
        has_bounds = False

        # User specified parameters to optimize, initialize data
        for key, val in optimize.items():
            idx = A_map(key.upper())
            ai[idx] = val["initial value"]
            if ai[idx] is not None and ai[idx] < 0.:
                raise ParameterizeError("B{0} must be > 0".format(idx+1))
            bounds[idx] = val["bounds"]
            if not has_bounds:
                has_bounds = any(x is not None for x in bounds[idx])
            v[idx] = idx
            continue

        # any ai that is None is converted to 0
        ai = [x if x is not None else 0. for x in ai]

        if not optimize or not ai[0]:
            # user did not specify any parameters, get the first guess in a two
            # step process:
            # 1) get A1 and A4 through a linear fit, if it is good, keep the
            #    result
            # 2) if needed, do a full curve fit, using A1 from above

            ione = self.SF_data[:, 0]
            rootj2 = self.SF_data[:, 1]

            # linear fit
            fit = np.polyfit(ione, rootj2, 1, full=True)
            if fit[1] < 1.e-16:
                # data was very linear -> drucker prager -> fix all values
                ai[3], ai[0] = np.abs(fit[0])
                ai[1], ai[2] = 0., 0.

            else:
                # least squares fit
                p0 = np.zeros(4)
                p0[0] = fit[0][1]
                step = int(round(len(ione) / 4.))
                i1 = ione[::step]
                rtj2 = rootj2[::step]
                curve_fit = scipy.optimize.curve_fit(rtxc, i1, rtj2, p0=p0)
                ai = np.abs(curve_fit[0])
                v = [0, 1, 2, 3]

        # restore or set fixed values
        for key, val in fix.items():
            idx = A_map(key.upper())
            ai[idx] = val["initial value"]
            fixed[idx] = key.upper()
            if ai[idx] is None:
                raise ParameterizeError(
                    "No initial value given for {0}".format(key.upper()))
            v[idx] = None
            continue

        # ai, bounds, and v now contain the initial values for the a
        # parameters, bounds, and an integer index array of which parameters
        # to optimize. filter the data
        self.SF_v = [x for x in v if x is not None]
        self.SF_ai = np.array(ai, dtype=float)
        self.SF_fixed = [x for x in fixed if x is not None]

        # optimizers like to work with numbers close to one => scale the
        # optimized parameters
        fac = [None] * 4
        for idx in self.SF_v:
            fac[idx] = eval(
                "1.e" + "{0:12.6E}".format(self.SF_ai[idx]).split("E")[1])
            continue
        self.SF_fac = [x for x in fac if x is not None]

        # Convert the bounds to inequality constraints
        lcons, ucons = [], []
        for iv, idx in enumerate(self.SF_v):
            lbnd, ubnd = bounds[idx]
            if lbnd is None:
                lbnd = 0.

            if ubnd is None:
                if self.SF_ai[idx]:
                    ubnd = self.SF_ai[idx] + 1000. * self.SF_ai[idx]
                elif idx == 3 or idx == 1:
                    ubnd = 2.
                else:
                    ubnd = 1.e99

            if lbnd >= ubnd:
                pu.report_error("lbnd({0:12.6E}) >= ubnd({1:12.6E})"
                                .format(lbnd, ubnd))

            lbnd, ubnd = lbnd / fac[idx], ubnd / fac[idx]
            bounds[idx] = [lbnd, ubnd]
            lcons.append(lambda z, idx=iv, bnd=lbnd: z[idx] - bnd)
            ucons.append(lambda z, idx=iv, bnd=ubnd: bnd - z[idx])
            continue

        # The inequality constraint that A1 - A3 > 0
        if A_map("A1") in self.SF_v and A_map("A3") in self.SF_v:
            a1, a3 = self.SF_v.index(A_map("A1")), self.SF_v.index(A_map("A3"))
            cons = [lambda z, a1=a1, a3=a3: z[a1] - z[a3]]
        elif A_map("A1") in self.SF_v:
            a1, a3 = self.SF_v.index(A_map("A1")), self.SF_ai[A_map("A3")]
            cons = [lambda z, a1=a1, a3=a3: z[1] - a3]
        elif A_map("A3") in self.SF_v:
            a1, a3 = self.SF_ai[A_map("A1")], self.SF_v.index(A_map("A3"))
            cons = [lambda z, a1=a1, a3=a3: a1 - z[a3]]
        else:
            cons = []
        self.SF_cons = lcons + ucons + cons
        self.SF_bounds = [x for x in bounds]

        if pu.error_count():
            pu.report_and_raise_error("ERROR: Resolve previous errors")

        return

    def init_hydrofit(self, *args):
        """ Initialize data needed by hydrofit

        To run shearfit, we need values for B0 - B4, in that order. The user
        may only want to optimize some, not all, of the B parameters. Here we
        initialize all B parameters to zero, and then populate the initial
        array with input from the user.

        """

        data_f, optimize, fix, options = args

        # pass arguments to class data
        self.HF_maxiter = int(options.get("MAXITER", 20))
        self.HF_tolerance = float(options.get("TOLERANCE", 1.e-6))
        self.HF_method = options.get("METHOD", "COBYLA")

        # check input vars
        for key, val in optimize.items():
            if B_map(key.upper()) is None:
                raise ParameterizeError(
                    "unrecognized optimization variable {0}".format(key))
        for key, val in fix.items():
            if B_map(key.upper()) is None:
                raise ParameterizeError(
                    "unrecognized fixed variable {0}".format(key))

        # --------------------------------------------------------------------
        # # check the data file. the user must specify a column of volume
        # strain "EVOL" and pressure "PRESSURE"
        min_vars = ["@EVOL", "@PRESSURE"]
        exargs = [data_f, "--silent", "--disp=1"] + min_vars
        try:
            evol_pres = pe.extract(exargs)[1]

        except ExtractError:
            raise ParameterizeError(
                "Could not find EVOL and PRESSURE columns from {0}"
                .format(os.path.basename(data_f)))

        # initial guess at plastic volume change
        ev_p = evol_pres[-1][0]

        # determine the unload curve
        ev_c, idx_c = 1.e9, None
        for idx, (evol, pres) in enumerate(evol_pres):
            if evol < ev_c:
                ev_c = evol
                idx_c = idx
            continue
        unload = [[x[0] - ev_p, x[1]] for x in reversed(evol_pres[idx_c:])]

        crush_pres = float(options.get("PC", 0.))
        if crush_pres:
            unload = [x for x in unload if x[1] < crush_pres]

        # determine the tangent bulk modulus from unload curve
        # K = -dp / dv
        bmod = [-(unload[idx + 1][1] - unload[idx][1]) /
                 (unload[idx + 1][0] - unload[idx][0])
                 for idx in range(len(unload) - 1)]
        if any(x < 0. for x in bmod):
            raise ParameterizeError(
                "Encountered negative bulk modulus in hydrofit data")

        # determine portions due to crushing and loading
        for idx, val in enumerate(bmod):
            devol = evol_pres[idx + 1][0] - evol_pres[idx][0]
            dpres = evol_pres[idx + 1][1] - evol_pres[idx][1]
            if abs((-val * devol - dpres) / dpres) > .1:
                crush = [[x[0], x[1]] for x in evol_pres[idx-1:]]
                load = [[x[0], x[1]] for x in evol_pres[:idx]]
                break

        # save the data to class attribute
        self.HF_data = np.array(evol_pres)
        self.HF_bmod = np.array(bmod)
        self.HF_unload = np.array(unload)
        self.HF_crush = np.array(crush)
        self.HF_load = np.array(load)



        if ro.VERBOSITY:
            # write out the elastic unloading curve
            with open(self.name + ".unload", "w") as fobj:
                fobj.write("{0:13s} {1:13s} {2:13s}\n"
                           .format("EVOL", "PRES", "BMOD"))
                for idx in range(len(self.HF_bmod)):
                    fobj.write("{0:12.6E} {1:12.6E} {2:12.6E}\n"
                               .format(self.HF_unload[:, 0][idx] + ev_p,
                                       self.HF_unload[:, 1][idx],
                                       self.HF_bmod[idx]))

            # write out the crush curve
            with open(self.name + ".crush", "w") as fobj:
                fobj.write("{0:13s} {1:13s}\n".format("EVOL", "PRES"))
                for idx in range(len(self.HF_crush)):
                    fobj.write("{0:12.6E} {1:12.6E}\n"
                               .format(self.HF_crush[:, 0][idx],
                                       self.HF_crush[:, 1][idx]))

            # write out the loading curve
            with open(self.name + ".load", "w") as fobj:
                fobj.write("{0:13s} {1:13s}\n".format("EVOL", "PRES"))
                for idx in range(len(self.HF_load)):
                    fobj.write("{0:12.6E} {1:12.6E}\n"
                               .format(self.HF_load[:, 0][idx],
                                       self.HF_load[:, 1][idx]))
        # -------------------------------------------------------------------- #
        bi = [None] * 5
        bounds = [[None, None] for i in range(5)]
        has_bounds = False
        v = [None] * 5
        fixed = [None] * 5

        # User specified parameters to optimize, initialize data
        for key, val in optimize.items():
            idx = B_map(key.upper())
            bi[idx] = val["initial value"]
            if bi[idx] is not None and bi[idx] < 0.:
                raise ParameterizeError("B{0} must be > 0".format(idx))
            bounds[idx] = val["bounds"]
            if not has_bounds:
                has_bounds = any(x is not None for x in bounds[idx])
            v[idx] = idx
            continue

        # any bi that is None is converted to 0
        bi = [x if x is not None else 0. for x in bi]

        if not optimize or not bi[0]:
            # user did not specify any parameters, get the first guess for B0
            # through a linear fit

            # least squares fit
            ione = -3. * self.HF_unload[:, 1][1:]
            p0 = np.zeros(5)
            p0[0] = self.HF_bmod[0]
            p0[1] = self.HF_bmod[-1] - p0[0]
            p0[2] = -np.log((self.HF_bmod[2] - p0[0]) / p0[1]) * np.abs(ione[2])
            curve_fit = scipy.optimize.curve_fit(
                kfunc, ione, self.HF_bmod, p0=p0)
            bi = np.abs(curve_fit[0])
            v = [0, 1, 2, 3, 4] if has_bounds else [None] * 5

        # restore or set fixed values
        for key, val in fix.items():
            idx = B_map(key.upper())
            bi[idx] = val["initial value"]
            fixed[idx] = key.upper()
            v[idx] = None
            continue

        # bi, bounds, and v now contain the initial values for the b
        # parameters, bounds, and an integer index array of which parameters
        # to optimize. filter the data
        self.HF_v = np.array([x for x in v if x is not None], dtype=int)
        self.HF_bi = np.array(bi, dtype=float)

        # optimizers like to work with numbers close to one => scale the
        # optimized parameters
        self.HF_fixed = [x for x in fixed if x is not None]

        # Convert the bounds to inequality constraints
        lcons, ucons = [], []
        for idx in self.HF_v:
            lbnd, ubnd = bounds[idx]
            if lbnd is None:
                lbnd = 0.
            if ubnd is None:
                ubnd = self.HF_bi[idx] + 10. * self.HF_bi[idx]

            if lbnd >= ubnd:
                pu.report_error("lbnd({0:12.6E}) > ubnd({1:12.6E})"
                                .format(lbnd, ubnd))

            bounds[idx] = [lbnd, ubnd]
            lcons.append(lambda z, idx=idx, bnd=lbnd: z[idx] - bnd)
            ucons.append(lambda z, idx=idx, bnd=ubnd: bnd - z[idx])

            continue
        self.HF_cons = lcons + ucons
        self.HF_bounds = [bounds[idx] for idx in self.HF_v]

        if pu.error_count():
            pu.report_and_raise_error("ERROR: Resolve previous errors")

        return

    def run_job(self, *args, **kwargs):
        r"""Run the optimization job

        Set up directory to run the optimization job and call the minimizer

        Parameters
        ----------
        None

        Returns
        -------
        opt_params : array_like
            The optimized parameters

        """

        logger = get_logger("root", fpath=self.name+".out")

        logger.log("Running parameterization job {0}".format(self.name),
                   cout=True)

        if self.shearfit:
            IOPT = 0
            shearlog = get_logger("shearfit", fpath=self.name+".shearfit")
            self.run_shearfit_job()
            shearlog.close()


        if self.hydrofit:
            IOPT = 0
            hydrolog = get_logger("hydrofit", fpath=self.name+".hydrofit")
            self.run_hydrofit_job()
            hydrolog.close()

        return 0

    def run_shearfit_job(self):

        # set up args and call optimzation routine
        args = (self.SF_ai, self.SF_data, self.SF_fac,
                self.SF_v, self.verbosity)

        msg_init = "".join(["{0} = {1:12.6E}\n".format(
                    A_map()[idx], self.SF_ai[idx]) for idx in self.SF_v])
        if self.SF_fixed:
            msg_fixd = ("\n\nFixed values:\n" +
                        "".join(["{0} = {1:12.6E}\n".format(
                            key, self.SF_ai[A_map(key)])
                                 for key in self.SF_fixed]))
        else:
            msg_fixd = ""
        msg_bnds = "".join(["{0}: ({1:12.6E}, {2:12.6E})\n".format(
                    A_map()[idx],
                    self.SF_bounds[idx][0] * self.SF_fac[iv],
                    self.SF_bounds[idx][1] * self.SF_fac[iv])
                            for iv, idx in enumerate(self.SF_v)])

        message = """\
Starting the shear fit optimization routine.
Optimization method: {0}
Initial values:
{1}{2}
Bounds:
{3}""".format(self.SF_method, msg_init, msg_fixd, msg_bnds)

        # get the log file
        logger = get_logger("shearfit")
        logger.log(message, cout=True)

        # call the optimizer
        xinit = self.SF_ai[self.SF_v] / self.SF_fac
        xopt = optimize(shearfit, xinit, self.SF_cons, method=self.SF_method,
                        consargs=(), args=args, disp=0, tol=self.SF_tolerance)
        xopt = np.array([max(0., x) for x in xopt])

        # optimum parameters found, write out final info
        self.A_opt = self.SF_ai
        self.A_opt[self.SF_v] = xopt * self.SF_fac
        self.new_params = kpc.old_2_new(*self.A_opt)
        stren, peaki1, fslope, yslope = self.new_params
        msg = ("A1 = {0:12.6E}, ".format(self.A_opt[0]) +
               "A2 = {0:12.6E}, ".format(self.A_opt[1]) +
               "A3 = {0:12.6E}, ".format(self.A_opt[2]) +
               "A4 = {0:12.6E}\n".format(self.A_opt[3]) +
               "STREN = {0:12.6E}, ".format(stren) +
               "PEAKI1 = {0:12.6E}, ".format(peaki1) +
               "FSLOPE = {0:12.6E}, ".format(fslope) +
               "YSLOPE = {0:12.6E}".format(yslope))

        message = ("Optimized shearfit parameters found on iteration {0}\n"
                   "Optimized shearfit parameters:\n{1}".format(IOPT, msg))
        logger.log(message, cout=True, root_write=True)

        return 0

    def run_hydrofit_job(self):

        # set up args and call optimzation routine
        ione = -3. * self.HF_unload[:, 1][1:]
        args = (self.HF_bi, ione, self.HF_bmod,
                self.HF_v, self.verbosity)

        # open the log file
        xinit = np.array(self.HF_bi)
        xinit[self.HF_v] = xinit[self.HF_v]
        msg_init = "".join(["{0} = {1:12.6E}\n".format(
                    B_map()[idx], xinit[idx]) for idx in self.HF_v])
        msg_fixd = "".join(["{0} = {1:12.6E}\n".format(
                    key, xinit[B_map(key)]) for key in self.HF_fixed])
        msg_bnds = "".join(["{0}: ({1:12.6E}, {2:12.6E})\n".format(
                    B_map()[idx], self.HF_bounds[idx][0],
                    self.HF_bounds[idx][1]) for idx in self.HF_v])

        message = """\
Starting the hydro fit optimization routine.
Optimization method: {0}
Initial values:
{1}
Fixed values
{2}
Bounds:
{3}""".format(self.HF_method, msg_init, msg_fixd, msg_bnds)

        logger = get_logger("hydrofit")
        logger.log(message, cout=True)

        # call the optimizer
        xopt = optimize(hydrofit, self.HF_bi[self.HF_v], self.HF_cons,
                        consargs=(), args=args, disp=0,
                        method=self.HF_method, tol=self.HF_tolerance)

        # optimum parameters found, write out final info
        self.B_opt = self.HF_bi
        self.B_opt[self.HF_v] = xopt
        msg = ("B0 = {0:12.6E}, ".format(self.B_opt[0]) +
               "B1 = {0:12.6E}, ".format(self.B_opt[1]) +
               "B2 = {0:12.6E}, ".format(self.B_opt[2]) +
               "B3 = {0:12.6E}, ".format(self.B_opt[3]) +
               "B4 = {0:12.6E}\n".format(self.B_opt[4]))
        message = ("Optimized hydrofit parameters found on iteration {0}\n"
                   "Optimized hydrofit parameters:\n{1}".format(IOPT, msg))
        logger.log(message, cout=True, root_write=True)

        return 0

    def finish(self):
        r""" finish up the optimization job """
        return


def get_rtj2_vs_i1(fpath):
    """Convert sigij in file fpath to ione and rootj2

    Parameters
    ----------
    fpath : str
        path to file

    """

    header = pu.get_header(fpath)
    data = pu.read_data(fpath)
    sig11, sig22, sig33, sig12, sig23, sig13 = [None] * 6
    for idx, item in enumerate(header):
        if item.lower() == "sig11":
            sig11 = data[:, idx]

        if item.lower() == "sig22":
            sig22 = data[:, idx]

        if item.lower() == "sig33":
            sig33 = data[:, idx]

        if item.lower() == "sig12":
            sig12 = data[:, idx]

        if item.lower() == "sig23":
            sig23 = data[:, idx]

        if item.lower() == "sig13":
            sig13 = data[:, idx]

        continue

    if sig11 is None or sig22 is None or sig33 is None:
        pu.report_and_raise_error(
            "insufficient information in {0} to compute I1 and ROOTJ2"
            .format(data_f))

    if sig12 is None:
        sig12 = np.zeros(len(sig11))

    if sig23 is None:
        sig23 = np.zeros(len(sig11))

    if sig13 is None:
        sig13 = np.zeros(len(sig11))

    i1 = sig11 + sig22 + sig33
    rtj2 = np.sqrt(1. / 6. * ((sig11 - sig22) ** 2 +
                              (sig22 - sig33) ** 2 +
                              (sig33 - sig11) ** 2) +
                   sig12 ** 2 + sig23 ** 2 + sig13 ** 2)

    fnam, fext = os.path.splitext(fpath)

    i1_rtj2 = []
    for idx in range(len(i1)):
        i1_rtj2.append([i1[idx], rtj2[idx]])

    return i1_rtj2


def shearfit(xcall, xinit, data, fac, v, verbosity, ncalls=[0]):
    """The Kayenta limit function

    Evaluates the Kayenta limit function

              f(ione) = a1 + a3*exp(-a2*ione) + a4*ione

    Parameters
    ----------
    ione : array_like
        values of ione
    a1 - a4 : float
        Limit surface parameters

    Returns
    -------
    rootj2 : array_like
        values of sqrt(j2) corresponding to ione

    """

    global IOPT
    ncalls[0] += 1
    IOPT = ncalls[0]

    # write trial params to file
    msg = []
    for iv, idx in enumerate(v):
        opt_val = xcall[iv] * fac[iv]
        pstr = "{0} = {1:12.6E}".format(A_map()[idx], opt_val)
        msg.append(pstr)
        continue

    logger = get_logger("shearfit")
    logger.log("Iteration {0:03d}, trial parameters: {1}"
               .format(ncalls[0], ", ".join(msg)))

    # compute rootj2 and error
    a = xinit
    a[v] = xcall * fac
    a1, a2, a3, a4 = a

    ione = data[:, 0]
    rootj2 = data[:, 1]

    error = rootj2 - rtxc(ione, a1, a2, a3, a4)
    error = math.sqrt(np.mean(error ** 2))
    dnom = abs(np.amax(rootj2))
    error = error / dnom if dnom >= 2.e-16 else error

    return error

def hydrofit(xcall, xinit, ione, k, v, verbosity, ncalls=[0]):
    """The Kayenta bulk modulus function """

    global IOPT
    ncalls[0] += 1
    IOPT = ncalls[0]

    # write trial params to file
    msg = []
    for iv, idx in enumerate(v):
        opt_val = xcall[iv] * fac[iv]
        pstr = "{0} = {1:12.6E}".format(B_map()[idx], opt_val)
        msg.append(pstr)
        continue

    logger = get_logger("hydrofit")
    logger.log("Iteration {0:03d}, trial parameters: {1}"
               .format(ncalls[0], ", ".join(msg)))

    # compute rootj2 and error
    b = xinit
    b[v] = xcall
    b0, b1, b2, b3, b4 = b

    error = k - kfunc(ione, b0, b1, b2, b3, b4)
    error = math.sqrt(np.mean(error ** 2))
    dnom = abs(np.amax(k))
    error = error / dnom if dnom >= 2.e-16 else error

    return error

def exps(arg):
    """Exponential that guards against under and overflow"""
    eunderflow = -34.53877639491 * 1.
    eoverflow = 92.1034037 * 1.
    return np.array([np.exp(min(max(x, eunderflow), eoverflow)) for x in arg])

def kfunc(i1, b0, b1, b2, b3=0., b4=0.):
    """The kayenta bulk modulus function"""
    n = len(i1)
    return b0 * np.ones(n) + b1 * exps(-b2 / np.abs(i1))

def rtxc(i1, a1, a2, a3, a4):
    """The radius in triaxial compression"""
    return a1 - a3 * exps(a2 * i1) - a4 * i1

def optimize(func, x0, cons=[], method="SIMPLEX",
             consargs=(), args=(), disp=0, tol=1.e-6, maxiter=100):
    """wrapper around scipy minimization routines"""

    method = method.upper()
    methods = ("SIMPLEX", "COBYLA", "POWELL", "SLSQP")
    if method not in methods:
        raise ParameterizeError(
            "Unrecognized optimization method {0}, choose from {1}"
            .format(method, ", ".join(methods)))

    if method == "SIMPLEX":
        xopt = scipy.optimize.fmin(
            func, x0, args=args, disp=disp, xtol=tol, ftol=tol)

    elif method == "POWELL":
        xopt = scipy.optimize.fmin_powell(
            func, x0, args=args, disp=disp, xtol=tol, ftol=tol)

    elif method == "COBYLA":
        xopt = scipy.optimize.fmin_cobyla(
            func, x0, cons, consargs=consargs, args=args, disp=disp,
            rhoend=1.e-7, maxfun=5000)
    else:
        sys.exit("slsqp not yet supported")
        xopt = scipy.optimize.fmin_slsqp(
            func, x0, ieqcons=cons, args=args, disp=disp,
            acc=1.e-7, iter=maxiter)

    return xopt
