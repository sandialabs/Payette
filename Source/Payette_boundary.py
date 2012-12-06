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

"""Main Payette boundary class definitions"""

import math
import sys
import re
import numpy as np

import Source.__runopts__ as ro
import Source.Payette_input_parser as pip
import Source.Payette_utils as pu


DTYPES = {"strain rate": (1, 6), "strain": (2, 6), "stress rate": (3, 6),
          "stress": (4, 6), "deformation gradient": (5, 9),
          "electric field": (6, 3), "displacement": (8, 3), "vstrain": (2, 1),
          "pressure": (4, 1), "efield": (6, 3), }


class Boundary(object):

    def __init__(self, bblock, lblock):

        # ---- set defaults ------------------------------------------------- #
        self._kappa = 0.
        self.stepfac, self.efac, self.tfac, self.sfac, self.ffac = [1.] * 5
        self.effac, self.dfac, self.ratfac = [1.] * 3
        self._nsteps = 0

        # structure of _bcontrol:
        # _bcontrol[key] = [type, default, extra [min, choices,]]
        self._bcontrol = {
            "kappa": [float, 0., None], "estar": [float, 1., None],
            "tstar": [float, 1., None], "sstar": [float, 1., None],
            "fstar": [float, 1., None], "dstar": [float, 1., None],
            "efstar": [float, 1., None], "stepstar": [float, 1., 1.],
            "ampl": [float, 1., None], "ratfac": [float, 1., None],
            "emit": ["choice", "all", ("all", "sparse",)],
            "nprints": [int, 0, None], "screenout": [bool, False, None],
            "initial stress": [bool, False, None]}

        # intialize container for legs
        # -- _legs has form
        #    [[lnum, t, control, cij], ...]
        self._legs = []

        # parse
        self._parse_boundary(bblock)
        self._parse_legs(lblock)

        pass

    def kappa(self):
        return self.bcontrol("kappa")[1]

    def initial_time(self):
        return self._legs[0][1]

    def termination_time(self):
        return self._legs[-1][1]

    def legs(self, idx=0):
        return self._legs[idx:]

    def nprints(self):
        return self.bcontrol("nprints")[1]

    def emit(self):
        return {"all": 1, "sparse": 0}[self.bcontrol("emit")[1]]

    def screenout(self):
        return self.bcontrol("screenout")[1]

    def nsteps(self):
        return self._nsteps

    def _parse_boundary(self, bblock):
        """Parse the boundary block

        Parameters
        ----------
        bblock : str
            the boundary block

        Returns
        -------

        """
        boundary_options = {}

        for line in bblock.split("\n"):
            line = re.sub(pip.I_EQ, " ", line).split()
            if not line:
                continue

            if len(line) != 2:
                pu.report_and_raise_error(
                    "Boundary control items must be key = val pairs, got (0}"
                    .format(line))

            kwd, val = line
            if kwd.lower() not in self.bcontrol():
                try:
                    boundary_options[kwd] = eval(val)
                except (TypeError, ValueError):
                    boundary_options[kwd] = val
                continue

            bc = self.bcontrol(kwd)  # [type, default, extra]
            if bc[0] == "choice":
                choices = bc[2]
                if val not in choices:
                    pu.report_and_raise_error(
                        "{0} must be one of {1}, got {2}"
                        .format(kwd, ", ".join(choices), val))
            else:
                # get right type for val and check against min
                val = bc[0](val)
                if bc[2] is not None:
                    val = max(bc[2], val)

            self.bcontrol(kwd, val)
            continue

        # the following are from Brannon's MED driver
        # estar is the "unit" of strain
        # sstar is the "unit" of stress
        # fstar is the "unit" of deformation gradient
        # efstar is the "unit" of electric field
        # dstar is the "unit" of displacement
        # tstar is the "unit" of time
        # All strains are multiplied by efac=ampl*estar
        # All stresses are multiplied by sfac=ampl*sstar
        # All deformation gradients are multiplied by ffac=ampl*fstar
        # All electric fields are multiplied by effac=ampl*efstar
        # All displacements are multiplied by dfac=ampl*dstar
        # All times are multiplied by tfac=abs(ampl)*tstar/ratfac

        # From these formulas, note that AMPL may be used to increase or
        # decrease the peak strain without changing the strain rate. ratfac is
        # the multiplier on strain rate and stress rate.
        ampl = self.bcontrol("ampl")[1]
        tstar = self.bcontrol("tstar")[1]
        estar = self.bcontrol("estar")[1]
        sstar = self.bcontrol("sstar")[1]
        fstar = self.bcontrol("fstar")[1]
        efstar = self.bcontrol("efstar")[1]
        dstar = self.bcontrol("dstar")[1]

        # factors to be applied to deformation types
        self.ratfac = self.bcontrol("ratfac")[1]
        self._kappa = self.bcontrol("kappa")[1]
        self.stepfac = self.bcontrol("stepstar")[1]
        self.efac = ampl * estar
        self.tfac = abs(ampl) * tstar / self.ratfac
        self.sfac = ampl * sstar
        self.ffac = ampl * fstar
        self.effac = ampl * efstar
        self.dfac = ampl * dstar
        return

    def bcontrol(self, name=None, value=None):
        """The boundary control parameters allowed for by Payette

        Parameters
        ----------
        name : str, optional [None]
            boundary control parameter
        value : float, optional [None]
            value to set control parameter

        Returns
        -------
        C : int
            Integer ID for deformation type
        N : int
            Length of deformation type
        """

        if name is None:
            return self._bcontrol.keys()

        name = " ".join(name.split()).lower()
        if name not in self._bcontrol:
            pu.report_and_raise_error(
                "{0} not a valid bcontrol parameter".format(name))

        if value is None:
            return self._bcontrol[name]

        default = self._bcontrol[name]
        try:
            default[1] = default[0](value)
        except TypeError:
            default[1] = value
        self._bcontrol[name] = default
        return

    def _parse_legs(self, lblock):
        """Parse the legs block

        Parameters
        ----------
        lblock : str
            The legs block

        Returns
        -------

        """

        # determine if the user specified legs as a table
        stress_control = False
        using = re.search(r"(?i)\busing\b.*", lblock)
        table = bool(using)

        num, time = 0, 0.
        if table:
            s, e = using.start(), using.end()
            line = re.sub(r"(?i)\busing\b", " ", lblock[s:e])
            lblock = (lblock[:s] + lblock[e:]).strip()
            ttype, cidxs, gcontrol = self._parse_leg_table_header(line)

        # --- first leg parsed, now go through rest
        for iline, line in enumerate(lblock.split("\n")):
            line = re.sub(pip.I_SEP, " ", line)
            line = line.split()
            if not line:
                continue

            if table:
                control = gcontrol
                if re.search(r"(?i)\btime\b", " ".join(line)):
                    # skip header row
                    continue
                steps = int(1 * self.stepfac)
                if ttype == "dt":
                    time += float(line[cidxs[0]])
                else:
                    time = float(line[cidxs[0]])
                # adjust the actual time using the time factor
                ltime = self.tfac * time
                try:
                    cij = [float(eval(line[i])) for i in cidxs[1:]]
                except (IndexError, ValueError):
                    pu.report_and_raise_error(
                        "Syntax error in leg {0}".format(num))
            else:
                # user specified leg in form:
                # step number, time, steps, control, values

                # leg must have at least 5 values
                if len(line) < 5:
                    pu.report_and_raise_error(
                        "leg {0} input must be of form:".format(num) +
                        "\n\tnum, time, steps, type, c[ij]")

                num = int(line[0])
                ltime = float(self.tfac * float(line[1]))
                steps = int(self.stepfac * float(line[2]))
                if num != 0 and steps == 0:
                    pu.report_and_raise_error(
                        "Leg number {0} has no steps".format(num))

                # get the control type
                control = line[3].strip()

                # the remaining part of the line are the actual ij values of the
                # deformation type
                try:
                    cij = [float(eval(x)) for x in line[4:]]
                except ValueError:
                    pu.report_and_raise_error(
                        "Syntax error in leg {0}".format(num))

            # --- begin processing the cij -------------------------------------- #
            # control should be a group of letters describing what type of
            # control type the leg is. valid options are:
            #  1: strain rate control
            #  2: strain control
            #  3: stress rate control
            #  4: stress control
            #  5: deformation gradient control
            #  6: electric field
            #  8: displacement
            if any(x not in "1234568" for x in control):
                pu.report_and_raise_error(
                    "Leg control parameters can only be one of [1234568]"
                    "got {0} for leg number {1:d}".format(control, num))

            # stress control if any of the control types are 3 or 4
            if not stress_control:
                stress_control = re.search(r"[34]", control)
                stress_control = any([x in "34" for x in control])

            # we need to know what to do with each deformation value, so the
            # length of the deformation values must be same as the control
            # values
            if len(control) != len(cij):
                pu.report_and_raise_error(
                    "Length of leg control != number of control "
                    "items in leg {0:d}".format(num))

            # get the electric field for current time and make sure it has length
            # 3
            efield, hold, efcntrl = [], [], "666"
            for idx, ctype in enumerate(control):
                if int(ctype) == 6:
                    efield.append(cij[idx])
                    hold.append(idx)
                    continue
            efield.extend([0.] * (3 - len(efield)))

            # separate out electric fields from deformations. electric field
            # will be appended to end of control list
            cij = [i for j, i in enumerate(cij) if j not in hold]
            control = "".join(
                [i for j, i in enumerate(control) if j not in hold])

            if len(control) != len(cij):
                pu.report_and_raise_error(
                    "Intermediate length of leg control != number of "
                    "control items in leg {0}".format(num))

            # make sure that the control is consistent with the limitations set by
            # Payette
            if re.search(r"5", control):
                # deformation gradient control check
                if re.search(r"[^5]", control):
                    pu.report_and_raise_error(
                        "Only components of deformation gradient "
                        "are allowed with deformation gradient "
                        "control in leg {0}, got '{1}'".format(num, control))

                # check for valid deformation
                defgrad = np.array([[cij[0], cij[1], cij[2]],
                                    [cij[3], cij[4], cij[5]],
                                    [cij[6], cij[7], cij[8]]])
                jac = np.linalg.det(defgrad)
                if jac <= 0:
                    pu.report_and_raise_error(
                        "Inadmissible deformation gradient in leg "
                        "{0} gave a Jacobian of {1:f}".format(num, jac))

                # convert defgrad to strain E with associated rotation given by
                # axis of rotation x and angle of rotation theta
                rot, lstretch = np.linalg.qr(defgrad)
                if np.max(np.abs(rot - np.eye(3))) > np.finfo(np.float).eps:
                    pu.report_and_raise_error(
                        "Rotation encountered in leg {0}. ".format(num) +
                        "rotations are not yet supported")

            elif re.search(r"8", control):
                # displacement control check

                # like deformation gradient control, if displacement is specified
                # for one, it must be for all
                if re.search(r"[^8]", control):
                    pu.report_and_raise_error(
                        "Only components of displacment are allowed with "
                        "displacment control in leg {0}, got '{1}'"
                        .format(num, control))

                # must specify all components
                elif len(cij) != 3:
                    pu.report_and_raise_error(
                        "all 3 components of displacement must "
                        "be specified for leg {0}".format(num))

                # convert displacments to strains
                # Seth-Hill generalized strain is defined
                # strain = (1/kappa)*[(stretch)^kappa - 1]
                # and
                # stretch = displacement + 1

                # In the limit as kappa->0, the Seth-Hill strain becomes
                # strain = ln(stretch).
                for j in range(3):
                    stretch = self.dfac * cij[j] + 1
                    if self._kappa != 0:
                        cij[j] = 1 / self._kappa * \
                            (stretch ** self._kappa - 1.)
                    else:
                        cij[j] = math.log(stretch)
                    continue

                # displacements now converted to strains
                control = "222222"
                cij.extend([0., 0., 0.])

            elif re.search(r"\b2\b", control):
                # only one strain value given -> volumetric strain
                evol = cij[0] * self.efac
                if self._kappa * evol + 1. < 0.:
                    pu.report_and_raise_error(
                        "1 + kappa * ev must be positive")

                if self._kappa == 0.:
                    eij = evol / 3.

                else:
                    eij = ((self._kappa * evol + 1.) ** (1. / 3.) - 1.)
                    eij = eij / self._kappa

                control = "222222"
                cij = [eij, eij, eij, 0., 0., 0.]
                efac_hold, self.efac = self.efac, 1.

            elif re.search(r"\b4\b", control):
                # only one stress value given -> pressure
                pres = cij[0] * self.sfac
                sij = -1. * pres
                control = "444444"
                cij = [sij, sij, sij, 0., 0., 0.]
                sfac_hold, self.sfac = self.sfac, 1.

            # fill in cij and control so that their lengths are always 9
            # the electric field control is added to the end of control
            if len(control) != len(cij):
                pu.report_and_raise_error(
                    "Final length of leg control != number of "
                    "control items in leg {0}".format(num))

            L = len(cij)
            cij.extend([0.] * (9 - L) + efield)
            control += "0" * (9 - L) + efcntrl

            # we have read in all controled items and checked them, now we
            # adjust them based on user input
            for idx, ctype in enumerate(control):
                ctype = int(ctype)
                if ctype == 1 or ctype == 3:
                    # adjust rates
                    cij[idx] = self.ratfac * cij[idx]

                elif ctype == 2:
                    # adjust strain
                    cij[idx] = self.efac * cij[idx]

                    if self._kappa * cij[idx] + 1. < 0.:
                        pu.report_and_raise_error(
                            "1 + kappa*c[{0}] must be positive".format(idx))

                elif ctype == 4:
                    # adjust stress
                    cij[idx] = self.sfac * cij[idx]

                elif ctype == 5:
                    # adjust deformation gradient
                    cij[idx] = self.ffac * cij[idx]

                elif ctype == 6:
                    # adjust electric field
                    cij[idx] = self.effac * cij[idx]

                continue

            try:
                self.efac = efac_hold
            except NameError:
                pass
            try:
                self.sfac = sfac_hold
            except NameError:
                pass

            # convert cij to array
            cij = np.array(cij)

            # append to legs
            if ltime == 0.:
                if re.search(r"3", control):
                    import Source.Payette_utils as pu
                    pu.report_and_raise_error(
                        "initial stress rate ambiguous")
                elif re.search(r"4", control):
                    # check if nonzero initial stress was given
                    if np.any(np.abs(cij[:6]) > 0.):
                        self.bcontrol("initial stress", True)

            self._legs.append([num, ltime, steps, control, cij])

            if table:
                # increment
                num += 1
            continue

        if stress_control:
            # stress and or stress rate is used to control this leg. For
            # these cases, kappa is set to 0. globally.
            if self._kappa != 0.:
                self._kappa = 0.
                pu.log_warning(
                    "Stress control boundary conditions "
                    "only compatible with kappa=0. Kappa is being "
                    "reset to 0. from {0:f}\n".format(kappa))
                self.bcontrol("kappa", 0.)

        # check that time is monotonic in lcontrol and find total number of
        # steps to be completed
        time_0, time_f = 0., 0.
        for ileg, leg in enumerate(self._legs):
            self._nsteps += leg[2]
            if ileg == 0:
                # set the initial time
                time_0 = leg[1]
                continue

            time_f = leg[1]
            if time_f <= time_0:
                pu.report_and_raise_error(
                    "time must be monotonic from {0:d} to {1:d}"
                    .format(leg[0] - 1, leg[0]))

            time_0 = time_f

        if not ileg:
            pu.report_and_raise_error("Only one time step found.")

        return

    def _parse_leg_table_header(self, header):
        """Parse the first leg of the legs block.

        The first leg of the legs block may be in one of two forms.  The usual

                      <leg_no>, <leg_t>, <leg_steps>, <leg_cntrl>, <c[ij]>

        or, if the user is prescribing the legs through a table

                      using <time, dt>, <deformation type> [from columns ...]

        here, we determine what kind of legs the user is prescrbing.

        Parameters
        ----------
        header : str
            Header of the first leg in the legs block of the user input

        Returns
        -------
        ttype : str
            time type
        cidxs : list
            column indexes from table
        control : str
            control string

        Raises
        ------

        Examples
        --------
        >>> parse_first_leg("using dt strain")
        "dt", [0,1,2,3,4,5,6], "222222"

        >>> parse_first_leg("using dt, strain, from columns 1:7")
        "dt", [0,1,2,3,4,5,6], "222222"

        >>> parse_first_leg([using dt stress from columns, 1,2,3,4,5,6,7")
        "dt", [0,1,2,3,4,5,6], "444444"

        >>> parse_first_leg("using dt strain from columns 1-7")
        "dt", [0,1,2,3,4,5,6], "222222"

        >>> parse_first_leg("using dt strain from columns, 1, 5 - 10")
        "dt", [0,4,5,6,7,8,9,20], "222222"

        >>> parse_first_leg("using dt strain from columns, 1, 5:7"])
        "dt", [0,4,5,6], "222"

        """

        # determine the time specifier
        ttypes = ("time", "dt")
        NT = 1
        for item in ttypes:
            T = re.search(r"(?i)\b{0}\b".format(item), header)
            if T:
                s, e = T.start(), T.end()
                T = header[s:e].lower()
                break
            continue
        if T is None:
            pu.report_and_raise_error(
                "time specifier '{0}' not found.  Choose from {1}"
                .format(T, ", ".join(ttypes)))
        header = re.sub(r"(?i)\b{0}\b".format(T), "", header).strip()

        # determine the deformation specifier
        dtype = re.sub(r"[^\S\n]+", " ", re.sub(r"[\,]", " ", header)).strip()
        efield, first = re.search(r"(?i)\befield\b", dtype), False
        C, NC, EFC, NEFC = (0, ) * 4

        pat = r"(?i)\bfrom.*columns\b"
        cspec = re.search(pat, dtype)
        if cspec is not None:
            # user specified something like
            # using strain, from columns ...
            s, e = cspec.start(), cspec.end()
            cspec = re.sub(pat, "", dtype[s:]).strip()

            # cspec MUST be specified last
            if cspec[-1] != dtype[-1]:
                pu.report_and_raise_error(
                    "'from columns ...' directive must come "
                    "after the 'using ...' directive")
            dtype = dtype[:s]

        dtype = dtype.strip()
        if efield:
            # we need to know if the efield was specified first or second
            efield = re.search(r"(?i)\befield\b", dtype)
            s, e = efield.start(), efield.end()
            first = e != len(dtype)
            dtype = dtype[:s] + dtype[e:]
            EFC, NEFC = dtypes("efield")

        try:
            C, NC = dtypes(dtype)
        except TypeError:
            pu.report_and_raise_error(
                "control type {0} not recognized".format(dtype))

        if cspec is None:
            # use default column indices
            cols = range(NT + NC + NEFC)

        else:
            # user could have specified something like 4 - 6 to indicate
            # columns 4 - 6, we replace - with :
            RSEP = r":"
            cspec = re.sub(r"-", RSEP, cspec).split()

            # loop through cspec and pair range specifications from column
            # specifications
            cols, skip = [], -1
            for idx, item in enumerate(cspec):
                if idx == skip:
                    continue
                try:
                    s, e = item.split(RSEP)
                except ValueError:
                    cols.append(int(item) - 1)
                    continue
                # range sepecified, get start and end and store as tuple
                s = int(s) - 1 if s else cols.pop()
                if not e:
                    e = cspec[idx + 1]
                    skip = idx + 1
                cols.extend(range(int(s), int(e)))
                continue

        if len(cols) > NC + NEFC + NT:
            pu.report_and_raise_error("Too many columns specified")
        if len(cols) < NC + NEFC + NT:
            pu.report_and_raise_error("Too few columns specified")

        control = "{0}".format(C) * NC
        if efield:
            efc = "{0}".format(EFC) * NEFC
            control = efc + control if first else control + efc

        return T, cols, control


def dtypes(dtype=None):
    """The deformationt types allowed for by Payette

    Parameters
    ----------
    dtype : str, optional [None]
        deformation type

    Returns
    -------
    C : int
        Integer ID for deformation type
    NC : int
        Length of deformation type
    """
    if dtype is None:
        return DTYPES.keys()
    return DTYPES.get(dtype.strip().lower())


class EOSBoundary(object):
    """The EOS boundary class"""

    def __init__(self, bblock, lblock):

        if not bblock.split("\n"):
            pu.report_and_raise_error("boundary block not found")

        self.boundary = bblock
        self.legs = lblock
        self._nsteps = 0

        # parse the boundary block
        self.allowed_unit_systems = ("MKSK", "CGSEV",)
        self.bcontrol = {
            "nprints": {"value": 4, "type": int},
            "input units": {"value": None, "type": "choice",
                            "choices": self.allowed_unit_systems},
            "output units": {"value": None, "type": "choice",
                             "choices": self.allowed_unit_systems},
            "density range": {"value": [0.0, 0.0], "type": list},
            "temperature range": {"value": [0.0, 0.0], "type": list},
            "surface increments": {"value": 10, "type": int},
            "path increments": {"value": 100, "type": int},
            "path isotherm": {"value": None, "type": list},
            "path hugoniot": {"value": None, "type": list},
        }
        self.parse_boundary_block()

        # parse the legs block
        self.lcontrol = []
        if self.legs:
            self.parse_legs_block()

        # determine the number of steps. In the most general case, the number
        # of steps N is:

        # N = len(density temperature pairs) + (surface increments) ** 2
        #   + 2 * (path increments)
        self._nsteps = len(self.lcontrol)
        Rrange = self.density_range()
        Trange = self.temperature_range()
        Sinc = self.surface_increments()
        Pinc = self.path_increments()
        isotherm = self.path_isotherm()
        hugoniot = self.path_hugoniot()
        if all(x is not None for x in (Sinc, Rrange, Trange,)):
            self._nsteps += Sinc ** 2
        if all(x is not None for x in (Pinc, isotherm, Rrange,)):
            self._nsteps += Pinc
        if all(x is not None for x in (Pinc, hugoniot, Trange, Rrange,)):
            self._nsteps += Pinc
        pass

    def parse_boundary_block(self):
        """parse the eos boundary block"""

        # get the input units
        iu = pip.get("input units", self.boundary)
        if iu is None:
            pu.report_and_raise_error(
                "Input units not found in boundary block, choose from: {0}"
                .format("input units " + ", ".join(self.allowed_unit_systems)))
        iu = iu.upper()
        if iu not in self.bcontrol["input units"]["choices"]:
            pu.report_and_raise_error("Unrecognized input unit system: '{0}'"
                                      .format(iu))
        self.bcontrol["input units"]["value"] = iu

        # get the output units
        ou = pip.get("output units", self.boundary)
        if ou is None:
            pu.report_and_raise_error(
                "Output units not found in boundary block, choose from: {0}"
                .format("input units " + ", ".join(self.allowed_unit_systems)))
        ou = ou.upper()
        if ou not in self.bcontrol["output units"]["choices"]:
            pu.report_and_raise_error("Unrecognized out unit system: '{0}'"
                                      .format(ou))
        self.bcontrol["output units"]["value"] = ou

        # nprints
        self.bcontrol["nprints"]["value"] = int(
            pip.get("nprints", self.boundary, 4))

        # density range
        val = sorted(
            [float(x) for x in
             pip.get("density range", self.boundary, "0. 0.").split()])
        if len(val) != 2 or val[0] == val[1]:
            pu.report_and_raise_error(
                "Unacceptable density range in boundary block.")
        self.bcontrol["density range"]["value"] = val

        # temperature range
        val = sorted(
            [float(x) for x in
             pip.get("temperature range", self.boundary, "0. 0.").split()])
        if len(val) != 2 or val[0] == val[1]:
            pu.report_and_raise_error(
                "Unacceptable temperature range in boundary block.")
        self.bcontrol["temperature range"]["value"] = val

        # surface increments
        val = int(pip.get("surface increments", self.boundary, 10))
        if val <= 0:
            pu.report_and_raise_error("Number of surface increments must be "
                                      "positive non-zero.")
        self.bcontrol["surface increments"]["value"] = val

        # path increments
        val = int(pip.get("path increments", self.boundary, 100))
        if val <= 0:
            pu.report_and_raise_error("Number of path increments must be "
                                      "positive non-zero.")
        self.bcontrol["path increments"]["value"] = val

        # the following depend on the density and temperature ranges that were
        # previously read and parsed.
        rho_0, rho_f = self.bcontrol["density range"]["value"]
        tmpr_0, tmpr_f = self.bcontrol["temperature range"]["value"]

        # isotherm expects [density, temperature]
        val = pip.get("path isotherm", self.boundary)
        if val is not None:
            isotherm = [float(x) for x in val.split()]
            bad_rho = not rho_0 <= isotherm[0] <= rho_f
            bad_temp = not tmpr_0 <= isotherm[1] <= tmpr_f
            if len(isotherm) != 2 or bad_rho or bad_temp:
                pu.report_and_raise_error("Bad initial state for isotherm.")
            self.bcontrol["path isotherm"]["value"] = isotherm

        val = pip.get("path hugoniot", self.boundary)
        if val is not None:
            # isotherm expects [density, temperature]
            hugoniot = [float(x) for x in val.split()]
            bad_rho = not rho_0 <= hugoniot[0] <= rho_f
            bad_temp = not tmpr_0 <= hugoniot[1] <= tmpr_f
            if len(hugoniot) != 2 or bad_rho or bad_temp:
                pu.report_and_raise_error("Bad initial state for hugoniot.")
            self.bcontrol["path hugoniot"]["value"] = hugoniot

        return

    def parse_legs_block(self):
        """Parse the eos legs block"""
        # parse parameters. If multiple declarations, use the last.
        # --- LEGS

        # the legs block of an eos simulation contains density/temperature
        # pairs of the form:
        #
        #       rho tmpr
        ir, it = 0, 1
        using = re.search(r"(?i)\busing\b.*", self.legs)
        if using:
            s, e = using.start(), using.end()
            line = [x.lower() for x in
                    re.sub(r"(?i)\busing\b|\,", " ", self.legs[s:e]).split()]
            self.legs = (self.legs[:s] + self.legs[e:]).strip()
            for idx, item in enumerate(line):
                if item == "density":
                    ir = idx
                elif item == "temperature":
                    it = idx
                else:
                    pu.report_and_raise_error("unrecognized: {0}".format(item))
                continue

        for item in self.legs.split("\n"):
            if not item.split():
                continue
            item = [float(x) for x in item.split()]
            if len(item) != 2:
                pu.report_and_raise_error(
                    "legs must be of form rho tmpr, got: {0}"
                    .format(" ".join(item)))
            self.lcontrol.append([item[ir], item[it]])
            continue
        return

    def nprints(self):
        """Number of times to print to screen during a boundary leg

        If nprints in zero, use default

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        nprints : int
          Number of times to print to screen during simulation leg if nonzero

        """
        return self.bcontrol["nprints"]["value"]

    def input_units(self):
        """Unit system in which parameters are given

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        input_units : str
          The unit system

        """
        return self.bcontrol["input units"]["value"]

    def output_units(self):
        """Unit system in which results are given

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        output_units : str
          The unit system

        """
        return self.bcontrol["output units"]["value"]

    def density_range(self):
        """Range of densities

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        density_range : array_like
          The density range

        """
        return self.bcontrol["density range"]["value"]

    def temperature_range(self):
        """Range of temperatures

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        temperature_range : array_like
          The temperature range

        """
        return self.bcontrol["temperature range"]["value"]

    def surface_increments(self):
        """Number of surface increments

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        surface_increments : int
          Number of surface increments to compute

        """
        return self.bcontrol["surface increments"]["value"]

    def path_increments(self):
        """Number of path increments

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        path_increments : int
          Number of path increments to compute

        """
        return self.bcontrol["path increments"]["value"]

    def path_isotherm(self):
        """Unfinished docstring"""
        return self.bcontrol["path isotherm"]["value"]

    def path_hugoniot(self):
        """Unfinished docstring"""
        return self.bcontrol["path hugoniot"]["value"]

    def rho_temp_pairs(self):
        """Unfinished docstring"""
        return self.lcontrol

    def get_leg_control_params(self):
        """Unfinished docstring"""
        return self.lcontrol

    def get_boundary_control_params(self):
        """Unfinished docstring"""
        return self.bcontrol

    def initial_time(self):
        """no initial or termination time explicitly set for EOS simulations"""
        return None

    def termination_time(self):
        """no initial or termination time explicitly set for EOS simulations"""
        return None

    def nsteps(self):
        return self._nsteps
