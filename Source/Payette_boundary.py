# The MIT License
#
# Copyright (c) 2011 Tim Fuller
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
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
import numpy as np

import Source.runopts as ro

class BoundaryError(Exception):
    """Boundar exception class"""
    def __init__(self, message):
        from Source.Payette_utils import who_is_calling
        caller = who_is_calling()
        self.message = message + " [reported by {0}]".format(caller)
        super(BoundaryError, self).__init__(self.message)


class Boundary(object):
    """Payette boundary class

    Raises
    ------
    BoundaryError

    """

    def __init__(self, *args, **kwargs):

        boundary = kwargs.get("boundary")
        legs = kwargs.get("legs")
        efield = kwargs.get("efield")

        if boundary is None:
            raise BoundaryError("boundary block not found")

        if legs is None:
            raise BoundaryError("legs block not found")

        # passed values
        self.boundary = boundary
        self.legs = legs
        self.efield = efield

        # defaults
        self.boundary_warnings = 0
        self.initial_time = None
        self.termination_time = None

        self.bcontrol = {
            "kappa": {"value": 0., "type": float},
            "estar": {"value": 1., "type": float},
            "tstar": {"value": 1., "type": float},
            "sstar": {"value": 1., "type": float},
            "fstar": {"value": 1., "type": float},
            "dstar": {"value": 1., "type": float},
            "efstar": {"value": 1., "type": float},
            "stepstar": {"value": 1., "type": float, "min": 1.},
            "ampl": {"value": 1., "type": float},
            "ratfac": {"value": 1., "type": float},
            "emit": {"value": "all", "type": "choice",
                     "choices": ("all", "sparse",)},
            "nprints": {"value": 0, "type": int},
            "screenout": {"value": False, "type": bool},
            }

        self.tfac = 1.
        self.efac = 1.
        self.sfac = 1.
        self.ffac = 1.
        self.effac = 1.
        self.dfac = 1.
        self.stepstar = 1.
        self.ratfac = 1.

        # parse the boundary block
        self.user_control_options = {}
        self.parse_boundary_block()

        # set up for parsing the legs block
        self.allowed_legs = {
            "strain rate": {"num": 1, "len": 6},
            "strain": {"num": 2, "len": 6},
            "stress rate": {"num": 3, "len": 6},
            "stress": {"num": 4, "len": 6},
            "deformation gradient": {"num": 5, "len": 9},
            "electric field": {"num": 6, "len": 3},
            "displacement": {"num": 8, "len": 3},
            "vstrain": {"num": 2, "len": 1},
            "pressure": {"num": 4, "len": 1}}
        self.allowed_time_specifier = ["time", "dt"]
        self.leg_table_data = None
        self.lcontrol = []
        self.table_input = False

        # parse the legs block
        self.parse_legs_block()

    def log_warning(self, msg):
        """Log warning to stderr

        Parameters
        ----------
        msg : str
          message to be printed to stderr

        """
        if ro.WARNING == "ignore":
            return
        sys.stdout.flush()
        sys.stderr.write("WARNING: {0}\n".format(msg))
        self.boundary_warnings += 1
        return

    def parse_boundary_block(self):
        """Scan the user input for a begin boundary .. end boundary block and
        parse it

        Parameters
        ----------

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

        for item in self.boundary:
            for char in "=,;":
                item = item.replace(char, " ")
            item = " ".join(item.split())

            try:
                kwd, val = item.split()
            except ValueError:
                raise BoundaryError(
                    "boundary control items must be key = val pairs")

            if kwd.lower() not in self.bcontrol:
                try:
                    self.user_control_options[kwd] = eval(val)
                except (TypeError, ValueError):
                    self.user_control_options[kwd] = val

            else:
                kwd = kwd.lower()
                kwd_type = self.bcontrol[kwd]["type"]
                if kwd_type == "choice":
                    choices = self.bcontrol[kwd]["choices"]
                    if val not in choices:
                        raise BoundaryError(
                            "{0} must be one of {1}, got {2}"
                            .format(kwd, ", ".join(choices), val))
                else:
                    val = kwd_type(val)
                    val_min = self.bcontrol[kwd].get("min")
                    if val_min is not None:
                        val = max(val_min, val)

                self.bcontrol[kwd]["value"] = val

            continue

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

        # From these formulas, note that AMPL may be used to increase or
        # decrease the peak strain without changing the strain rate. ratfac is
        # the multiplier on strain rate and stress rate.
        ampl = self.bcontrol["ampl"]["value"]
        tstar = self.bcontrol["tstar"]["value"]
        self.ratfac = self.bcontrol["ratfac"]["value"]
        self.stepstar = self.bcontrol["stepstar"]["value"]
        estar = self.bcontrol["estar"]["value"]
        sstar = self.bcontrol["sstar"]["value"]
        fstar = self.bcontrol["fstar"]["value"]
        efstar = self.bcontrol["efstar"]["value"]
        dstar = self.bcontrol["dstar"]["value"]
        self.tfac = abs(ampl) * tstar / self.ratfac
        self.efac = ampl * estar
        self.sfac = ampl * sstar
        self.ffac = ampl * fstar
        self.effac = ampl * efstar
        self.dfac = ampl * dstar

        return

    def parse_legs_block(self):
        """Parse the legs block of the user input

        """
        # parse the electric field block first
        efield_blk = self.parse_efield_block()
        efield_blk_times = [x[0] for x in efield_blk]
        efield_blk_vals = [x[1] for x in efield_blk]

        stress_control = False
        kappa = self.kappa()

        # control should be a group of letters describing what type of
        # control type the leg is. valid options are:
        #  1: strain rate control
        #  2: strain control
        #  3: stress rate control
        #  4: stress control
        #  5: deformation gradient control
        #  6: electric field
        #  8: displacement
        for ileg, leg in enumerate(self.legs):

            # for each leg, we parse the line to define the following:
            # control: control type
            # leg_no: leg number
            # leg_t: time at end of leg
            # leg_steps: number of steps for the leg
            # lcntrl: mechanical control
            # efcntrl: electric field control
            # cij: values of deformation
            # efield: electric field values

            leg = " ".join(leg.replace(",", " ").split())
            leg = leg.split()

            if ileg == 0:
                self.parse_first_leg(leg)

                if self.table_input:
                    # the first leg let us know the user specified a table,
                    # continue on to the actual table
                    g_leg_no = 0
                    g_time = 0.
                    continue

            if self.table_input:
                # user is specifying deformation by using a table

                if leg[0].lower() == "time":
                    # table could have had a header row with time as first
                    # entry
                    continue

                control = self.leg_table_data["control"]
                leg_steps = int(self.stepstar)
                leg_no = g_leg_no
                g_leg_no += 1
                if self.leg_table_data["ttyp"] == "dt":
                    g_time += float(leg[self.leg_table_data["col_idxs"][0]])
                else:
                    g_time = float(leg[self.leg_table_data["col_idxs"][0]])

                # adjust the actual time using the time factor
                leg_t = self.tfac * g_time

                try:
                    cij = [float(eval(leg[x])) for x in
                           self.leg_table_data["col_idxs"][1:]]
                except (IndexError, ValueError):
                    raise BoundaryError(
                        "syntax error in leg {0}".format(leg[0]))

            else:

                # user specified leg in form:
                # number, time, steps, control, values

                # leg must have at least 5 values
                if len(leg) < 5:
                    raise BoundaryError(
                        "leg {0} input must be of form:".format(leg[0]) +
                        "\n       leg number, time, steps, type, c[ij]")

                # get the leg number, time, steps
                leg_no = int(leg[0])
                leg_t = float(self.tfac * float(leg[1]))
                leg_steps = int(self.stepstar * float(leg[2]))
                if ileg != 0 and leg_steps == 0:
                    raise BoundaryError(
                        "leg number {0} has no steps".format(leg_no))

                # get the control type
                control = leg[3].strip()

                # the remaining part of the line are the actual ij values of
                # the deformation type
                try:
                    cij = [float(eval(y)) for y in leg[4:]]
                except ValueError:
                    raise BoundaryError(
                        "syntax error in leg {0}".format(leg[0]))

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
                msg = ("leg control parameters can only be one of "
                       "[{0}] got {1} for leg number {2:d}"
                       .format(allwd_cntrl, control, leg_no))
                raise BoundaryError(msg)

            lcntrl = [int(x) for x in control]

            # stress control if any of the control types are 3 or 4
            if not stress_control:
                stress_control = any([x in (3, 4) for x in lcntrl])

            # we need to know what to do with each deformation value, so the
            # length of the deformation values must be same as the control
            # values
            if len(lcntrl) != len(cij):
                raise BoundaryError(
                    "length of leg control != number of control "
                    "items in leg {0:d}".format(leg_no))

            # get the electric field for current time and make sure it has
            # length 3
            if efield_blk:
                efcntrl, hold = [6, 6, 6], []
                if leg_t < efield_blk_times[0]:
                    # not yet reached time in which efield is given
                    efield = [0., 0., 0.]

                elif leg_t > efield_blk_times[-1]:
                    # past given efield, use last given value
                    efield = efield_blk_vals[-1]

                elif leg_t in efield_blk_times:
                    # time corresponds to a time in the efield block
                    efield = efield_blk_vals[efield_blk_times.index(leg_t)]

                else:
                    # User gave an efield block in the input, interpolate it to
                    # get the right value for the current time
                    t0 = [x for x in efield_blk_times if x < leg_t][-1]
                    tf = [x for x in efield_blk_times if x > leg_t][0]
                    y0 = efield_blk_vals[efield_blk_times.index(t0)]
                    yf = efield_blk_vals[efield_blk_times.index(tf)]
                    efield = [(yf[i] - y0[i]) / (tf - t0) * leg_t
                              for i in range(3)]

            else:
                efield, hold, efcntrl = [], [], []
                for idx, cntrl_type in enumerate(lcntrl):
                    if cntrl_type == 6:
                        if efield_blk:
                            raise BoundaryError(
                                "Encountered efield block and efield "
                                "specification in legs")
                        efield.append(cij[idx])
                        hold.append(idx)
                        efcntrl.append(cntrl_type)
                    continue

                efield.extend([0.] * (3 - len(efield)))
                efcntrl.extend([6] * (3 - len(efcntrl)))

            # separate out electric fields from deformations. electric field
            # will be appended to end of control list
            cij = [i for j, i in enumerate(cij) if j not in hold]
            lcntrl = [i for j, i in enumerate(lcntrl) if j not in hold]

            if len(lcntrl) != len(cij):
                msg = ("final length of leg control != number of "
                       "control items in leg {0:d}".format(leg_no))
                raise BoundaryError(msg)

            # make sure that the lcntrl is consistent with the limitations set
            # by Payette
            reduced_lcntrl = list(set(lcntrl))

            if 5 in reduced_lcntrl:
                # deformation gradient control check

                # if deformation gradient is specified, all control types must
                # be of deformation gradient
                if len(reduced_lcntrl) != 1:
                    msg = ("only components of deformation gradient "
                           "are allowed with deformation gradient "
                           "control in leg {0:d}, got {1}"
                           .format(leg_no, control))
                    raise BoundaryError(msg)

                # user must specify all 9 components
                elif len(cij) != 9:
                    msg = ("all 9 components of deformation gradient "
                           "must be specified for leg {0:d}".format(leg_no))
                    raise BoundaryError(msg)

                else:
                    # check for valid deformation
                    defgrad = np.array([[cij[0], cij[1], cij[2]],
                                        [cij[3], cij[4], cij[5]],
                                        [cij[6], cij[7], cij[8]]])
                    jac = np.linalg.det(defgrad)
                    if jac <= 0:
                        msg = ("inadmissible deformation gradient in leg "
                               "{0:d} gave a Jacobian of {1:f}"
                               .format(leg_no, jac))
                        raise BoundaryError(msg)

                    # convert defgrad to strain E with associated rotation
                    # given by axis of rotation x and angle of rotation theta
                    rot, lstretch = np.linalg.qr(defgrad)
                    if np.max(np.abs(rot - np.eye(3))) > np.finfo(np.float).eps:
                        msg = ("rotation encountered in leg {0}. "
                               .format(leg_no) +
                               "rotations are not yet supported")
                        raise BoundaryError(msg)

            elif 8 in reduced_lcntrl:
                # displacement control check

                # like deformation gradient control, if displacement is
                # specified for one, it must be for all
                if len(reduced_lcntrl) != 1:
                    msg = ("only components of displacment are allowed "
                           "with displacment control in leg {0:d}, got {1}"
                           .format(leg_no, control))
                    raise BoundaryError(msg)

                # must specify all components
                elif len(cij) != 3:
                    msg = ("all 3 components of displacement must "
                           "be specified for leg {0:d}".format(leg_no))
                    raise BoundaryError(msg)

                # convert displacments to strains
                # Seth-Hill generalized strain is defined
                # strain = (1/kappa)*[(stretch)^kappa - 1]
                # and
                # stretch = displacement + 1

                # In the limit as kappa->0, the Seth-Hill strain becomes
                # strain = ln(stretch).
                for j in range(3):
                    stretch = self.dfac * cij[j] + 1
                    if kappa != 0:
                        cij[j] = 1 / kappa * (stretch ** kappa - 1.)
                    else:
                        cij[j] = math.log(stretch)
                    continue

                 # displacements now converted to strains
                lcntrl = [2, 2, 2]

            if lcntrl == [2]:

                # only one strain value given -> volumetric strain
                evol = cij[0] * self.efac
                if kappa * evol + 1. < 0.:
                    raise BoundaryError("1 + kappa*ev must be positive")

                if kappa == 0.:
                    eij = evol / 3.

                else:
                    eij = ((kappa * evol + 1.) ** (1. / 3.) - 1.) / kappa

                lcntrl = [2, 2, 2]
                cij = [eij, eij, eij]
                efac_hold = self.efac
                self.efac = 1.0

            elif lcntrl == [4]:

                # only one stress value given -> pressure
                pres = cij[0] * self.sfac
                sij = -1. * pres
                lcntrl = [4, 4, 4]
                cij = [sij, sij, sij]
                sfac_hold = self.sfac
                self.sfac = 1.0

            # fill in cij and lcntrl so that their lengths are always 9
            # the electric field control is added to the end of lcntrl
            cij.extend([0.] * (6 - len(cij)) + efield)
            lcntrl.extend([0] * (6 - len(lcntrl)) + efcntrl)

            # we have read in all controled items and checked them, now we
            # adjust them based on user input
            for idx, cntrl_type in enumerate(lcntrl):
                if cntrl_type == 1 or cntrl_type == 3:
                    # adjust rates
                    cij[idx] = self.ratfac * cij[idx]

                elif cntrl_type == 2:
                    # adjust strain
                    cij[idx] = self.efac * cij[idx]

                    if kappa * cij[idx] + 1. < 0.:
                        raise BoundaryError(
                            "1 + kappa*c[{0:d}] must be positive".format(idx))

                elif cntrl_type == 4:
                    # adjust stress
                    cij[idx] = self.sfac * cij[idx]

                elif cntrl_type == 5:
                    # adjust deformation gradient
                    cij[idx] = self.ffac * cij[idx]

                elif cntrl_type == 6:
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

            # append leg control
            self.lcontrol.append(
                [leg_no, leg_t, leg_steps, lcntrl, np.array(cij)])

            continue

        if stress_control:
            # stress and or stress rate is used to control this leg. For
            # these cases, kappa is set to 0. globally.
            if kappa != 0.:
                self.log_warning(
                    "WARNING: stress control boundary conditions "
                    "only compatible with kappa=0. kappa is being "
                    "reset to 0. from %f\n"%kappa)
                self.bcontrol["kappa"]["value"] = 0.

        # check that time is monotonic in lcontrol
        time_0, time_f = 0., 0.
        for ileg, leg in enumerate(self.lcontrol):
            if ileg == 0:
                # set the initial time
                self.initial_time = leg[1]
                continue

            time_f = leg[1]
            if time_f <= time_0:
                msg = ("time must be monotonic from {0:d} to {1:d}"
                       .format(leg[0] - 1, leg[0]))
                raise BoundaryError(msg)

            time_0 = time_f

        if not ileg:
            raise BoundaryError("Only one time step found.")

        self.termination_time = time_f

        return

    def parse_first_leg(self, leg):
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

        if "using" not in leg[0]:
            return

        self.table_input = True

        t_typ = leg[1]

        if t_typ not in self.allowed_time_specifier:
            msg = ("requested bad time type {0} in {1}, expected one of [{2}]"
                   .format(t_typ, leg, ", ".join(self.allowed_time_specifier)))
            raise BoundaryError(msg)

        col_spec = [x for x in leg[2:] if "from" in x or "column" in x]

        if not col_spec:
            # default value for col_idxs
            use_typ = " ".join(leg[2:])
            if use_typ not in self.allowed_legs:
                raise BoundaryError(
                    "requested bad control type {0}".format(use_typ))

            col_idxs = range(self.allowed_legs[use_typ]["len"] + 1)

        elif col_spec and len(col_spec) != 2:
            # user specified a line of the form
            # using <dt, time> <deftyp> from ...
            # or
            # using <dt, time> <deftyp> columns ...
            msg = ("expected {0} <deftyp> from columns ..., got {1}"
                   .format(t_typ, leg))
            raise BoundaryError(msg)

        else:
            use_typ = " ".join(leg[2:leg.index(col_spec[0])])
            if use_typ not in self.allowed_legs:
                raise BoundaryError(
                    "requested bad control type {0}".format(use_typ))

            # now we need to find the column indexes
            col_idxs = " ".join(leg[leg.index(col_spec[-1]) + 1:])
            col_idxs = col_idxs.replace("-", ":")

            if ":" in col_idxs.split():
                tmpl = col_idxs.split()

                # user may have specified something like 1 - 6 which would now
                # be 1 : 6, which is a bad range specifier, we need to fix it
                idx = tmpl.index(":")

                if len(tmpl) == 3 and idx == 1:
                    # of form: from columns 1:7
                    col_idxs = "".join(tmpl)

                elif len(tmpl) == 4 and idx != 2:
                    # of form: from columns 1:6, 7 -> not allowed
                    raise BoundaryError(
                        "bad column range specifier in: '{0}'"
                        .format(" ".join(leg)))

                elif len(tmpl) == 4:
                    # of form: from columns 1, 2:7
                    col_idxs = tmpl[0] + " " + "".join(tmpl[1:])

            if col_idxs.count(":") > 1:
                # only one range allowed
                raise BoundaryError(
                    "only one column range supported".format(use_typ))

            col_idxs = col_idxs.split()
            if len(col_idxs) == 1 and not [x for x in col_idxs if ":" in x]:
                # of form: from columns 8 -> not allowed
                raise BoundaryError(
                    "not enough columns specified in: '{0}'"
                    .format(" ".join(leg)))

            elif len(col_idxs) == 1 and [x for x in col_idxs if ":" in x]:
                # of form: from columns 2:8
                col_idxs = col_idxs[0].split(":")
                col_idxs = range(int(col_idxs[0]) - 1, int(col_idxs[1]))

            elif len(col_idxs) == 2 and [x for x in col_idxs if ":" in x]:
                # specified a single index and range
                if col_idxs.index([x for x in col_idxs if ":" in x][0]) != 1:
                    # of form: from columns 2:8, 1 -> not allowed
                    raise BoundaryError(
                        "bad column range specifier in: '{0}'"
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
            if len(col_idxs) > self.allowed_legs[use_typ]["len"] + 1:
                raise BoundaryError("too many columns specified")

        # we have exhausted all ways of specifying columns that I can think
        # of, save the info and return
        self.leg_table_data = {
            "table": True,
            "col_idxs": col_idxs,
            "ttyp": t_typ,
            "deftyp": use_typ,
            "len": self.allowed_legs[use_typ]["len"],
            "control":("{0}".format(self.allowed_legs[use_typ]["num"])
                       * self.allowed_legs[use_typ]["len"])
            }

        return

    def kappa(self):
        """kappa, the Seth-Hill strain measure identifier.

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        kappa : float
          Seth-Hill strain measure identifier

        """
        return self.bcontrol["kappa"]["value"]

    def emit(self):
        """Specifier of how much info to print during simulation

        If emit is "sparse" very little information is printed
        If emit is "all" a lot more is printed

        Parameters
        ----------
        self : class instance
          Boundary class instance

        Returns
        -------
        emit : str

        """
        return self.bcontrol["emit"]["value"]

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

    def screenout(self):
        """Unfinished docstring"""
        return self.bcontrol["screenout"]["value"]

    def get_leg_control_params(self):
        """Unfinished docstring"""
        return self.lcontrol

    def get_boundary_control_params(self):
        """Unfinished docstring"""
        return self.bcontrol

    def parse_efield_block(self):
        """Parse the electric field block

        The user has the option of specifying the electric field either in its
        own standalone block or in a general legs block. Look for the electric
        field block and parse it if encountered.

        Parameters
        ----------
        self : instance
            Boundary instance

        Returns
        -------
        efield : list
            List of form [[t_0, [ef_x, ef_y, ef_z]_0],
                                      .
                          [t_n, [ef_x, ef_y, ef_z]_n]]

        """
        if self.efield is None:
            return []

        # the electric field block contains time - electric field
        # pairs of the form:
        #
        # time efield_x efield_y efield_z
        efield = []
        tn = -1.
        for line in self.efield:
            try:
                line = [float(x) for x in line.split()]
            except ValueError:
                raise BoundaryError("Syntax error at {0}".format(line))
            t, ef = line[0], line[1:]
            if len(ef) != 3:
                raise BoundaryError(
                    "efield vector must be of length 3, got <{0}>"
                    .format(", ".join([str(x) for x in ef])))
            if t < tn:
                raise BoundaryError(
                    "time must be monotonically increasing in efield block")
            tn = t
            efield.append([t, ef])
            continue

        return efield


class EOSBoundary(object):
    """The EOS boundary class"""

    def __init__(self, *args, **kwargs):

        boundary = kwargs.get("boundary")
        legs = kwargs.get("legs")

        if boundary is None:
            raise BoundaryError("boundary block not found")

        self.boundary = boundary
        self.legs = legs

        # parse the boundary block
        self.allowed_unit_systems = ("MKSK", "CGSEV",)
        self.bcontrol = {
            "nprints": {"value": 4, "type": int},
            "input units": {"value": None, "type":"choice",
                            "choices": self.allowed_unit_systems},
            "output units": {"value": None, "type":"choice",
                             "choices": self.allowed_unit_systems},
            "density range": {"value": [0.0, 0.0], "type": list},
            "temperature range": {"value": [0.0, 0.0], "type": list},
            "surface increments": {"value": 10, "type": int},
            "path increments": {"value": 100, "type": int},
            "path isotherm": {"value": None, "type": list},
            "path hugoniot": {"value": None, "type": list},
            }
        self.user_control_options = {}
        self.parse_boundary_block()

        self.lcontrol = []

        # parse the legs block
        self.parse_legs_block()

        # no initial or termination time explicitly set for EOS simulations
        self.initial_time = None
        self.termination_time = None

    def parse_boundary_block(self):
        """parse the eos boundary block"""

        # --- BOUNDARY
        for item in self.boundary:
            for char in "=,;":
                item = item.replace(char, " ")
            item = " ".join(item.split()).split()
            kwd = " ".join(item[0:2]).lower()

            if "nprints" in kwd:
                val = item[1]
                self.bcontrol["nprints"]["value"] = int(val)

            elif kwd == "input units":
                val = item[2]
                choices = self.bcontrol[kwd]["choices"]
                if val.upper() not in choices:
                    raise BoundaryError("Unrecognized input unit system.")
                self.bcontrol[kwd]["value"] = val

            elif kwd == "output units":
                val = item[2]
                choices = self.bcontrol[kwd]["choices"]
                if val.upper() not in choices:
                    raise BoundaryError("Unrecognized output unit system.")
                self.bcontrol[kwd]["value"] = val

            elif kwd == "density range":
                val = [float(x) for x in item[2:4]]
                if len(val) != 2 or val[0] == val[1]:
                    raise BoundaryError(
                        "Unacceptable density range in boundary block.")
                self.bcontrol[kwd]["value"] = sorted(val)

            elif kwd == "temperature range":
                val = [float(x) for x in item[2:4]]
                if len(val) != 2 or val[0] == val[1]:
                    raise BoundaryError(
                        "Unacceptable temperature range in boundary block.")
                self.bcontrol[kwd]["value"] = sorted(val)

            elif kwd == "surface increments":
                val = int("".join(item[2:3]))
                if val <= 0:
                    raise BoundaryError(
                        "Number of surface increments must be "
                        "positive non-zero.")
                self.bcontrol[kwd]["value"] = val

            elif kwd == "path increments":
                val = int(item[2])
                self.bcontrol[kwd]["value"] = val

            elif kwd not in self.bcontrol:
                kwd, val = kwd
                try:
                    self.user_control_options[kwd] = eval(val)
                except (TypeError, ValueError):
                    self.user_control_options[kwd] = val

            continue

        # check that user has specified the minimum input
        if self.bcontrol["input units"] is None:
            msg = ("Missing 'input units XYZ' keyword in boundary block.\n"
                   "Please include that line with one of the following\n"
                   "unit systems:\n" + "\n".join(self.allowed_unit_systems))
            raise BoundaryError(msg)

        if self.bcontrol["output units"] is None:
            msg = ("Missing 'output units XYZ' keyword in boundary block.\n"
                   "Please include that line with one of the following\n"
                   "unit systems:\n" + "\n".join(self.allowed_unit_systems))
            raise BoundaryError(msg)

        # the following depend on the density and temperature ranges that were
        # previously read and parsed.
        rho_0, rho_f = self.bcontrol["density range"]["value"]
        tmpr_0, tmpr_f = self.bcontrol["temperature range"]["value"]
        for item in self.boundary:
            for char in "=,;":
                item = item.replace(char, " ")
            item = item.split()
            kwd = " ".join(item[0:2]).lower()
            if kwd not in ("path isotherm", "path hugoniot",):
                continue

            if kwd == "path isotherm":
                # isotherm expects [density, temperature]
                isotherm = [float(x) for x in item[2:4]]
                bad_rho = not rho_0 <= isotherm[0] <= rho_f
                bad_temp = not tmpr_0 <= isotherm[1] <= tmpr_f
                if len(isotherm) != 2 or bad_rho or bad_temp:
                    raise BoundaryError("Bad initial state for isotherm.")
                self.bcontrol[kwd]["value"] = isotherm


            elif kwd == "path hugoniot":
                # isotherm expects [density, temperature]
                hugoniot = [float(x) for x in item[2:4]]
                bad_rho = not rho_0 <= hugoniot[0] <= rho_f
                bad_temp = not tmpr_0 <= hugoniot[1] <= tmpr_f
                if len(hugoniot) != 2 or bad_rho or bad_temp:
                    raise BoundaryError("Bad initial state for hugoniot.")
                self.bcontrol[kwd]["value"] = hugoniot

            continue

        return

    def parse_legs_block(self):
        """Parse the eos legs block"""
        # parse parameters. If multiple declarations, use the last.
        # --- LEGS

        if self.legs is None:
            return

        # the legs block of an eos simulation contains density/temperature
        # pairs of the form:
        #
        #       rho tmpr
        for item in self.legs:
            item = item.strip().split()
            vals = [float(x) for x in item]
            if len(vals) != 2:
                raise BoundaryError(
                    "legs must be of form rho tmpr, got: {0}"
                    .format(" ".join(item)))
            self.lcontrol.append(vals)
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

