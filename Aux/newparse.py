#!/usr/bin/env python
import re, sys, os
from textwrap import fill as textfill
import math
import numpy as np

import runopts as ro
import Payette_utils as pu

# --- module leve constants
I_EQ = r"[:,=]"
I_SEP = r"[:,;]"
DTYPES = {"strain rate": (1, 6), "strain": (2, 6), "stress rate": (3, 6),
          "stress": (4, 6), "deformation gradient": (5, 9),
          "electric field": (6, 3), "displacement": (8, 3), "vstrain": (2, 1),
          "pressure": (4, 1),}

class InputParserError(Exception):
    def __init__(self, message):
        if not ro.DEBUG:
            sys.tracebacklimit = 0
        caller = pu.who_is_calling()
        self.message = message + " [reported by {0}]".format(caller)
        super(InputParserError, self).__init__(message)
        pass


class InputParser(object):
    """Payette user input class

    Reads and sets up blocks from user input

    Raises
    ------
    InputParserError

    """

    def __init__(self, ilines=None):
        """Initialize the InputParser object.

        Parameters
        ----------
        ilines : str
            The user input

        Notes
        -----
        ilines should be obtained by first sending the user input through
        parse_user_input

        """

        if ilines is None:
            raise InputParserError("No user input sent to InputParser")

        # --- required information ------------------------------------------ #
        self.inp = find_block("input", ilines)
        if self.inp is None:
            raise InputParserError("User input not found")

        # find the contents of the input block, popping found content along
        # the way
        content = get_content(self.inp)
        self.name, content = find_item_name(content, "name", pop=True)
        if self.name is None:
            raise InputParserError("Simulation name not found")
        self.stype, content = find_item_name(content, "type", pop=True)
        self._options = parse_options(content)
        pass

    def find_block(self, name, default=None):
        """Class method to the public find_block method """
        return find_block(name, self.inp, default=default)

    def find_nested_blocks(self, major, nested, default=None):
        """Class method to the public find_nested_blocks """
        return find_nested_blocks(major, nested, self.inp, default=default)

    def options(self):
        return self._options

    def user_input(self, pop=None):
        lines = "begin input\n{0}\nend input".format(self.inp.strip())
        if pop is None:
            return lines
        if not isinstance(pop, (list, tuple)):
            pop = [pop]
        for item in pop:
            lines = pop_block(item, lines)
            continue
        return lines


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
    N : int
        Length of deformation type
    """
    if dtype is None:
        return DTYPES.keys()
    return DTYPES.get(dtype.lower())


class Boundary(object):
    def __init__(self, bblock, lblock):

        # ---- set defaults ------------------------------------------------- #
        self._kappa = 0.
        self.stepfac, self.efac, self.tfac, self.sfac, self.ffac = [1.] * 5
        self.effac, self.dfac = [1.] * 2

        # structure of _bcontrol:
        # _bcontrol[key] = [type, default, extra [min, choices,]]
        self._bcontrol = {
            "kappa": [float, 0., None], "estar": [float, 1., None],
            "tstar": [float, 1., None], "sstar": [float, 1., None],
            "fstar": [float, 1., None], "dstar": [float, 1., None],
            "efstar": [float, 1., None], "stepstar": [float, 1., 1.],
            "ampl": [float, 1., None], "ratfac": [float, 1., None],
            "emit": ["choice", "all", ("all", "sparse",)],
            "nprints": [int, 0, None], "screenout": [bool, False, None],}

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
        return self.bcontrol("emit")[1]

    def screenout(self):
        return self.bcontrol("screenout")[1]

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
            line = re.sub(I_EQ, " ", line).split()
            if not line:
                continue

            if len(line) != 2:
                raise InputParserError(
                    "Boundary control items must be key = val pairs, got (0}"
                    .format(line))

            kwd, val = line
            if kwd.lower() not in self.bcontrol():
                try:
                    boundary_options[kwd] = eval(val)
                except (TypeError, ValueError):
                    boundary_options[kwd] = val
                continue

            bc = self.bcontrol(kwd) # [type, default, extra]
            if bc[0] == "choice":
                choices = bc[2]
                if val not in choices:
                    raise InputParserError(
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
        ratfac = self.bcontrol("ratfac")[1]
        estar = self.bcontrol("estar")[1]
        sstar = self.bcontrol("sstar")[1]
        fstar = self.bcontrol("fstar")[1]
        efstar = self.bcontrol("efstar")[1]
        dstar = self.bcontrol("dstar")[1]

        # factors to be applied to deformation types
        self._kappa = self.bcontrol("kappa")[1]
        self.stepfac = self.bcontrol("stepstar")[1]
        self.efac = ampl * estar
        self.tfac = abs(ampl) * tstar / ratfac
        self.sfac = ampl * sstar
        self.ffac = ampl * fstar
        self.effac = ampl * efstar
        self.dfac = ampl * dstar
        return

    def bcontrol(self, btype=None, value=None):
        """The boundary control parameters allowed for by Payette

        Parameters
        ----------
        btype : str, optional [None]
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

        if btype is None:
            return self._bcontrol.keys()

        btype = btype.lower()
        if btype not in self._bcontrol:
            raise InputParserError(
                "{0} not a valid bcontrol parameter".format(btype))

        if value is None:
            return self._bcontrol[btype]

        default = self._bcontrol[btype]
        try:
            default[1] = default[0](value)
        except TypeError:
            default[1] = value
        self._bcontrol[btype] = default
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

        if table:
            num = -1
            s, e = using.start(), using.end()
            line = re.sub(r"(?i)\busing\b", " ", lblock[s:e])
            lblock = (lblock[:s] + lblock[e:]).strip()
            ttype, cidxs, gcontrol = self._parse_leg_table_header(line)

        # --- first leg parsed, now go through rest
        time = 0.
        for iline, line in enumerate(lblock.split("\n")):
            line = re.sub(I_SEP, " ", line)
            line = line.split()
            if not line:
                continue

            if table:
                num += 1
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
                    raise InputParserError(
                        "Syntax error in leg {0}".format(num))
            else:
                # user specified leg in form:
                # time, steps, control, values

                # leg must have at least 5 values
                if len(line) < 5:
                    raise InputParserError(
                        "leg {0} input must be of form:".format(num) +
                        "\n\tnum, time, steps, type, c[ij]")

                num = int(line[0])
                ltime = float(self.tfac * float(line[1]))
                steps = int(self.stepfac * float(line[2]))
                if num != 0 and steps == 0:
                    raise InputParserError(
                        "Leg number {0} has no steps".format(num))

                # get the control type
                control = line[3].strip()

                # the remaining part of the line are the actual ij values of the
                # deformation type
                try:
                    cij = [float(eval(x)) for x in line[4:]]
                except ValueError:
                    raise BoundaryError("Syntax error in leg {0}".format(num))

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
                raise InputParserError(
                    "Leg control parameters can only be one of [1234568]"
                    "got {0} for leg number {1:d}".format(control, num))

            # stress control if any of the control types are 3 or 4
            if not stress_control:
                stress_control = any([x in "34" for x in control])

            # we need to know what to do with each deformation value, so the
            # length of the deformation values must be same as the control values
            if len(control) != len(cij):
                raise InputParserError(
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
                raise InputParserError(
                    "Intermediate length of leg control != number of "
                    "control items in leg {0}".format(num))

            # make sure that the control is consistent with the limitations set by
            # Payette
            if re.search(r"5", control):
                # deformation gradient control check
                if re.search(r"[^5]", control):
                    raise InputParserError(
                        "Only components of deformation gradient "
                        "are allowed with deformation gradient "
                        "control in leg {0}, got '{1}'".format(num, control))

                # check for valid deformation
                defgrad = np.array([[cij[0], cij[1], cij[2]],
                                    [cij[3], cij[4], cij[5]],
                                    [cij[6], cij[7], cij[8]]])
                jac = np.linalg.det(defgrad)
                if jac <= 0:
                    raise InputParserError(
                        "Inadmissible deformation gradient in leg "
                        "{0} gave a Jacobian of {1:f}".format(num, jac))

                # convert defgrad to strain E with associated rotation given by
                # axis of rotation x and angle of rotation theta
                rot, lstretch = np.linalg.qr(defgrad)
                if np.max(np.abs(rot - np.eye(3))) > np.finfo(np.float).eps:
                    raise InputParserError(
                        "Rotation encountered in leg {0}. ".format(num) +
                        "rotations are not yet supported")

            elif re.search(r"8", control):
                # displacement control check

                # like deformation gradient control, if displacement is specified
                # for one, it must be for all
                if re.search(r"[^8]", control):
                    raise InputParserError(
                        "Only components of displacment are allowed with "
                        "displacment control in leg {0}, got '{1}'"
                        .format(num, control))

                # must specify all components
                elif len(cij) != 3:
                    raise InputParserError(
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
                        cij[j] = 1 / self._kappa * (stretch ** self._kappa - 1.)
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
                    raise InputParserError("1 + kappa * ev must be positive")

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
                raise InputParserError(
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
                        raise InputParserError(
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

            try: self.efac = efac_hold
            except NameError: pass
            try: self.sfac = sfac_hold
            except NameError: pass

            # append to legs
            self._legs.append([num, ltime, steps, control, np.array(cij)])

            # increment
            num += 1
            continue

        if stress_control:
            # stress and or stress rate is used to control this leg. For
            # these cases, kappa is set to 0. globally.
            if self._kappa != 0.:
                self._kappa = 0.
                sys.stdout.write(
                    "WARNING: stress control boundary conditions "
                    "only compatible with kappa=0. Kappa is being "
                    "reset to 0. from {0:f}\n".format(kappa))
                self.bcontrol("kappa", 0.)

        # check that time is monotonic in lcontrol
        time_0, time_f = 0., 0.
        for ileg, leg in enumerate(self._legs):
            if ileg == 0:
                # set the initial time
                time_0 = leg[1]
                continue

            time_f = leg[1]
            if time_f <= time_0:
                raise InputParserError(
                    "time must be monotonic from {0:d} to {1:d}"
                    .format(leg[0] - 1, leg[0]))

            time_0 = time_f

        if not ileg:
            raise InputParserError("Only one time step found.")

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
        for item in ttypes:
            ttype = re.search(r"(?i)\b{0}\b".format(item), header)
            if ttype:
                s, e = ttype.start(), ttype.end()
                ttype = header[s:e].lower()
                break
            continue
        if ttype is None:
            raise InputParserError(
                "time specifier '{0}' not found.  Choose from {1}"
                .format(ttype, ", ".join(ttypes)))
        header = re.sub(r"(?i)\b{0}\b".format(ttype), "", header).strip()

        # determine the deformation specifier
        cspec = re.search(r"(?i)\bfrom.*columns\b.*", header)
        if cspec is None:
            dtype = " ".join(header.split()).lower()
            dtype = re.sub(r"[\,]", "", dtype).strip().lower()
            if dtype not in dtypes():
                raise InputParserError(
                    "Requested bad control type {0}".format(dtype))
            C, N = dtypes(dtype)

            # use default column indexes
            cidxs = range(N + 1)

        else:
            s, e = cspec.start(), cspec.end()
            dtype = " ".join(header[:s].split())
            dtype = re.sub(r"[\,]", "", dtype).strip().lower()
            if dtype not in dtypes():
                raise InputParserError(
                    "Requested bad control type {0}".format(dtype))
            C, N = dtypes(dtype)

            cidxs = []
            cspec = re.sub(r"\s|(?i)\bfrom.*columns\b", "", header[s:e]).strip()
            cspec = re.sub(r"-", ":", cspec).split(",")
            for item in cspec:
                item = item.split(":")
                if len(item) == 1:
                    cidxs.append(int(item[0]) - 1)
                else:
                    for i in range(len(item) - 1):
                        start, stop = int(item[i]) - 1, int(item[i+1])
                        cidxs.extend(range(start, stop))
                        continue
                continue

        if len(cidxs) > N + 1:
            raise InputParserError("Too many columns specified")
        if len(cidxs) < N + 1:
            raise InputParserError("Too few columns specified")

        control = "{0}".format(C) * N

        return ttype, cidxs, control



def parse_options(lines):
    """Parse lines for options

    Parameters
    ----------
    lines : str

    Returns
    -------
    options : dict

    """
    options = {}
    known_options = (re.compile(r"\bwrite.*input\b", re.I|re.M),
                     re.compile(r"\bnowriteprops\b", re.I|re.M),
                     re.compile(r"\brestart\b", re.I|re.M),)
    for option in known_options:
        found = option.search(lines)
        if found:
            s, e = found.start(), found.end()
            key = re.sub(r"\s", "_", " ".join(lines[s:e].split())).upper()
            lines = (lines[:s] + lines[e:]).strip()
            options[key] = True
        continue

    for line in lines.split("\n"):
        line = re.sub(I_EQ, " ", line).split()
        if not line:
            continue
        if len(line) == 1:
            key, val = line[0].upper(), True
        else:
            key, val = "_".join(line[:-1]).upper(), line[-1]
        options[key] = val
        continue

    return options


def find_item_name(lines, item, pop=False):
    """Find the item name in lines

    Parameters
    ----------
    lines : str
        block of lines to search for item
    item : str
        item for which name is desired
    pop : bool, optional
        if True, return lines with item line popped off

    Returns
    -------
    name : str
        desired name
    lines : str [only if pop]
        lines with item line popped off

    """
    name = re.search(r"(?i)\b{0}\s.*".format(item), lines)
    if name:
        s, e = name.start(), name.end()
        name = re.sub(r"\s", "_",
                      re.sub(r"(?i)\b{0}\s".format(item),
                             "", lines[s:e].strip()))
        if pop:
            lines = (lines[:s] + lines[e:]).strip()

    if pop:
        return name, lines
    return name


def get_content(lines, pop=False):
    block = []
    rlines, content = [], []
    bexp = re.compile(r"\bbegin\s*", re.I|re.M)
    eexp = re.compile(r"\bend\s.*", re.I|re.M)
    for iline, line in enumerate(lines.split("\n")):
        if bexp.search(line):
            block.append(1)
        if eexp.search(line):
            block.pop()
            rlines.append(line)
            continue

        if not block:
            content.append(line)
            if pop:
                continue

        rlines.append(line)
        continue

    content = "\n".join([x for x in content if x])
    rlines = "\n".join(rlines)
    if pop:
        return content, lines
    return content


def parse_user_input(lines):
    """Find simulation and parameterization block in the user input

    Parameters
    ----------
    lines : str
        the user input

    Returns
    -------
    simulations : dict
       simulation_name:simulation input
    parameterizations : dict
       parameterization_name:parameterization input

    """

    # strip the input of comments and extra lines and preprocess
    lines = fill_in_inserts(lines)
    lines = preprocess(lines)
    lines = strip_cruft(lines)

    simulations = find_block("simulation", lines, findall=True)
    opt = re.compile(r"\bbegin\s*optimization\b.*", re.I|re.M)
    prm = re.compile(r"\bbegin\s*permutation\b.*", re.I|re.M)
    post = "\nend input"
    for name, content in simulations.items():
        check_incompatibilities(content)
        if opt.search(content):
            stype = "optimization"
        elif prm.search(content):
            stype = "permutation"
        else:
            stype = "simulation"
        preamble = "begin input\nname {0}\ntype {1}\n".format(name, stype)
        content = preamble + content.strip() + post
        simulations[name] = content
        continue

    parameterizations = find_block("parameterization", lines, findall=True)
    stype = "parameterization"
    for name, content in parameterizations.items():
        check_incompatibilities(content)
        preamble = "begin input\nname {0}\ntype {0}\n".format(name, stype)
        content = preamble + content.strip() + post
        simulations[name] = content
        continue

    return simulations.values()


def check_incompatibilities(lines):
    """Check the user input for any incompatible blocks

    Parameters
    ----------
    lines : str
        User input

    """
    incompatible_blocks = (("optimization", "permutation",),)
    for blocks in incompatible_blocks:
        incompatibilites = []
        for block in blocks:
            content = find_block(block, lines)
            if content is None:
                continue
            incompatibilites.append(1)
            continue
        if len(incompatibilites) > 1:
            raise InputParserError(
                "Blocks: '{0}' incompatible in same input"
                .format(", ".join(blocks)))
        continue
    return


def strip_cruft(lines):
    """Strip lines of blank lines and comments

    Parameters
    ----------
    lines : str
        user input

    Returns
    -------
    lines : str
        lines stripped of all comments and blank lines

    """
    if not isinstance(lines, (list, tuple)):
        lines = lines.split("\n")

    stripped = []
    cmnt = re.compile(r"[#$]", re.I|re.M)
    for line in lines:
        line = line.strip()
        if not line.split():
            continue
        comment = cmnt.search(line)
        if comment:
            line = line[:comment.start()]
        if line.split():
            stripped.append(line)
        continue
    return "\n".join(stripped)


def preprocess(lines, preprocessor=None):
    """Preprocess lines

    Parameters
    ----------
    lines : str
        user input
    preprocessor : str, optional
        if preprocessor is None, find the preprocessing block in lines
        else use the passed preprocessor block.

    Returns
    -------
    lines : str
        preprocessed user input

    Notes
    -----

    """
    if preprocessor is None:
        preprocessor = find_block("preprocessing", lines)

    if preprocessor is None:
        return lines

    # split the preprocessor into a list of (pattern, repl) pairs
    preprocessor = [x.split()
                    for x in re.sub(I_EQ, " ", preprocessor).split("\n") if x]

    for pat, repl in preprocessor:
        full = re.compile(r"{{.*{0:s}.*}}".format(pat), re.I|re.M)
        while True:
            found = full.search(lines)
            if not found:
                break
            bn, en = found.start(), found.end()
            npat = re.compile(re.escape(r"{0}".format(lines[bn:en])), re.I|re.M)
            repl = re.sub(pat, repl, lines[bn+1:en-1])
            if re.search("[\*+/\-]", repl):
                repl = "{0:12.6E}".format(eval(repl))
            lines = npat.sub(repl, lines)
            continue
        continue

    return lines


def find_nested_blocks(major, nested, lines, default=None):
    """Find the nested blocks in major block of lines

    Parameters
    ----------
    major : str
        name of major block
    nested : list
        list of names of blocks to find in major
    lines : str
        lines to look for blocks
    default : None, optional
        default value

    Returns
    -------
    blocks : list
        blocks[0] is the major block
        blocks[1:n] are the nested blocks (in order requested)

    """
    blocks = []
    blocks.append(find_block(major, lines))
    for name in nested:
        bexp = re.compile(r"\bbegin\s*{0}\b.*".format(name), re.I|re.M)
        eexp = re.compile(r"\bend\s*{0}\b.*".format(name), re.I|re.M)
        start = bexp.search(blocks[0])
        stop = eexp.search(blocks[0])
        if start and not stop:
            raise InputParserError("End of block {0} not found".format(name))
        if not start:
            blocks.append(default)
            continue

        s, e = start.end(), stop.start()
        blocks.append(blocks[0][start.end():stop.start()])
        blocks[0] = blocks[0][:start.start()] + blocks[0][stop.end():]
        continue
    return blocks


def find_block(name, lines, default=None, findall=False, named=False):
    """Find the input block of form
        begin block [name]
        ...
        end block

    Parameters
    ----------
    lines : str
    name : str
        the block name

    Returns
    -------
    bname : str
        the block name
    block : str
        the block of input
    """
    blocks = {}
    pat = r"\bbegin\s*{0}\b".format(name)
    fpat = pat + r".*"
    namexp = re.compile(pat, re.I)
    bexp = re.compile(fpat, re.I|re.M)
    eexp = re.compile(r"\bend\s*{0}\b.*".format(name), re.I|re.M)
    k = 0

    named = True if findall else named

    while True:
        # get the block
        start = bexp.search(lines)
        stop = eexp.search(lines)

        if findall and not start:
            return blocks

        if start and not stop:
            raise InputParserError("End of block '{0}' not found".format(name))

        if not start:
            bname, block = None, default

        else:
            if named:
                # block name is everything from "begin block" to end of line
                s, e = start.start(), start.end()
                bname = re.sub(r"\s", "_", namexp.sub("", lines[s:e]).strip())
                if not bname:
                    bname = "default_{0}".format(k)

            block = lines[start.end():stop.start()].strip()

        if not findall:
            if named:
                return bname, block
            return block

        k += 1
        lines = lines[:start.start()] + lines[stop.end():]
        blocks[bname] = block
        continue

    return blocks


def pop_block(name, lines):
    """Pop the input block from lines

    Parameters
    ----------
    name : str
        the block name
    lines : str

    Returns
    -------
    lines : str
        lines with name popped

    """
    bexp = re.compile(r"\bbegin\s*{0}\b.*".format(name), re.I|re.M)
    eexp = re.compile(r"\bend\s*{0}\b.*".format(name), re.I|re.M)
    bexp, eexp = bexp.search(lines), eexp.search(lines)
    if bexp and eexp:
        s, e = bexp.start(), eexp.end()
        lines = lines[:s].strip() + lines[e:]
    return lines


def fill_in_inserts(lines):
    """Look for 'insert' commands in lines and insert then contents in place

    Parameters
    ----------
    lines : str
        User input

    Returns
    -------
    lines : str
        User input, modified in place, with inserts inserted

    """
    pat = r"^.*\binsert\b\s"
    namexp = re.compile(pat, re.I)
    fpat = pat + r".*"
    regexp = re.compile(fpat, re.I|re.M)
    while True:
        lines = strip_cruft(lines)
        found = regexp.search(lines)
        if not found:
            break

        # insert command found, find name
        s, e = found.start(), found.end()
        name = namexp.sub("", lines[s:e])
        insert = find_block(name, lines)
        if insert is None:
            fpath = os.path.realpath(os.path.expanduser(name))
            try:
                insert = open(fpath, "r").read()
            except IOError:
                raise InputParserError(
                    "Cannot find insert: {0}".format(repr(name)))

        # substitute the contents of the insert
        lines = regexp.sub(insert, lines)
        continue

    return lines


def parse_mathplot(mblock):
    """parse the mathplot block of the input file

    Parameters
    ----------
    mblock : str
        the mathplot block

    Returns
    -------
    mathplot : list
        list of mathplot variables

    """
    mathplot = []
    for item in mblock.split("\n"):
        mathplot.extend([x.upper() for x in re.sub(I_SEP, " ", item).split()])
        continue
    return sorted(list(set(mathplot)))


def parse_output(oblock):
    """parse the output block of the input file

    Parameters
    ----------
    oblock : str
        the output block

    Returns
    -------
    ovars : list
        list of output variables
    oformat : str
        output format

    """
    oformats = ("ascii", )
    ovars = []

    if not oblock:
        return ["ALL"], oformats[0]

    oformat, oblock = find_item_name(oblock, "format", pop=True)
    if oformat is None:
        oformat = "ascii"

    if oformat not in oformats:
        raise InputParserError(
            "Output format '{0}' not supported, choose from {1}"
            .format(oformat, ", ".join(oformats)))

    if re.search(r"(?i)\ball\b", oblock):
        ovars.append("ALL")

    else:
        for item in oblock.split("\n"):
            ovars.extend([x.upper() for x in re.sub(I_SEP, " ", item).split()])
            continue

    specials = {
        "stress": ["SIG11", "SIG22", "SIG33", "SIG12", "SIG23", "SIG13"],
        "strain": ["EPS11", "EPS22", "EPS33", "EPS12", "EPS23", "EPS13"],
        "efield": ["EFIELD1", "EFIELD2", "EFIELD3"],}

    for idx, ovar in enumerate(ovars):
        try:
            ovars[idx] = specials[ovar.lower()]
        except KeyError:
            pass
    ovars = sorted(list(set(flatten(ovars))))

    if "TIME" not in ovars:
        ovars.insert(0, "TIME")

    elif ovars.index("TIME") != 0:
        ovars.remove("TIME")
        ovars.insert(0, "TIME")

    return ovars, oformat


def parse_extraction(eblock):
    """Parse the extraction block of the input file

    Parameters
    ----------
    eblock : str
        The extraction block

    """
    extraction_vars = []
    for items in eblock.split("\n"):
        items = re.sub(I_SEP, " ", items).split()
        for item in items:
            if not re.search(r"^[%@]", item) and not item[0].isdigit():
                pu.log_warning(
                    "unrecognized extraction request {0}".format(item))
                continue
            extraction_vars.append(item)
        continue
    return [x.upper() for x in extraction_vars]


def flatten(x):
    result = []
    for el in x:
        if isinstance(el, list): result.extend(flatten(el))
        else: result.append(el)
    return result
