import os, sys
import numpy as np
import Source.Payette_utils as pu
import Source.Payette_extract as pe

# Do operations on the gold file at the module level so they are only done once
DIR = os.path.dirname(os.path.realpath(__file__))
GOLD_F = os.path.join(DIR, "exmpls.gold")
if not os.path.isfile(GOLD_F):
    pu.report_and_raise_error("{0} not found".format(GOLD_F))

# extract only what we want from the gold and output files
COMP = ["@strain11", "@sig11"]
NC = len(COMP)
XG = np.array(pe.extract([GOLD_F] + COMP, silent=True))

# find the Young's modulus
EG = []
EPS, SIG = XG[:, 0], XG[:, 1]
for IDX in range(len(SIG) - 1):
    DEPS = EPS[IDX + 1] - EPS[IDX]
    DSIG = SIG[IDX + 1] - SIG[IDX]
    if abs(DEPS) > 1.e-16:
        EG.append(DSIG / DEPS)
    continue
EG = np.mean(np.array(EG))


def obj_fn(*args):
    """Evaluates the error between the simulation output and the "gold" answer

    Parameters
    ----------
    args : tuple
        args[0] : output file from simulation

    Returns
    -------
    error : float
        The error between the output and the "gold" answer

    Notes
    -----
    With this objective function, the maximum root mean squared error between
    the Young's modulus computed from the gold file and the output file is
    returned as the error.

    """

    out_f = args[0]

    # extract only what we want from the gold and output files
    xo = np.array(pe.extract([out_f] + COMP, silent=True))

    # do the comparison
    Eo = []
    for idx in range(len(xo[:, 0]) - 1):
        deps = xo[:, 0][idx + 1] - xo[:, 0][idx]
        dsig = xo[:, 1][idx + 1] - xo[:, 1][idx]
        if abs(deps) > 1.e-16:
            Eo.append(dsig / deps)
        continue
    Em = np.mean(np.array(Eo))
    error = np.abs((EG - Em) / EG)
    return error
