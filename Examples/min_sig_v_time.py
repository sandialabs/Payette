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
COMP = ["@sig11", "@sig22", "@sig33"]
NC = len(COMP)
XG = np.array(pe.extract([GOLD_F] + ["@time"] + COMP, silent=True))


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
    SIG11, SIG22, and SIG33 from the simulation output and the gold result is
    returned as the error.

    """

    out_f = args[0]

    # extract only what we want from the gold and output files
    xo = np.array(pe.extract([out_f] + ["@time"] + COMP, silent=True))

    # do the comparison
    anrmsd, armsd = np.empty(NC), np.empty(NC)
    for idx in range(1, NC + 1):
        rmsd, nrmsd = pu.compute_rms(XG[:, 0], XG[:, idx], xo[:, 0], xo[:, idx])
        anrmsd[idx-1] = nrmsd
        armsd[idx-1] = rmsd
        continue

    return np.amax(np.abs(anrmsd))
