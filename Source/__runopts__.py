# *************************************************************************** #
# This file contains common runtime options for Payette.  It is NOT           #
# intended to be edited directly.  The file attributes can be changed through #
# command line arguments to payette, or through simulation input file         #
# directives.                                                                 #
# *************************************************************************** #

import sys
from inspect import isfunction, ismodule

# --- runtime configurable options
DISP = 1
VERBOSITY = 1
SQA = False
DEBUG = False
STRICT = False
WRITERESTART = False
RESTART_TIME = 0.
NOWRITEPROPS = False
USE_TABLE = False
KEEP = False
WRITE_VANDD_TABLE = False
TESTRESTART = False
PROPORTIONAL = False
NPROC = 1
CHECK_SETUP = False
WRITE_INPUT = False
WARNING = "warn"
ERROR = "stop"
SIMDIR = None
SKIP_ALREADY_RUN = False
EMIT = None
CCHAR = ("#", "$")

# not set through command line
EFIELD_SIM = False

# number of simulation steps
ISTEP = 0
NSTEPS = 0

if __name__ != "__main__":
    import Source.__config__ as cfg
    MTLDB = cfg.MTLDB

def set_command_line_options(opts):
    """Set global Payette options based on options passed to payette

    Parameters
    ----------
    opts : instance
      Parsed options of an OptionParsing instance.

    Returns
    -------
    None

    Notes
    -----
    Througout Payette, there are several global variabes used to custom taylor
    each simulation. For example, additional SQA coding is run if the user
    specifies

    % payette input_file --sqa

    Other options can be found by executing

    % payette -h

    In payette, the user options are passed in to this function where there
    are made module attributes that can then be used throughout Payette by
    importing this module.

    """
    from Source.Payette_utils import log_message
    module = sys.modules[__name__]
    global_options = get_global_options()
    for gopt, val in global_options.items():
        _register_default_option(gopt, val)
        continue
    for key, val in opts.__dict__.items():
        key = key.upper()
        if key not in global_options:
            continue
        gval = global_options[key]
        if val != gval:
            log_message("Setting {0} = {1}".format(key, val))
        setattr(module, key, val)
        _register_default_option(key, val)
        continue
    return


def set_control_options(control):
    """Set global Payette options based on options passed to payette through
    the input files 'control' block.

    Parameters
    ----------
    control : list
      control options in (key, val) pairs

    Returns
    -------
    None

    Notes
    -----
    Througout Payette, there are several global variabes used to custom taylor
    each simulation. For example, additional SQA coding is run if the user
    specifies

    begin control
      sqa
    end control

    In payette, the user options are passed in to this function where there
    are made module attributes that can then be used throughout Payette by
    importing this module.

    """
    from Source.Payette_utils import log_message
    module = sys.modules[__name__]
    global_options = get_global_options()
    for key, val in control:
        key = key.upper()
        try:
            gval = global_options[key]
            if val != gval:
                log_message("Setting {0} = {1}".format(key, val))
        except KeyError:
            pass
        setattr(module, key, val)
        _register_default_option(key, val)
        continue
    return


def set_global_option(attr, val, default=False):
    """Set/create global Payette options

    Parameters
    ----------
    attr : string
      attribute name to be set
    val :
      attribute value

    Returns
    -------
    None

    """
    module = sys.modules[__name__]
    attr = attr.upper()
    if default:
        _register_default_option(attr, val)
    setattr(module, attr, val)
    return


def restore_default_options():
    """Restore the default global options.

    If an input file has several simulations in it, each can specify its own
    options. At the end of each simulation, options are restored to the
    default values so that the options of one simulation do not pass to
    another. Global options are restored based on the command line options.

    """
    module = sys.modules[__name__]

    # get defaults
    defaults = get_default_options()
    for key, val in defaults.items():
        setattr(module, key, val)
        continue
    set_number_of_steps(reset=True)

    return


def _register_default_option(opt, val, inquire=False, options={}):
    if inquire:
        return options
    options[opt] = val
    return


def get_default_options():
    return _register_default_option(None, None, inquire=True)


def set_number_of_steps(N=0, I=0, done=[0], reset=False):
    if reset:
        done[0] = 0
    elif done[0]:
        raise OSError("Number of steps already set")
    else:
        done[0] = 1
    sys.modules[__name__].NSTEPS = N
    sys.modules[__name__].ISTEP = I
    return


def get_global_options():
    global_options = {}
    for k, v in sys.modules[__name__].__dict__.items():
        if k.startswith("_") or isfunction(v) or ismodule(v):
            continue
        global_options[k] = v
    return global_options


if __name__ == "__main__":
    for k, v in get_global_options().items():
        print k, v
