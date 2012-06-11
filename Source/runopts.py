# *************************************************************************** #
# This file contains common runtime options for runPayette.  It IS NOT        #
# intended to be edited directly.  The file attributes can be changed through #
# command line arguments to runPayette, or through simulation input file      #
# directives.                                                                 #
# *************************************************************************** #

import sys

# --- runtime configurable options
DISP = False
VERBOSITY = 3
SQA = False
DEBUG = False
STRICT = False
WRITERESTART = False
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
TESTERROR = False

# not set through command line
EFIELD_SIM = False


def set_command_line_options(opts):
    """Set global Payette options based on options passed to runPayette

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

    % runPayette input_file --sqa

    Other options can be found by executing

    % runPayette -h

    In runPayette, the user options are passed in to this function where there
    are made module attributes that can then be used throughout Payette by
    importing this module.

    """
    module = sys.modules[__name__]
    global_options = [x for x in dir(module) if x.isupper()]

    for gopt in global_options:
        val = getattr(module, gopt)
        _register_default_option(gopt, val)
        continue

    for opt in dir(opts):
        if opt.upper() in dir(module):
            gopt, val = opt.upper(), getattr(opts, opt)
            setattr(module, gopt, val)
            _register_default_option(gopt, val)
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

    return


def _register_default_option(opt, val, inquire=False, options={}):
    if inquire:
        return options
    options[opt] = val
    return


def get_default_options():
    return _register_default_option(None, None, inquire=True)
