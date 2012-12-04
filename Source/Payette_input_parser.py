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

import re
import sys
import os
import shutil
import math
import numpy as np

import Source.__runopts__ as ro
import Payette_utils as pu

# --- module leve constants
I_EQ = r"[:,=]"
I_SEP = r"[:,;]"


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

    def get_option(self, option, default=None):
        return self._options.get(option, default)

    def get_options_as_string(self):
        return "\n".join(
            ["{0} = {1}".format(k, v) for k, v in self._options.items()])

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

    def write_input_file(self, fpath):
        """write the input to a formatted file"""
        if os.path.isfile(fpath):
            fnam, fext = os.path.splitext(fpath)
            shutil.copyfile(fpath, fnam + ".orig" + fext)

        # write the file
        ns = 2
        lines = "begin simulation {0}\n{1}\nend simulation".format(
            self.name,
            re.sub(r"(?i)simdir.*\n|write\s*input.*\n"
                   "|name\s.*\n|[begin,end] input.*\n"
                   "|\stype\s.*\n", "", self.inp))
        with open(fpath, "w") as fobj:
            fobj.write(lines)
        return


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
    known_options = (re.compile(r"\bwrite.*input\b", re.I | re.M),
                     re.compile(r"\bnowriteprops\b", re.I | re.M),
                     re.compile(r"\brestart\b", re.I | re.M),)
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
    bexp = re.compile(r"\bbegin\s*", re.I | re.M)
    eexp = re.compile(r"\bend\s.*", re.I | re.M)
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
    if isinstance(lines, (list, tuple)):
        lines = "\n".join(lines)
    lines = fill_in_inserts(lines)
    lines = preprocess(lines)
    lines = strip_cruft(lines)

    simulations = find_block("simulation", lines, findall=True)
    opt = re.compile(r"\bbegin\s*optimization\b.*", re.I | re.M)
    prm = re.compile(r"\bbegin\s*permutation\b.*", re.I | re.M)
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
        preamble = "begin input\nname {0}\ntype {1}\n".format(name, stype)
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
    # return re.sub(r"\n\s*\n*", "\n", re.sub(r"[#$].*","", lines)) + "\n"
    # strip comments
    lines = re.sub(r"[#$].*", "", lines)
    # strip blank lines
    lines = re.sub(r"\n\s*\n*", "\n", lines)
    # remove all consectutive spaces, i.e. A    string -> A string
    lines = re.sub(r"(?m)[^\S\n]+", " ", lines)
    return lines.strip() + "\n"


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
    If a preprocessor is given, the preprocessor is of form:

       PAT = REPL

    and then all occurrences of PAT will be replaced by REPL in the input
    file. For example,

       BMOD = 1.E+9

    might be used to replace all occurrences of {BMOD} with 1.E+9

    If no preprocessor block is sent in, it is looked for a "preprocessing"
    block in lines

    """
    import random
    safe_eval_dict = {
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
        'pi': math.pi, 'log': math.log, 'exp': math.exp, 'floor': math.floor,
        'ceil': math.ceil, 'abs': math.fabs, 'random': random.random, }

    if preprocessor is None:
        preprocessor = find_block("preprocessing", lines)

    if preprocessor is None:
        return lines

    # split the preprocessor into a list of (pattern, repl) pairs
    preprocessor = [x.split(None, 1)
                    for x in re.sub(I_EQ, " ", preprocessor).split("\n") if x]

    # Add the preprocessor values into the safe_eval_dict
    gdict, ldict, I = {"__builtins__": None}, safe_eval_dict, 0
    while True:
        errors = 0
        for idx, [pat, repl] in enumerate(preprocessor):
            tmp = repl.lstrip('{').rstrip('}')
            try:
                tmpval = eval(tmp, gdict, ldict)
            except NameError:
                errors += 1
                continue
            safe_eval_dict[pat] = tmpval
            preprocessor[idx][1] = "{0:12.6E}".format(tmpval)
            continue
        if not errors or I > 10:
            break
        I += 1
        continue

    if errors or I > 10:
        pu.report_and_raise_error("Unresolvable preprocessing block")

    for pat, repl in preprocessor:
        # Check that each preprocessor variable is used somewhere else in the
        # file. Issue a warning if not. It is done here outside of the
        # replacement loop to be sure that we can check all variables. Because
        # variables in the preprocessor block were put in the safe_eval_dict
        # it is possible for, say, var_2 to be replaced when replacing var_1
        # if the user specified something like
        #          param = {var_1 * var_2}
        # So, since we want to check that all preprocessed variables are used,
        # we do it here. Checking is important for permutate and optimization
        # jobs to make sure that all permutated and optimized variables are
        # used properly in the file before the job actually begins.
        if not re.search(r"(?i){{.*?\b{0:s}\b.*?}}".format(pat), lines):
            pu.log_warning(
                "Preprocessing key '{0}' not found in input".format(pat))
        continue

    for pat, repl in preprocessor:
        full = re.compile(r"{{.*?\b{0:s}\b.*?}}".format(pat), re.I | re.M)
        while True:
            found = full.search(lines)
            if not found:
                break
            bn, en = found.start(), found.end()
            npat = re.compile(
                re.escape(r"{0}".format(lines[bn:en])), re.I | re.M)
            # tmprepl is used because I (Scot) was having issues with
            # persistence of just 'repl' when it was used in multiple {} sets.
            tmprepl = re.sub(
                r"(?i){0}".format(pat), repl, lines[bn + 1:en - 1])
            try:
                tmprepl = "{0:12.6E}".format(eval(tmprepl, gdict, ldict))
            except:
                pu.report_and_raise_error(
                    "failure evaluating '{0}' in preprocessor.".format(tmprepl))

            lines = npat.sub(tmprepl, lines)
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
        bexp = re.compile(r"\bbegin\s*{0}\b.*".format(name), re.I | re.M)
        eexp = re.compile(r"\bend\s*{0}\b.*".format(name), re.I | re.M)
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
    bexp = re.compile(fpat, re.I | re.M)
    eexp = re.compile(r"\bend\s*{0}\b.*".format(name), re.I | re.M)
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
    bexp = re.compile(r"\bbegin\s*{0}\b.*".format(name), re.I | re.M)
    eexp = re.compile(r"\bend\s*{0}\b.*".format(name), re.I | re.M)
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
    regexp = re.compile(fpat, re.I | re.M)
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
        lines = regexp.sub(insert, lines, 1)
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
        "efield": ["EFIELD1", "EFIELD2", "EFIELD3"], }

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

    Notes
    -----
    The extraction block will look something like

        sig11, sig22, sig33, %1, %rootj2

    and we return
        ['@SIG11', '@SIG22', '@SIG33', '%1', '%ROOTJ2']
    """
    prefix = {True: "%", False: "@"}
    extraction_vars = []
    opts = {"step": "1"}
    for opt in opts:
        pat = r"\b{0}\s".format(opt)
        found = re.search(pat, eblock)
        if found is not None:
            s = found.end()
            e = re.search(pat + r".*\n", eblock).end()
            opts[opt] = re.sub(I_EQ, " ", eblock[s:e]).strip()
            eblock = eblock[:found.start()] + eblock[e:]
        continue

    eblock = re.sub(r"[\.\,\n]", " ", eblock).split()
    for item in eblock:
        if not re.search(r"^[%@]", item):
            # user did not specify a prefix to the extraction var, add it here
            item = prefix[item.isdigit()] + item
        extraction_vars.append(item.upper())
        continue
    return extraction_vars, opts


def flatten(x):
    result = []
    for el in x:
        if isinstance(el, list):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def get(option, lines, default=None):
    """ Find the option in lines
    """
    option = ".*".join(option.split())
    pat = r"(?i)\b{0}\s".format(option)
    fpat = pat + r".*"
    option = re.search(fpat, lines)
    if option:
        s, e = option.start(), option.end()
        option = lines[s:e]
        lines = (lines[:s] + lines[e:]).strip()
        option = re.sub(pat, "", option)
        option = re.sub(r"[\,=]", " ", option).strip()
    else:
        option = default
    return option
