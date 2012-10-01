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

import re, sys, os
from textwrap import fill as textfill
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
    if isinstance(lines, (list, tuple)):
        lines = "\n".join(lines)
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
    return re.sub(r"\n\s*\n*", "\n", re.sub(r"[#$].*","", lines)) + "\n"


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
