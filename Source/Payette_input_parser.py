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

# --- module level regular expressions
I_EQ = r"[:,=]"
I_SEP = r"[:,;]"
SP = " "
RSEP = r":"
# regex to find contents of block
R_B = r"(?is)\bbegin\s*{0}\W*?(?P<c>.*?)\bend\s*{0}\W?"
R_Bn = r"(?is)\bbegin\s*{0}\s*{1}\W*?(?P<c>.*?)\bend\s*{0}\W?"
# regex to find a line of the form: begin <section> [name]
R_S = r"(?i)\bbegin\s+(?P<s>\w+).*\n?"
R_Sn = r"(?i)(?<=\bbegin)\s*(?P<s>{0})(?P<n>.*\n?)"
#R_S = r"(?i)(?<=\bbegin)\s*(?P<N>{0})\s*(?P<n>[a-z0-9_\-\. ]*)\W+"



RAND = np.random.RandomState(17)


class InputParser(object):
    """Payette user input class

    Reads and sets up blocks from user input


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
            pu.report_and_raise_error("No user input sent to InputParser")

        # --- required information ------------------------------------------ #
        self.inp = find_block("input", ilines, co=True)
        if self.inp is None:
            pu.report_and_raise_error("User input not found")

        # find the contents of the input block, popping found content along
        # the way
        content = get_content(self.inp)
        self.name, content = find_item_name(content, "name", pop=True)
        if self.name is None:
            pu.report_and_raise_error("Simulation name not found")
        self.stype, content = find_item_name(content, "type", pop=True)
        self._options = parse_options(content)
        pass

    def find_block(self, name, default=None, co=False):
        """Class method to the public find_block method """
        return find_block(name, self.inp, default=default, co=co)

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

    def formatted_input(self, ii="", sp="  "):
        ui = "begin simulation {0}\n{1}\nend simulation".format(
            self.name,
            re.sub(r"(?i)\W*simdir.*\n|\W*write\s*input.*\n"
                   "|\W*name\s.*\n|\W*type\s.*\n", "", self.inp))
        _ui = ""
        level = 0
        for line in ui.split("\n"):
            if not line.split():
                continue
            line = " ".join(line.split())
            if re.search(r"\bend\s*[a-z0-9_]+\W*?", line):
                level -= 1
            _ui += "{0}{1}{2}\n".format(ii, sp * level, line)
            if re.search(r"\bbegin\s*[a-z0-9_]+\s*[a-z0-9\-\.]*\W*?", line):
                level += 1
            continue
        return _ui

    def write_input_file(self, fpath):
        """write the input to a formatted file"""
        if os.path.isfile(fpath):
            fnam, fext = os.path.splitext(fpath)
            shutil.copyfile(fpath, fnam + ".orig" + fext)
        # write the file
        lines = self.formatted_input()
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
    def _get_opt(x):
        bool_map = {'false': False, 'true': True, 'on': True, 'off': False}
        try:
            return int(x)
        except ValueError:
            return bool_map.get(x.lower())
    for line in lines.split("\n"):
        line = [x for x in re.split(r"[:,= ]", line) if x]
        if not line:
            continue
        val = _get_opt(line[-1])
        if val is None:
            key, val = "_".join(line), True
        else:
            key, val = "_".join(line[:-1]), bool(val)
        options[key] = val
        continue
    if options:
        opts = "\n".join("  {0} = {1}".format(k, v) for k, v in options.items())
        blk = "begin control\n{0}\nend control\n...".format(opts)
        pu.log_warning("Simulations options should be moved to the root level "
                       "'control' input block as:\n{0}\n"
                       .format(blk), fmt=False)
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
    regex = r"(?i)\b{0}(\W*)(?P<w0>[a-z0-9\-\._]+)(\W*?)".format(item)
    match = re.search(regex, lines)
    if not match:
        if pop:
            return None, lines
        return None

    name = match.group("w0")
    if pop:
        return name, re.sub(match.group(), "", lines).strip()
    return name


def get_content(lines):
    """Return everything in lines that is not between
    begin keyword
      ...
    end keyword

    sections

    """
    content = lines
    regex = r"(?i)(?<=\bbegin).*[a-z0-9_\-\.]+.*"
    regex_1 = r"(?is)\bbegin\s*{0}.*\bend\s*{0}.*?"
    sections = re.findall(regex, content)
    for section in sections:
        section = re.search(regex_1.format(section.strip()), content)
        if section:
            content = content.replace(section.group(), "")
    return "\n".join(x.strip() for x in content.split("\n") if x.split())


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

    allsims = find_block("simulation", lines, findall=True, co=True)

    post = "\nend input"
    simulations = []
    for name, content in allsims:
        check_incompatibilities(content)
        for item in ("optimization", "permutation", ):
            if find_block(item, content):
                stype = item
                break
        else:
            stype = "simulation"
        preamble = "begin input\nname {0}\ntype {1}\n".format(name, stype)
        content = preamble + content.strip() + post
        simulations.append(content)
        continue

    allparams = find_block("parameterization", lines, findall=True, co=True)
    stype = "parameterization"
    for name, content in allparams:
        check_incompatibilities(content)
        preamble = "begin input\nname {0}\ntype {1}\n".format(name, stype)
        content = preamble + content.strip() + post
        simulations.append(content)
        continue

    return simulations


def check_incompatibilities(lines):
    """Check the user input for any incompatible blocks

    Parameters
    ----------
    lines : str
        User input

    """
    if lines is None:
        return
    incompatible_blocks = (("optimization", "permutation",),)
    for blocks in incompatible_blocks:
        present = 0
        for block in blocks:
            if re.search(r"(?is)\bbegin\W*{0}.*?".format(block), lines):
                present += 1
        if present > 1:
            pu.report_and_raise_error(
                "Blocks: '{0}' incompatible in same input"
                .format(", ".join(blocks)))
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
    # safe values to be used in eval
    GDICT = {"__builtins__": None}
    SAFE = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "asin": math.asin, "acos": math.acos,
            "atan": math.atan, "atan2": math.atan2, "pi": math.pi,
            "log": math.log, "exp": math.exp, "floor": math.floor,
            "ceil": math.ceil, "abs": math.fabs, "random": RAND.random_sample, }

    P = find_block("permutation", lines, co=True)
    if P is not None:
        P = re.findall(r"(?i)\bpermutate\s*(?P<N>[a-z0-9_\-]+)\W*?", P)
        P = r"(?i)\{.*[" + r"".join(r"({0})".format(x) for x in P) + "].*?\}"

    # some private functions
    def _split(s):
        _strip = lambda v: [x.strip() for x in v]
        return [_strip(re.split(I_EQ, x, 1)) for x in s.split("\n") if x]


    def _make_subs(inp, subs):
        # make substitutions
        nsubs, R = 0, r"{{.*\b{0}\b.*}}"
        for pat, repl in subs:
            repl = "(" + re.sub(r"[\{\}]", " ", repl).strip() + ")"
            matches = re.findall(R.format(pat.strip()), inp)
            for match in matches:
                mrepl = re.sub(r"\b{0}\b".format(pat), repl, match)
                inp, _n = re.subn(re.escape(match), mrepl, inp)
                nsubs += _n
                continue
            continue
        return inp, nsubs


    def _eval_subs(inp):
        # evaluate everything in { }
        regex = r"(?i){.*?}"
        for line in inp.split("\n"):
            matches = re.findall(regex, line)
            for match in matches:
                repl = re.sub(r"[\{\}]", " ", match)
                try:
                    repl = "{0:12.6E}".format(eval(repl, GDICT, SAFE))
                except NameError:
                    # Decide if this is due to a permutation block
                    if P and re.search(P, line):
                        continue
                    pu.report_error(
                        "Don't know what to do with '{0}'".format(line))
                    continue
                nl = re.sub(re.escape(match), repl, line)
                inp = re.sub(re.escape(line), nl, inp)
                line = nl
            continue
        if pu.error_count():
            pu.report_and_raise_error("Stopping due to previous errors")
        return inp

    if preprocessor is None:
        preprocessor = find_block("preprocessing", lines, co=True)

    if preprocessor is None:
        return lines

    # pop the preprocessing block
    _lines = lines
    lines = pop_block("preprocessing", lines)

    # preprocess the preprocessor and split it into "pat = repl" pairs
    I = 0
    while I < 25:
        preprocessor, nsubs = _make_subs(preprocessor, _split(preprocessor))
        if nsubs == 0:
            break
    else:
        pu.report_and_raise_error(
            "Exceeded maximum levels of nesting in preprocessing block")
    preprocessor = _eval_subs(preprocessor)
    preprocessor = _split(preprocessor)

    # Add the preprocessor values into the SAFE dict
    for pat, repl in preprocessor:
        SAFE[pat] = eval(repl, GDICT, SAFE)

    # the regular expression that defines the preprocessing
    pregex = r"(?i){{.*\b{0:s}\b.*}}"

    for pat, repl in preprocessor:
        # Check that each preprocessor variable is used somewhere else in the
        # file. Issue a warning if not. It is done here outside of the
        # replacement loop to be sure that we can check all variables. Because
        # variables in the preprocessor block were put in the SAFE
        # it is possible for, say, var_2 to be replaced when replacing var_1
        # if the user specified something like
        #          param = {var_1 * var_2}
        # So, since we want to check that all preprocessed variables are used,
        # we do it here. Checking is important for permutate and optimization
        # jobs to make sure that all permutated and optimized variables are
        # used properly in the file before the job actually begins.
        if not re.search(pregex.format(pat), _lines):
            pu.log_warning(
                "Preprocessing key '{0}' not found in input".format(pat))
        continue
    del _lines

    # Print out preprocessed values for debugging
    if ro.DEBUG:
        pu.log_message("Preprocessor values:")
        name_len = max([len(x[0]) for x in preprocessor])
        for pat, repl in preprocessor:
            pu.log_message("    {0:<{1}s} {2}"
                           .format(pat + ':', name_len + 2, repl))

    # do the replacement
    lines, nsubs = _make_subs(lines, preprocessor)
    lines = _eval_subs(lines)
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
    major = find_block(major, lines, " ", co=True)
    minor = []
    for name in nested:
        block = find_block(name, major)
        if block is not None:
            # remove block from major
            major = major.replace(block, "")
            block = block_contents(name, block)
        minor.append(block)
        continue
    return [major] + minor


def block_contents(bname, block):
    """Given an input block, return only its contents

    """
    return re.search(R_B.format(bname), block).group("c").strip()


def find_block(name, lines, default=None, findall=False, named=False, co=False,
               _k=[0]):
    """Find the input block of form
        begin block [name]
        ...
        end block

    Parameters
    ----------
    lines : str
    name : str
        the block name
    co : bool
        If true, return only the contents of the block, stripping the
        begin <name> and end <name> tags

    Returns
    -------
    bname : str
        the block name
    block : str
        the block of input
    """
    # get all sections that match 'name'
    sections = re.findall(R_Sn.format(name), lines)
    if not sections:
        if named:
            bname = "default_{0}".format(_k[0])
            _k[0] += 1
            return (None, default)
        if findall:
            return []
        return default

    # loop over all of the sections and add their contents to 'blocks'
    blocks = []
    for section in sections:
        sname, bname = section
        _b = r"\W*".join("({0})".format(x.strip()) for x in bname.split())
        blk = re.search(R_Bn.format(sname, _b), lines)
        if not blk:
            pu.report_and_raise_error(
                "End of block {0} not found".format(sname))
        block = blk.group()
        if co:
            block = blk.group("c")
        blocks.append(("_".join(bname.split()), block))
        if not findall:
            break

    if named or findall:
        return blocks

    return blocks[0][1]


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
    block = find_block(name, lines)
    if block is not None:
        lines = re.sub(re.escape(block), "", lines).strip()
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
    regex = r"(?i).*\binsert\b\s*(?P<insert>[a-z0-9_\-\. ]+)\W*"
    blx = []
    while True:
        lines = strip_cruft(lines)
        insert = re.search(regex, lines)
        if insert is None:
            break
        name = insert.group("insert").strip()
        fill = find_block(name, lines, co=True)
        if fill is not None:
            blx.append(name)
        else:
            fpath = os.path.realpath(os.path.expanduser(name))
            try:
                fill = open(fpath, "r").read()
            except IOError:
                pu.report_and_raise_error(
                    "Cannot find insert: {0}".format(repr(name)))

        # substitute the contents of the insert
        lines = re.sub(re.escape(insert.group()), "\n" + fill + "\n", lines, 1)
        continue

    for blk in blx:
        lines = pop_block(blk, lines)

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
        pu.report_and_raise_error(
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
    extraction_vars = []
    opts = {"step": "1"}
    for opt in opts:
        pat = r"(?P<name>\b{0}\s).*(?P<val>[a-z0-9_]+)\W*".format(opt)
        found = re.search(pat, eblock)
        if found is not None:
            opts[opt] = re.sub(I_EQ, " ", found.group("val")).strip()
            eblock = re.sub(re.escape(found.group()), "", eblock)
        continue

    _strip = lambda v: [x.strip() for x in v if x]
    eblock = _strip(re.split(r"[\,\,\n]", eblock))
    prefix = {True: "%", False: "@"}
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
    option = r".*".join(re.split(r"[ _]", option))
    regex = r"(?i)(?P<opt>\b{0})\s+(?P<val>[a-z0-9_\- ]*)\W*".format(option)
    option = re.search(regex, lines)
    if option:
        option = re.sub(r"[^\S\n]+", "_", option.group("val"))
    else:
        option = default
    return option


def _mynewone(lines):
    _c = {}
    sections = re.findall(R_S, lines)
    for section in sections:
        contents = find_block(section, lines, co=True)
        for item in [x for x in sections if x != section]:
            tr = find_block(item, contents)
            if tr is not None:
                contents = contents.replace(tr, "")
            continue
        _c[section] = [x.strip() for x in contents.split("\n") if x.split()]
        continue
    print _c
    sys.exit()
