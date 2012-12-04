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

import numpy
import re

cth_mat_globals = ['MTOT', 'ETOT', 'EK', 'EINT', 'VTOT', 'XMOM', 'YMOM',
                   'ZMOM']


def get_comment(line):
    if line[0] == '%':
        return line
    else:
        return ''


def get_linesplit(line):
    vn = []
    for value in line.split(","):
        vn.append(value.strip())
    return vn


def check_candidate_consistency(patvars, patids):
    """Try to determine if the number of occurrences of the variables
    and the indices are consistent. Consistency requires two things. (1)
    The value for each key in patvars is the same, and is also the same
    as the number of patid keys. (2) The value for each key in patids is
    the same, and is also the same as the number of keys in patvars.

    Return True if these two conditions are met, False otherwise.

    The purpose of this function is to try to determine if there are the
    same number of patids for each patvar. If there are not, then it is
    possible that some of the patvars are cth mat_global variables and
    others are tracers. However, it is not a perfect test because if
    there are the same number of tracers and materials, the two
    conditions are still met.

    See docstring for find_candidates for a description of patvars and
    patids.
    """
    samecount = True
    if len(patids) > 0:
        count0 = patids[patids.keys()[0]]
        for id in patids.keys():
            if not patids[id] == count0:
                samecount = False

        if not count0 == len(patvars):
            samecount = False

    if len(patvars) > 0:
        count0 = patvars[patvars.keys()[0]]
        for id in patvars.keys():
            if not patvars[id] == count0:
                samecount = False

        if not count0 == len(patids):
            samecount = False

    return samecount


def sort_vars(varnames):
    """Sort variables in varnames to global variables and tracer
    variables. Collect the tracer variables by tracer. Treat material
    global variables as global variables.

    Check for the consistency of the names and indices.

    Return a list of global variables and a dictionary with tracer
    indices as the keys and lists of tracer variables names as the
    values
    """
    # pattern to find variable names endingin .dd, where dd is some
    # number of digits
    tracpat = r'(.*)\.(\d+)'
    # group(1) is the part without the .dd
    # group(2) is the dd, without the .
    # group(0) is the whole name

    globalvarnames = []  # These will be returned
    tracers = {}

    tracervarnames = []  # These are internal variables
    patvars = {}
    patids = {}

    for var in varnames:
        match = re.search(tracpat, var)
        if match:
            if match.group(1) in cth_mat_globals:
                globalvarnames.append(var)
            else:
                tracervarnames.append(var)
                # The following are just for checking consistency
                if match.group(1) in patvars.keys():
                    patvars[match.group(1)] += 1
                else:
                    patvars[match.group(1)] = 1
                if match.group(2) in patids.keys():
                    patids[match.group(2)] += 1
                else:
                    patids[match.group(2)] = 1
        else:
            globalvarnames.append(var)

    if check_candidate_consistency(patvars, patids):
        tracers['varbasenames'] = tracerbasenames = patvars.keys()
        tracers['indices'] = patids.keys()
#    for index in tracerindices:
#      tracers[index] = []
#      for basename in tracerbasenames:
#        vn = basename + '.' + index
#        tracers[index].append(vn)
    else:
        pass    # add handling of failed consistency check. Note that
                # tracevarnames is also not consistent.

    return globalvarnames, tracers


########################################################################
def read_hscth_file(filename):
    hscth_file = open(filename, "r")
    skips = 3

    # Expect one comment line, one varname line, and one label line
    # comment line
    line = hscth_file.readline()
    first_comment = get_comment(line)

    # variable name line
    line = hscth_file.readline()
    varnames = get_linesplit(line)

    # label line, including units
    line = hscth_file.readline()
    labels = get_linesplit(line)

    hscth_file.close()

    gvarnames, tvarD = sort_vars(varnames)

    gvars = {}
    gvars['varnames'] = []
    gvars['labels'] = []
    gvars['units'] = []
    col_list = []
    for name in gvarnames:
        idx = varnames.index(name)
        col_list.append(idx)
        gvars['varnames'].append(varnames[idx])
        gvars['labels'].append(labels[idx])
        # Alegra and CTH put units in the labels in different ways. For now,
        # just put the labels as is
        gvars['units'].append(labels[idx])
    gvars['data'] = numpy.loadtxt(filename, comments="%", skiprows=skips,
                                  delimiter=",", unpack=True, usecols=col_list)

    tracers = {}
    for tracidx in tvarD['indices']:
        trac = {}
        trac['index'] = tracidx
        trac['varnames'] = []
        trac['labels'] = []
        trac['units'] = []
        col_list = []
        for name in tvarD['varbasenames']:
            trac['varnames'].append(name)
            idx = varnames.index(
                name + '.' + tracidx)   # The column in the file
            col_list.append(idx)
            trac['labels'].append(labels[idx])
            # Alegra and CTH put units in the labels in different ways. For now,
            # just put the labels as is
            trac['units'].append(labels[idx])
        trac['data'] = numpy.loadtxt(filename, comments="%", skiprows=skips,
                                     delimiter=",", unpack=True, usecols=col_list)
        tracers[tracidx] = trac

    return (gvars, tracers)

if __name__ == '__main__':
    fn = "CuStPene.hscth"
    print read_hscth_file(fn)
