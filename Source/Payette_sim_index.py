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

"""Contains classes and functions for writing index files for permutation and
optimization simulations

"""
import os
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

from Source.Payette_utils import who_is_calling
import Source.__runopts__ as ro


class SimulationIndexError(Exception):
    """SimulationIndex exception class"""
    def __init__(self, message):
        if not ro.DEBUG:
            sys.tracebacklimit = 0
        caller = who_is_calling()
        self.message = message + " [reported by {0}]".format(caller)
        super(SimulationIndexError, self).__init__(self.message)


class SimulationIndex(object):
    """Class for indexing simulatons run for permutation and optimization jobs

    """
    def __init__(self, base_dir=None, index_file=None):
        """Initialize the SimulationIndex object

        Parameters
        ----------
        name : str

          Path to directory, or file, where simulations are run, the index file
          will be dumped to this directory.

        mode : str
          Mode for file type.  Currently, only binary is supported.

        """

        if index_file is not None and base_dir is not None:
            raise SimulationIndexError(
                "Specify one of base_dir or index_file, not both")

        # check existence of base directory
        elif base_dir is not None:
            if not os.path.isdir(base_dir):
                raise SimulationIndexError("base_dir not found")
            else:
                # default index file name
                self._index_file = os.path.join(base_dir, "index.pkl")
        elif index_file is not None:
            if not os.path.isfile(index_file):
                raise SimulationIndexError("index_file not found")
            else:
                # default index file name
                self._index_file = index_file
        else:
            raise SimulationIndexError(
                "One of base_dir or index_file must be specified")

        # initialize class data
        self.index = {}
        self.loaded_index = {}

        # load the index file if it exists
        if os.path.isfile(self._index_file):
            self.load()

    def store(self, key, name, job_dir, variables, outfile):
        """Store all kwargs in to the index dict"""
        self.index[key] = {
            "name": name, "directory": job_dir, "variables": variables,
            "outfile": outfile}
        return

    def dump(self):
        """Dump self.index to a file"""
        # dup the index file
        with open(self._index_file, "wb") as fobj:
            pickle.dump(self.index, fobj)
        return

    def load(self):
        """Load the index file"""
        # check existence of file
        if not os.path.isfile(self._index_file):
            raise SimulationIndexError("index file {0} not found"
                                       .format(self._index_file))
        # load it in
        self.loaded_index = pickle.load(open(self._index_file, "rb"))
        return self.loaded_index

    def get_index(self):
        """Get the index file"""
        if not self.loaded_index:
            self.load()
        return self.loaded_index

    def output_files(self):
        return [v["outfile"] for k, v in self.loaded_index.items()]

    def index_file(self):
        return self._index_file
