# The MIT License

# Copyright (c) 2011 Tim Fuller

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
try:
    import cPickle as pickle
except ImportError:
    import pickle

from Source.Payette_utils import who_is_calling
import Source.runopts as ro


class SimulationIndexError(Exception):
    """SimulationIndex exception class"""
    def __init__(self, message):
        if not ro.DEBUG:
            sys.tracebacklimit = 0
        caller = who_is_calling()
        self.message = message + " [reported by {0}]".format(caller)
        super(SimulationIndexError, self).__init__(self.message)


class SimulationIndex(object):

    def __init__(self, base_dir):
        """Initialize the SimulationIndex object

        Parameters
        ----------
        base_dir : str
          Path to directory where simulations are run, the index file will be
          dumped to this directory.
        mode : str
          Mode for file type.  Currently, only binary is supported.

        """

        # check existence of base directory
        if not os.path.isdir(base_dir):
            raise SimulationIndexError("base_dir not found")

        # default index file name
        self.index_file = os.path.join(base_dir, "index.pkl")

        # initialize class data
        self.index = {}
        self.loaded_index = {}

        # load the index file if it exists
        if os.path.isfile(self.index_file):
            self.load()

    def store(self, key, **kwargs):
        """Store all kwargs in to the index dict"""
        self.index[key] = kwargs
        return

    def dump(self):
        """Dump self.index to a file"""
        # dup the index file
        with open(self.index_file, "wb") as fobj:
            pickle.dump(self.index, fobj)
        return

    def load(self):
        """Load the index file"""
        # check existence of file
        if not os.path.isfile(self.index_file):
            raise SimulationIndexError("index file {0} not found"
                                       .format(self.index_file))
        # load it in
        self.loaded_index = pickle.load(open(self.index_file, "rb"))
        return self.loaded_index

    def get_index(self):
        """Get the index file"""
        if not self.loaded_index:
            self.load()
        return self.loaded_index
