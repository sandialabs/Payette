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
import pickle

from Source.Payette_utils import who_is_calling
import Source.runopts as ro


class MultiIndexError(Exception):
    """MultiIndex exception class"""
    def __init__(self, message):
        if not ro.DEBUG:
            sys.tracebacklimit = 0
        caller = who_is_calling()
        self.message = message + " [reported by {0}]".format(caller)
        super(MultiIndexError, self).__init__(self.message)


class MultiIndex(object):

    def __init__(self, base_dir, mode="binary"):

        if not os.path.isdir(base_dir):
            raise MultiIndexError("base_dir not found")

        if mode == "binary":
            self.write_mode = "wb"
            self.read_mode = "rb"

        else:
            raise MultiIndexError("unrecognized mode")

        self.index_file = os.path.join(base_dir, "index.pkl")

        # initialize class data
        self.index = {}
        self.index_read_from_file = {}

    def store_job_info(self, key, **kwargs):
        self.index[key] = kwargs
        return

    def write_index_file(self, key=None, name=None, directory=None,
                         variables=None):
        with open(self.index_file, self.write_mode) as fobj:
            pickle.dump(self.index, fobj)

    def read_index_file(self):
        # read in the index file
        if not os.path.isfile(self.index_file):
            raise MultiIndexError("index file {0} not found"
                                  .format(self.index_file))

        self.index_read_from_file = pickle.load(
            open(self.index_file, self.read_mode))

        return

    def get_index_dict(self):
        return self.index_read_from_file
