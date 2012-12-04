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

import uuid

from enthought.traits.api import HasStrictTraits, String, List, File, Date, Time, Bool, Enum, Instance, Directory, Tuple

from Viz_ModelData import PayetteModel
from Viz_ModelPlot import Viz_ModelPlot


class VizMetaData(HasStrictTraits):
    name = String
    base_directory = Directory
    index_file = File
    out_file = File
    log_file = File
    surface_file = File
    data_type = Enum('Simulation', 'Optimization', 'Visualization', 'Imported')
    model_type = String
    path_files = List(Tuple(String, File))
    created_date = Date
    created_time = Time
    object_id = String
    session_id = String
    successful = Bool(False)
    model = Instance(PayetteModel)
    plot = Instance(Viz_ModelPlot)

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)

        if len(self.object_id) < 1:
            self.object_id = str(uuid.uuid1())
