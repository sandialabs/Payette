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

import os
import numpy
import linecache
import pickle
import datetime

from enthought.traits.api import (HasStrictTraits, File, Directory, Enum,
                                  String, Bool, List, Array)
from enthought.traits.ui.api import (View, Group, VGroup, HGroup, Label,
                                     Item, UItem)
from enthought.traits.ui.menu import OKButton, CancelButton

from Viz_MetaData import VizMetaData
from Viz_HSCTHReader import read_hscth_file

class DataImportDialog(HasStrictTraits):
    destination = Directory        
    import_file = File
    file_type = Enum('Whitespace Delimited',
                     'Comma Separated Values',
                     'Tab Delimited',
                     'Custom Delimiter',
                     'HSCTH'
                    )
    delimiter = String('<WS>')
    load_header = Bool(True)

    def _import_file_changed(self, value):
        lower = value.lower()
        if lower.endswith('.csv'):
            self.file_type = 'Comma Separated Values'
        elif lower.endswith('.hscth'):
            self.file_type = 'HSCTH'

    def _file_type_changed(self, value):
        if value == 'Whitespace Delimited':
            self.delimiter = '<WS>'
        elif value == 'Comma Separated Values':
            self.delimiter = ','
        elif value == 'Tab Delimited':
            self.delimiter = '<TAB>'
        else:
            self.delimiter = ''

    def ImportSelectedData(self):
        if self.file_type == 'HSCTH':
            return self.ImportHSCTH()

        return self.ImportTextDelimited()

    def ImportTextDelimited(self):
        delim = self.delimiter
        if self.file_type == 'Whitespace Delimited':
            delim = None
        elif self.file_type == 'Comma Separated Values':
            delim = ','
        elif self.file_type == 'Tab Delimited':
            delim = '\t'

        if self.load_header:
            columns = linecache.getline(self.import_file, 1).split(delim)

            data = numpy.loadtxt(self.import_file, skiprows=1, delimiter=delim)
        else:
            data = numpy.loadtxt(self.import_file, delimiter=delim)
            columns = []
            for i in range(len(data[0])):
                columns.append("f_%d" % (i))

        base, name = os.path.split(self.import_file)
        new_name = name + '.imported'
        new_path = os.path.join(self.destination, new_name)

        with open(new_path, 'w') as out:
            out.write('\t'.join(columns))
            for r in data:
                row = []
                for c in r:
                    row.append(str(c))
                out.write('\t'.join(row) + '\n')

        now = datetime.datetime.now()

        metadata = VizMetaData(
            name = name,
            base_directory = self.destination,
            out_file = new_name,
            data_type = 'Imported',
            created_date = now.date(),
            created_time = now.time(),
            successful = True
        )
        return metadata

    def ImportHSCTH(self):
        gvars, tracers = read_hscth_file(self.import_file)

        base, name = os.path.split(self.import_file)

        # Material global vars
        new_name = name + '.gvars.imported'
        new_path = os.path.join(self.destination, new_name)

        with open(new_path, 'w') as out:
            out.write('\t'.join(gvars['varnames']) + '\n')
            for r in gvars['data']:
                row = []
                for c in r:
                    row.append(str(c))
                out.write('\t'.join(row) + '\n')

        tracer_files = []
        for tracer in tracers.itervalues():
            index = tracer['index']
            tracer_name = 'tracer%s' % (index)
            new_name = name + '.%s.imported' % (tracer_name)
            new_path = os.path.join(self.destination, new_name)

            with open(new_path, 'w') as out:
                out.write('\t'.join(tracer['varnames']) + '\n')
                for r in tracer['data']:
                    row = []
                    for c in r:
                        row.append(str(c))
                    out.write('\t'.join(row) + '\n')

            tracer_files.append((tracer_name, new_name))

        now = datetime.datetime.now()

        metadata = VizMetaData(
            name = name,
            base_directory = self.destination,
            out_file = new_name,
            path_files = tracer_files,
            data_type = 'Imported',
            created_date = now.date(),
            created_time = now.time(),
            successful = True
        )
        return metadata

    
    trait_view = View(
        VGroup(
            Item('import_file', label='File'),
            Item('file_type'),
            Item('delimiter', enabled_when='file_type == "Custom Delimiter"'),
            Item('load_header'),
            padding=2,
        ),
        kind='livemodal',
        buttons = [CancelButton, OKButton]
    )

if __name__ == '__main__':
    did = DataImportDialog(destination='.')
    if did.configure_traits():
        did.ImportSelectedData()
