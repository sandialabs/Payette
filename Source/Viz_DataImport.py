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

class DataImportDialog(HasStrictTraits):
    destination = Directory        
    import_file = File
    file_type = Enum('Whitespace Delimited',
                     'Comma Separated Values',
                     'Tab Delimited',
                     'Custom Delimiter'
                    )
    delimiter = String('<WS>')
    load_header = Bool(True)

    def _import_file_changed(self, value):
        lower = value.lower()
        if lower.endswith('.csv'):
            self.file_type = 'Comma Separated Values'

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
            columns = None
            data = numpy.loadtxt(self.import_file, delimiter=delim)

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
