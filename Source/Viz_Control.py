from enthought.traits.api import (HasStrictTraits, String, Int, List,
                                  Property, Button, Instance, File,
                                  Enum, implements, cached_property)
from enthought.traits.ui.api import (View, HGroup, VGroup, Item,
                                     UItem, ListEditor, TabularEditor, Label)
from enthought.traits.ui.tabular_adapter import TabularAdapter
from enthought.traits.ui.menu import OKButton, CancelButton

import uuid
import os
import time
import sys
import datetime
import pickle
import re

from Viz_MetaData import VizMetaData
from Viz_ModelRunner import IModelRunnerCallbacks
from Viz_ModelSelector import PayetteMaterialModelSelector
from Viz_ModelPlot import create_Viz_ModelPlot
from Viz_Search import filter_metadata
from Viz_DataImport import DataImportDialog

class MetadataTabularAdapter(TabularAdapter):
    columns = [('Name', 'name'), ('Model Type', 'model_type'), ('Data Type', 'data_type'), ('Created', 'created_timestamp')]
    created_timestamp_content = Property
    bg_color = Property
    text_color = Property

    def _get_created_timestamp_content(self):
        metadata = self.object.files[self.row]
        date = metadata.created_date
        time = metadata.created_time
        return "%04d-%02d-%02d %d:%02d:%02d" % (date.year, date.month, date.day, time.hour, time.minute, time.second)

    def _get_bg_color(self):
        metadata = self.object.files[self.row]
        if metadata.session_id == self.object.session_id:
            if metadata.successful:
                return "#dddddd"
            return "#ff7777"

        if metadata.successful:
            return None
        return "#ff3333"

    def _get_text_color(self):
        metadata = self.object.files[self.row]
        if metadata.successful:
            return None
        return "yellow"

class ModelTypeDialog(HasStrictTraits):
    model_type = Enum('Solid Mechanics', 'Equation of State')

    trait_view = View(
        Label('Choose a Model Type'),
        UItem('model_type'),
        buttons=[CancelButton, OKButton],
        kind='modal',
        title='Choose a Model Type'
    )

class ControlWindow(HasStrictTraits):
    implements(IModelRunnerCallbacks)

    search_string = String
    files = List(VizMetaData)
    filtered_files = Property(depends_on=['files','search_string'])
    selected_metadata = Instance(VizMetaData)
    session_id = String
    create_button = Button("Create Simulation")
    rerun_button = Button("Edit/Re-run Selected")
    optimize_button = Button("Perform Optimization")
    visualize_button = Button("Visualize Data")
    import_button = Button("Import Data")
    
    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)
        self.session_id = str(uuid.uuid1())

        files = []
        for filepath in self.FindMetadata('.'):
            with open(filepath, 'r') as f:
                metadata = pickle.load(f)
            base_dir = os.path.split(filepath)[0]
            metadata.base_directory = base_dir
            files.append(metadata)

        self.files = files


    def FindMetadata(self, path):
        all_files = os.listdir(path)
        files = []
        for f in all_files:
            filepath = os.path.join(path,f)
            base, ext = os.path.splitext(filepath)
            if os.path.isdir(filepath):
                files2 = self.FindMetadata(filepath)
                files.extend(files2)
                continue

            if 'vizmetadata' in filepath:
                files.append(filepath)

        return files

    def _get_filtered_files(self):
        return filter_metadata(self.files, self.search_string)

    def _create_button_fired(self):
        dialog = ModelTypeDialog()
        if not dialog.configure_traits():
            return

        if dialog.model_type == 'Solid Mechanics':
            mt = 'Mechanical'
        else:
            mt = 'eos'
        selector = PayetteMaterialModelSelector(model_type=mt,callbacks=self)
        selector.configure_traits()

    def _rerun_button_fired(self):
        metadata = self.selected_metadata

        selector = PayetteMaterialModelSelector(models=[metadata.model],
                                                selected_model=metadata.model,
                                                simulation_name=self.GenerateRerunName(metadata.name),
                                                auto_generated=False,
                                                callbacks=self,
                                                rerun=True
                                               )
        selector.configure_traits()

    def _optimize_button_fired(self):
        pass

    def _visualize_button_fired(self):
        metadata = self.selected_metadata
        args = {}
        if len(metadata.index_file) > 0:
            args['index file'] = os.path.join(metadata.base_directory, metadata.index_file)
        elif len(metadata.out_file) > 0:
            args['output file'] = os.path.join(metadata.base_directory, metadata.out_file)

        create_Viz_ModelPlot(metadata.name, **args)

    def _optimize_button_fired(self):
        pass

    def GenerateRerunName(self, name):
        match = re.search('(.*\.rerun)([0-9]+)$', name)
        if match is not None:
            return match.group(1) + str(int(match.group(2)) + 1)
        return name + '.rerun1'

    def _import_button_fired(self):
        selector = DataImportDialog(destination='.')
        if not selector.configure_traits():
            return

        metadata = selector.ImportSelectedData()
        if metadata is not None:
            with open(os.path.join(metadata.base_directory, metadata.name + '.vizmetadata'), 'w') as f:
                pickle.dump(metadata, f)
            self.files.append(metadata)


    # IModelRunnerCallbacks implementation
    def RunFinished(self, metadata):
        metadata.session_id = self.session_id
        with open(os.path.join(metadata.base_directory, metadata.name + '.vizmetadata'), 'w') as f:
            pickle.dump(metadata, f)
        self.files.append(metadata)

    trait_view = View(
        HGroup(
            VGroup(
                Item('search_string', label='Search'),
                UItem("filtered_files", editor=TabularEditor(adapter=MetadataTabularAdapter(),
                                                   selected='selected_metadata',
                                                   editable=False),
                     ),
            ),
            VGroup(
                UItem('create_button'),
                UItem('rerun_button', enabled_when="selected_metadata is not None and selected_metadata.data_type == 'Simulation'"),
                UItem('optimize_button', enabled_when="selected_metadata is not None and selected_metadata.data_type == 'Imported'"),
                UItem('visualize_button', enabled_when="selected_metadata is not None"),
                UItem('import_button'),
                padding=5
            )
        ),
        width=800,
        height=600,
        resizable=True
    )


window = ControlWindow()
window.configure_traits()
