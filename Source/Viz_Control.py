from enthought.traits.api import (HasStrictTraits, String, Int, List,
                                  Property, Button, Instance, File,
                                  Enum, Interface, implements,
                                  cached_property)
from enthought.traits.ui.api import (View, HGroup, VGroup, Item,
                                     UItem, ListEditor, TabularEditor, Label,
                                    Handler)
from enthought.traits.ui.tabular_adapter import TabularAdapter
from enthought.traits.ui.menu import OKButton, CancelButton
from enthought.traits.ui.message import message

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
from Viz_Utility import message_box
#from Viz_Optimizer import VizOptimizer

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

    traits_view = View(
        VGroup(
            Label('Choose a Model Type'),
            UItem('model_type'),
            padding=10,
        ),
        buttons=[CancelButton, OKButton],
        kind='modal',
        title='Choose a Model Type',
    )

class IControlWindow(Interface):
    """
    This class serves as an interface for use in the VisualizationHandler
    below. It doesn't define any functions, but mainly ensures that only
    the ControlWindow can be passed as a parameter.
    """
    pass

class VisualizationHandler(Handler):
    """
    Prompts the user to save the closed visualization if they desire.
    """

    control_window = Instance(IControlWindow)
    metadata = Instance(VizMetaData)
    original = Instance(VizMetaData)

    def close(self, info, is_ok):
        result = message_box('Do you wish to save these visualization parameters?', 'Close Visualization',
                             ['Cancel', None, 'Don\'t Save', 'Save'])
        if result == 'Cancel':
            return False

        if result == 'Save':
            self.metadata.successful = True
            self.metadata.plot = info.object
            self.control_window.files.append(self.metadata)

        return True

class ControlWindow(HasStrictTraits):
    implements(IControlWindow)
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
                                                simulation_name=self.GenerateName(metadata.name, 'rerun'),
                                                auto_generated=False,
                                                callbacks=self,
                                                rerun=True
                                               )
        selector.configure_traits()

    def _visualize_button_fired(self):
        metadata = self.selected_metadata
        previous = None
        if metadata.data_type == 'Visualization':
            previous = metadata.clone_traits()
            previous.object_id = str(uuid.uuid1())

        args = {}
        if len(metadata.index_file) > 0:
            args['index file'] = os.path.join(metadata.base_directory, metadata.index_file)
        elif len(metadata.out_file) > 0:
            args['output file'] = os.path.join(metadata.base_directory, metadata.out_file)

        now = datetime.datetime.now()

        metadata2 = VizMetaData(
            name = self.GenerateName(metadata.name, 'viz'),
            base_directory = metadata.base_directory,
            data_type = 'Visualization',
            model_type = metadata.model_type,
            created_date = now.date(),
            created_time = now.time(),
            session_id = self.session_id
        )

        create_Viz_ModelPlot(metadata.name, handler=VisualizationHandler(
            control_window=self, metadata=metadata2, original=metadata), metadata=previous, **args)

    def _optimize_button_fired(self):
        message_box('This functionality is not yet implemented', 'Not Implemented', ['OK'])
        #metadata = self.selected_metadata
        #gold_file = os.path.join(metadata.base_directory, metadata.out_file)
        #optimizer = VizOptimizer(gold_file=gold_file)
        #optimizer.configure_traits()

    def GenerateName(self, name, prefix):
        match = re.search('(.*\\.%s)([0-9]+)$' % (prefix), name)
        if match is not None:
            return match.group(1) + str(int(match.group(2)) + 1)
        return name + ('.%s1' % (prefix))

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
