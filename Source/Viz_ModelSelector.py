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

from enthought.traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant, Enum, Tuple, File, on_trait_change
from enthought.traits.ui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, InstanceEditor, ListEditor, Spring, Action, Handler
from enthought.traits.ui.tabular_adapter import TabularAdapter

import Payette_model_index as pmi
import Payette_xml_parser as px

from Viz_ModelData import PayetteModel, PayetteModelParameter, PayetteMaterial, PayetteMaterialParameter
from Viz_ModelRunner import ModelRunner, IModelRunnerCallbacks
import Viz_Utility as vu

class PayetteInputStringPreview(HasStrictTraits):
    class ISPHandler(Handler):
        def _run(self, info):
            preview = info.ui.context['object']
            preview.runner.RunInputString(preview.input_string, preview.model)

        def _close(self, info):
            info.ui.dispose()

    input_string = String
    runner = Instance(ModelRunner)
    model = Instance(PayetteModel)

    trait_view = View(
        VGroup(
            Item('input_string', style='custom', show_label=False),
        ),
        buttons=[Action(name='Close', action='_close'), Action(name='Run', action='_run')],
        handler=ISPHandler(),
        width=800,
        height=600,
        resizable=True
    )

class PayetteMaterialModelSelector(HasStrictTraits):
    model_type = Enum('Mechanical', 'eos', 'any')
    models = List(Instance(PayetteModel))
    selected_model = Instance(PayetteModel)
    simulation_name = String
    auto_generated = Bool(True)
    none_constant = Constant("None")
    show_button = Button("Show Input File")
    run_button = Button("Run Material Model")
    model_index = pmi.ModelIndex()
    rerun = Bool(False)
    supplied_data = List(Tuple(String, File))
    callbacks = Instance(IModelRunnerCallbacks)

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)
        if self.models is None or len(self.models) < 1:
            self.models = vu.loadModels()
            if len(self.models) > 0:
                self.selected_model = self.models[0]
            for model in self.models:
                model.supplied_data = self.supplied_data

    def _simulation_name_changed(self, info):
        self.auto_generated = False

    @on_trait_change('selected_model.selected_material')
    def update_sim_name(self):
        if self.auto_generated:
            if self.selected_model is not None and self.selected_model.selected_material is not None:
                self.simulation_name = self.selected_model.model_name + "_" + self.selected_model.selected_material.name
            # A trick to reset the flag, since _simulation_name_changed() is called first
            self.auto_generated = True

    def _run_button_fired(self, event):
        runner = ModelRunner(simulation_name=self.simulation_name, material_models=[self.selected_model],
                            callbacks=self.callbacks)
        runner.RunModels()

    def _show_button_fired(self, event):
        runner = ModelRunner(simulation_name=self.simulation_name, material_models=[self.selected_model],
                            callbacks=self.callbacks)
        input_string = runner.CreateModelInputString(self.selected_model)
        preview = PayetteInputStringPreview(input_string=input_string, runner=runner, model=self.selected_model)
        preview.configure_traits()


    traits_view = View(
        VGroup(
            HGroup(
                VGroup(
                    Label("Installed Models"),
                    UItem('models',
                        editor = TabularEditor(
                            show_titles = True,
                            editable = False,
                            selected = 'selected_model',
                            multi_select = False,
                            adapter = TabularAdapter(columns = [('Models', 'model_name')])
                        )
                    ),
                    VGroup(
                        Label("Available Materials"),
                        UItem('selected_model', label="Foo", editor = InstanceEditor(view='material_view'),
                              visible_when="selected_model is not None and len(selected_model.materials) > 0"),
                        Item("none_constant", style='readonly', show_label=False,
                              visible_when="selected_model is not None and len(selected_model.materials) < 1")
                    ),
                    visible_when='not rerun',
                    show_border = True
                ),
                VGroup(
                    Label("Material Parameters"),
                    UItem('selected_model', editor=InstanceEditor(view='param_view')),
                    show_border = True
                )
            ),
            VGroup(
                Label("Boundary Parameters"),
                UItem('selected_model',
                    editor = InstanceEditor(
                        view='boundary_legs_view'
                    ),
                    visible_when='selected_model is not None and "eos" not in selected_model.model_type'
                ),
                UItem('selected_model',
                    editor = InstanceEditor(
                        view='eos_boundary_view'
                    ),
                    visible_when='selected_model is not None and "eos" in selected_model.model_type'
                ),
                show_border = True,
            ),
            Item('simulation_name', style="simple"),
            HGroup(
                Spring(),
                Item('show_button', show_label = False, enabled_when="selected_model is not None"),
                Item('run_button', show_label = False, enabled_when="selected_model is not None"),
                show_border = True
            )
        ),
        style='custom',
        width=1024,
        height=768,
        resizable=True
    )

if __name__ == "__main__":
    pm = PayetteMaterialModelSelector(model_type='any', supplied_data = [('Foo', 'elastic_al_6061.out')])
    pm.configure_traits()

