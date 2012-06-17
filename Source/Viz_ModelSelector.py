try:
    from traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant
    from traitsui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, InstanceEditor, ListEditor, Spring, Action, Handler
    from traitsui.tabular_adapter import TabularAdapter

except ImportError:
    # support for MacPorts install of enthought tools
    from enthought.traits.api import (
        HasStrictTraits, List, Instance, String,
        BaseInt, Int, Float, Bool, Property, Button, Constant)
    from enthought.traits.ui.api import (
        View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor,
        InstanceEditor, ListEditor, Spring, Action, Handler)
    from enthought.traits.ui.tabular_adapter import TabularAdapter

import Payette_model_index as pmi
import Payette_xml_parser as px

from Viz_ModelData import PayetteModel, PayetteModelParameter, PayetteMaterial, PayetteMaterialParameter
from Viz_ModelRunner import ModelRunner

class PayetteInputStringPreview(HasStrictTraits):
    class ISPHandler(Handler):
        def _run(self, info):
            preview = info.ui.context['object']
            preview.runner.RunInputString(preview.input_string)

        def _close(self, info):
            info.ui.dispose()

    input_string = String
    runner = Instance(ModelRunner)

    trait_view = View(
        VGroup(
            Item('input_string', style='custom', show_label=False),
        ),
        buttons=[Action(name='Close', action='_close'), Action(name='Running', action='_run')],
        handler=ISPHandler(),
        width=800,
        height=600,
        resizable=True
    )

class PayetteMaterialModelSelector(HasStrictTraits):
    models = List(Instance(PayetteModel))
    selected_model = Instance(PayetteModel)
    simulation_name = String
    auto_generated = Bool(True)
    none_constant = Constant("None")
    show_button = Button("Show Input File")
    run_button = Button("Run Material Model")
    model_index = pmi.ModelIndex()

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)
        self.loadModels()

    def loadModels(self):
        for modelName in self.model_index.constitutive_models():
            control_file = self.model_index.control_file(modelName)
            cmod = self.model_index.constitutive_model_object(modelName)
            cmod_obj = cmod(control_file)

            params = []
            for param in cmod_obj.get_parameter_names_and_values():
                params.append(
                    PayetteModelParameter(name = param[0], description = param[1], value = param[2])
                )

            model = PayetteModel(model_name = modelName, parameters = params, model_type = [cmod_obj.material_type])
            model.on_trait_change(self.update_sim_name, 'selected_material')
            self.models.append(model)

        self.selected_model = self.models[0]

    def _selected_model_changed(self, info):
        if info is None or len(info.materials) > 0:
            return

        info.materials = self.loadModelMaterials(info.model_name)

    def _simulation_name_changed(self, info):
        self.auto_generated = False

    def update_sim_name(self):
        if self.auto_generated:
            if self.selected_model is not None and self.selected_model.selected_material is not None:
                self.simulation_name = self.selected_model.model_name + "_" + self.selected_model.selected_material.name
            # A trick to reset the flag, since _simulation_name_changed() is called first
            self.auto_generated = True

    def loadModelMaterials(self, modelName):
        if modelName not in self.model_index.constitutive_models():
            return []

        materials = []
        material_database = self.model_index.control_file(modelName)
        if material_database is not None:
            xml_obj = px.XMLParser(material_database)
            mats = xml_obj.get_parameterized_materials()
            for mat in mats:
                names, params = mat
                name = None
                aliases = ""
                if isinstance(names, (str, unicode)):
                    name = names
                else:
                    for n in names:
                        if name is None:
                            name = n
                        else:
                            if len(aliases) > 0:
                                aliases += '; ' + str(n)
                            else:
                                aliases = n
                parameters = []
                for param in xml_obj.get_material_parameterization(mat[0]):
                    key, default = param
                    if key == "Units":
                        continue
                    parameters.append(PayetteMaterialParameter(name = key, default = default))
                materials.append(PayetteMaterial(name = name, aliases = aliases, defaults = parameters))


        return materials

    def _run_button_fired(self, event):
        runner = ModelRunner(simulation_name=self.simulation_name, material_models=[self.selected_model])
        runner.RunModels()

    def _show_button_fired(self, event):
        runner = ModelRunner(simulation_name=self.simulation_name, material_models=[self.selected_model])
        input_string = runner.CreateModelInputString(self.selected_model)
        preview = PayetteInputStringPreview(input_string=input_string, runner=runner)
        preview.configure_traits()


    view = View(
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
    pm = PayetteMaterialModelSelector()
    pm.configure_traits()

