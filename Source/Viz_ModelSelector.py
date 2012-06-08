try:
    from traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant
    from traitsui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, InstanceEditor, ListEditor, Spring
    from traitsui.tabular_adapter import TabularAdapter
except ImportError:
    from enthought.traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant
    from enthought.traits.ui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, InstanceEditor, ListEditor, Spring
    from enthought.traits.ui.tabular_adapter import TabularAdapter

import Payette_utils as pu
import Payette_xml_parser as px

from Viz_ModelData import PayetteModel, PayetteModelParameter, PayetteMaterial, PayetteMaterialParameter
from Viz_ModelRunner import ModelRunner

class PayetteMaterialModelSelector(HasStrictTraits):
    models = List(Instance(PayetteModel))
    selected_model = Instance(PayetteModel)
    none_constant = Constant("None")
    run_button = Button("Run Material Model")

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)
        self.loadModels()

    def loadModels(self):
        data = pu.get_installed_models()
        for modelName in data.keys():
            control_file = pu.get_constitutive_model_control_file(modelName)
            cmod = pu.get_constitutive_model_object(modelName)
            cmod_obj = cmod(control_file)

            params = []
            for param in cmod_obj.get_parameter_names_and_values():
                params.append(
                    PayetteModelParameter(name = param[0], description = param[1], value = param[2])
                )

            model = PayetteModel(model_name = modelName, parameters = params, model_type = [cmod_obj.material_type])
            self.models.append(model)

        self.selected_model = self.models[0]

    def _selected_model_changed(self, info):
        if info is None or len(info.materials) > 0:
            return

        info.materials = self.loadModelMaterials(info.model_name)

    def loadModelMaterials(self, modelName):
        data = pu.get_installed_models()
        if modelName not in data:
            return []

        materials = []
        material_database = data[modelName]["control file"]
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
        print event
        runner = ModelRunner(material_models=[self.selected_model])
        runner.RunModels()

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
            HGroup(
                Spring(),
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

