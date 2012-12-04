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

from enthought.traits.api import HasTraits, List, Button, String, Dict
from enthought.traits.ui.api import View, VGroup, HGroup, Label, UItem, Spring, Handler
from enthought.traits.ui.menu import NoButtons

import Payette_model_index as pmi
import Payette_xml_parser as px

from Viz_ModelData import PayetteModel, PayetteModelParameter, PayetteMaterial, PayetteMaterialParameter


class MessageBox(HasTraits):
    """
    Similar to the traits ui dialog box, this one has slightly better
    formatting, and returns which button was clicked.

    Please use the function message_box(...) below to display the window
    """
    message = String
    button_return_values = Dict(String, String)
    button_ = Button
    result = String


class MessageBoxHandler(Handler):
    def setattr(self, info, object, name, value):
        if 'button' in name:
            object.result = object.button_return_values[name]
            info.ui.dispose()

    def close(self, info, is_ok):
        return


def message_box(message, title='Message', buttons=['OK']):
    button_group = HGroup(padding=10, springy=True)
    view = View(
        VGroup(
            UItem('message', style='readonly'),
            button_group,
            padding=15,
        ),
        title=title,
        width=480,
        kind='livemodal',
        buttons=NoButtons,
        handler=MessageBoxHandler()
    )

    button_return_values = {}
    box = MessageBox(message=message)
    button_group.content.append(Spring())
    for i in range(len(buttons)):
        if buttons[i] is None:
            button_group.content.append(Spring())
        else:
            name = 'button_%d' % i
            box.add_trait(name, Button())
            button_group.content.append(UItem(name, label=buttons[i]))
            button_return_values[name] = buttons[i]
    box.button_return_values = button_return_values

    button_group.content.append(Spring())

    box.configure_traits(view=view)

    return box.result


def loadModelMaterials(model_index, model_name):
    if model_name not in model_index.constitutive_models():
        return []

    materials = []
    material_database = model_index.control_file(model_name)
    if material_database is not None:
        xml_obj = px.XMLParser(material_database)
        mats = xml_obj.get_parameterized_materials()
        for mat in mats:
            name, aliases = mat

            parameters = []
            for param in xml_obj.get_material_parameterization(mat[0]):
                key, default = param
                if key == "Units":
                    continue
                parameters.append(
                    PayetteMaterialParameter(name=key, default=default))
            materials.append(PayetteMaterial(
                name=name, aliases=aliases, defaults=parameters))

    return materials


def loadModels(model_type='any'):
    models = []
    model_index = pmi.ModelIndex()

    for model_name in model_index.constitutive_models():
        control_file = model_index.control_file(model_name)
        cmod = model_index.constitutive_model_object(model_name)
        cmod_obj = cmod(control_file)
        if model_type != 'any' and model_type not in cmod_obj.material_type:
            continue

        params = []
        for param in cmod_obj.get_parameter_names_and_values():
            params.append(
                PayetteModelParameter(
                    name=param[0], description=param[1],
                    distribution='Specified', specified=param[2])
            )

        model = PayetteModel(
            model_name=model_name, parameters=params, model_type=[
                cmod_obj.material_type],
            materials=loadModelMaterials(model_index, model_name))
        models.append(model)

    return models


if __name__ == '__main__':
    print message_box('Do you wish to save?', title='Save your work', buttons=['Cancel', None, 'Yes', 'No'])
