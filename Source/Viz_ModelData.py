try:
    from traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant
    from traitsui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, InstanceEditor, ListEditor, Spring
    from traitsui.tabular_adapter import TabularAdapter
except ImportError:
    from enthought.traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant
    from enthought.traits.ui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, InstanceEditor, ListEditor, Spring
    from enthought.traits.ui.tabular_adapter import TabularAdapter

class TraitPositiveInteger(BaseInt):

    default_value = 1

    def validate(self, obj, name, value):
        value = super(TraitPositiveInteger, self).validate(object, name, value)
        if value > 0:
            return value
        self.error(obj, name, value)

    def info(self):
        return 'a positive integer'

class PayetteModelParameter(HasStrictTraits):
    name = String
    description = String
    value = Float(0.0)

class PayetteModelParameterAdapter(TabularAdapter):
    columns = [('Parameter', 'name'), ('Value', 'value')]

    def get_tooltip (self, object, trait, row, column):
        return object.get(trait)[trait][row].description

class PayetteMaterialParameter(HasStrictTraits):
    name = String
    default = Float

class PayetteMaterial(HasStrictTraits):
    name = String
    aliases = String
    defaults = List(Instance(PayetteMaterialParameter))

class PayetteMaterialAdapter(TabularAdapter):
    columns = [('Materials Available for Selected Model', 'name'), ('Aliases', 'aliases')]

class PayetteEOSBoundary(HasStrictTraits):
    path_increments = TraitPositiveInteger(10000)
    surface_increments = TraitPositiveInteger(20)
    min_density = Float(8000)
    max_density = Float(16000)
    min_temperature = Float(200)
    max_temperature = Float(2000)
    isotherm = Bool(True)
    hugoniot = Bool(True)
    isentrope = Bool(False)

class PayetteLeg(HasStrictTraits):
    time = Float(0.0)
    nsteps = Int(0)
    types = String
    components = String

    view = View(
        HGroup(
            Item('time'),
            Item('nsteps'),
            Item('types'),
            Item('components')
        )
    )

class PayetteModel(HasStrictTraits):
    model_name = String
    model_type = List(String)
    parameters = List(Instance(PayetteModelParameter))
    selected_parameter = Instance(PayetteModelParameter)
    materials = List(Instance(PayetteMaterial))
    selected_material = Instance(PayetteMaterial)
    eos_boundary = Instance(PayetteEOSBoundary, PayetteEOSBoundary())
    legs = List(PayetteLeg, [PayetteLeg()])

    def _selected_material_changed(self, info):
        if info is None:
            return

        for default in info.defaults:
            for param in self.parameters:
                if param.name == default.name:
                    param.value = default.default
                    break

        # Dumb workaround to get the list to update, I'm sure there's a better way
        params = self.parameters
        self.parameters = []
        self.parameters = params

    def _legs_changed(self, info):
        for i in range(len(info)):
            if info[i] is None:
                info[i] = PayetteLeg()

    param_view = View(
        UItem('parameters',
            editor      = TabularEditor(
                show_titles = True,
                selected    = 'selected_parameter',
                editable    = True,
                adapter     = PayetteModelParameterAdapter()),
        )
    )

    material_view = View(
        UItem('materials',
            editor      = TabularEditor(
                show_titles = True,
                selected    = 'selected_material',
                editable    = False,
                adapter     = PayetteMaterialAdapter()),
        )
    )

    eos_boundary_view = View(
        UItem('eos_boundary'),
        style='custom',
    )

    boundary_legs_view = View(
        UItem('legs',
            editor = ListEditor(
                style='custom'
            ),
        ),
        style='custom',
    )

