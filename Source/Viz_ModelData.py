try:
    from traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant, Enum
    from traitsui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, InstanceEditor, ListEditor, Spring
    from traitsui.tabular_adapter import TabularAdapter

except ImportError:
    # support for MacPorts install of enthought tools
    from enthought.traits.api import (
        HasStrictTraits, List, Instance, String, BaseInt, Int,
        Float, Bool, Property, Button, Constant, Enum)
    from enthought.traits.ui.api import (
        View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor,
        InstanceEditor, ListEditor, Spring)
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
    value = String("0")

class PayetteModelParameterAdapter(TabularAdapter):
    columns = [('Parameter', 'name'), ('Value', 'value')]

    def get_tooltip (self, object, trait, row, column):
        return object.get(trait)[trait][row].description

class PayetteMaterialParameter(HasStrictTraits):
    name = String
    default = String

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
    auto_density = Bool(True)

    def _min_density_changed(self, info):
        self.auto_density = False

    def _max_density_changed(self, info):
        self.auto_density = False

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
    TSTAR = Float(1.0)
    FSTAR = Float(1.0)
    SSTAR = Float(1.0)
    DSTAR = Float(1.0)
    ESTAR = Float(1.0)
    EFSTAR = Float(1.0)
    AMPL = Float(1.0)
    RATFAC = Float(1.0)
    leg_defaults = Enum('Custom', 'Uniaxial Strain', 'Biaxial Strain',
                        'Spherical Strain', 'Uniaxial Stress',
                        'Biaxial Stress', 'Spherical Stress')
    legs = List(PayetteLeg, [PayetteLeg()])

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)
        for param in self.parameters:
            if param.name == "R0" and param.value is not None:
                param.on_trait_change(self.update_density(param), 'value')

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

    def update_density(self, param):
        def do_update():
            if self.eos_boundary.auto_density:
                try:
                    r0 = float(param.value)
                    self.eos_boundary.min_density = r0 * 0.9 * 1000
                    self.eos_boundary.max_density = r0 * 1.1 * 1000
                except:
                    pass
                self.eos_boundary.auto_density = True

        return do_update

    def _leg_defaults_changed(self, info):
        if info == 'Uniaxial Strain':
            self.legs = [
                PayetteLeg(time=0, nsteps=0, types='222222', components='0 0 0 0 0 0'),
                PayetteLeg(time=1, nsteps=100, types='222222', components='1 0 0 0 0 0'),
            ]
        elif info == 'Biaxial Strain':
            self.legs = [
                PayetteLeg(time=0, nsteps=0, types='222222', components='0 0 0 0 0 0'),
                PayetteLeg(time=1, nsteps=100, types='222222', components='1 1 0 0 0 0'),
            ]
        elif info == 'Spherical Strain':
            self.legs = [
                PayetteLeg(time=0, nsteps=0, types='222222', components='0 0 0 0 0 0'),
                PayetteLeg(time=1, nsteps=100, types='222222', components='1 1 1 0 0 0'),
            ]
        elif info == 'Uniaxial Stress':
            self.legs = [
                PayetteLeg(time=0, nsteps=0, types='444444', components='0 0 0 0 0 0'),
                PayetteLeg(time=1, nsteps=100, types='444444', components='1 0 0 0 0 0'),
            ]
        elif info == 'Biaxial Stress':
            self.legs = [
                PayetteLeg(time=0, nsteps=0, types='444444', components='0 0 0 0 0 0'),
                PayetteLeg(time=1, nsteps=100, types='444444', components='1 1 0 0 0 0'),
            ]
        elif info == 'Spherical Stress':
            self.legs = [
                PayetteLeg(time=0, nsteps=0, types='444444', components='0 0 0 0 0 0'),
                PayetteLeg(time=1, nsteps=100, types='444444', components='1 1 1 0 0 0'),
            ]

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
        HGroup(
            Item('leg_defaults', style='simple'),
            VGroup(
                Item('TSTAR',label='TSTAR'),
                Item('FSTAR',label='FSTAR')
            ),
            VGroup(
                Item('SSTAR',label='SSTAR'),
                Item('DSTAR',label='DSTAR')
            ),
            VGroup(
                Item('ESTAR',label='ESTAR'),
                Item('EFSTAR',label='EFSTAR')
            ),
            VGroup(
                Item('AMPL',label='AMPL'),
                Item('RATFAC',label='RATFAC')
            ),
            style='simple'
        ),
        UItem('legs',
            editor = ListEditor(
                style='custom'
            ),
        ),
        style='custom',
    )

