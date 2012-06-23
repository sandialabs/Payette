try:
    from traits.api import HasStrictTraits, List, Instance, String, BaseInt, Int, Float, Bool, Property, Button, Constant, Enum, Event
    from traitsui.api import View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor, TableEditor, InstanceEditor, ListEditor, Spring, ObjectColumn
    from traitsui.tabular_adapter import TabularAdapter

except ImportError:
    # support for MacPorts install of enthought tools
    from enthought.traits.api import (
        HasStrictTraits, List, Instance, String, BaseInt, Int,
        Float, Bool, Property, Button, Constant, Enum, Event)
    from enthought.traits.ui.api import (
        View, Label, Group, HGroup, VGroup, Item, UItem, TabularEditor,
        InstanceEditor, ListEditor, Spring, TableEditor, ObjectColumn)
    from enthought.traits.ui.tabular_adapter import TabularAdapter

import random

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
    distribution = Enum('Specified', '+/-', 'Range', 'Uniform', 'Gaussian', 'AbsGaussian', 'Weibull')
    value = Property
    default = Property
    specified = String("0")
    percent = Float(10.0)
    minimum = Float(0.0)
    maximum = Float(1.0)
    mean = Float(0.0)
    std_dev = Float(1.0)
    scale = Float(1.0)
    shape = Float(1.0)
    samples = Int(10)

    def _get_value(self):
        if self.distribution == 'Specified':
            return self.specified
        elif self.distribution == '+/-':
            return "%s +/- %s%%" % (self.specified, self.percent)
        elif self.distribution == 'Range':
            return "Range(start = %s, end = %s, steps = %d)" % (self.minimum, self.maximum, self.samples)
        elif self.distribution == 'Uniform':
            return "Uniform(min = %s, max = %s, N = %d)" % (self.minimum, self.maximum, self.samples)
        elif self.distribution == 'Gaussian':
            return "Gaussian(mean = %s, std. dev = %s, N = %d)" % (self.mean, self.std_dev, self.samples)
        elif self.distribution == 'AbsGaussian':
            return "Abs. Gaussian(mean = %s, std. dev = %s, N = %d)" % (self.mean, self.std_dev, self.samples)
        elif self.distribution == 'Weibull':
            return "Weibull(scale = %s, shape = %s, N = %d)" % (self.scale, self.shape, self.samples)
        return "#ERR"

    def _get_default(self):
        if self.distribution == 'Specified':
            return self.specified
        elif self.distribution == '+/-':
            return self.specified
        elif self.distribution == 'Range':
            return self.minimum
        elif self.distribution == 'Uniform':
            return self.minimum
        elif self.distribution == 'Gaussian':
            return self.mean
        elif self.distribution == 'AbsGaussian':
            return self.mean
        elif self.distribution == 'Weibull':
            return random.weibullvariate(self.scale, self.shape)
        return 0.0

    edit_view = View(
        Item('name', label='Parameter', style='readonly'),
        Item('description', style='readonly'),
        Item('distribution'),
        Group(
            VGroup(
                Item('specified', label='Value'),
                visible_when="distribution == 'Specified'"
            ),
            VGroup(
                Item('specified', label='Value'),
                Item('percent'),
                visible_when="distribution == '+/-'"
            ),
            VGroup(
                Item('minimum', label='Start'),
                Item('maximum', label='End'),
                Item('samples', label='Steps'),
                visible_when="distribution == 'Range'"
            ),
            VGroup(
                Item('minimum', label='Min'),
                Item('maximum', label='Max'),
                Item('samples'),
                visible_when="distribution == 'Uniform'"
            ),
            VGroup(
                Item('mean'),
                Item('std_dev'),
                Item('samples'),
                visible_when="distribution == 'Gaussian' or distribution == 'AbsGaussian'"
            ),
            VGroup(
                Item('scale'),
                Item('shape'),
                Item('samples'),
                visible_when="distribution == 'Weibull'"
            ),
            layout='tabbed'
        ),
        buttons=['OK','Cancel']
    )

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
    permutation_method = Enum('Zip', 'Combine')
    cell = Event

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
                    param.distribution = 'Specified'
                    param.specified = default.default
                    break

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

    def _cell_fired(self, info):
        info[0].configure_traits()

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
            editor      = TableEditor (
                auto_size   = False,
                reorderable = False,
                sortable    = True,
                click       = 'cell',
                columns     = [
                    ObjectColumn(name='name', editable=False, width=0.3),
                    ObjectColumn(name='value', editable=False, width=0.7, horizontal_alignment='right')
                ],
            )
        ),
        Label('Permutation Method'),
        UItem('permutation_method', style='custom')
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

