import StringIO
import sys

try:
    from traits.api import HasStrictTraits, List, Instance, String

except ImportError:
    # support for MacPorts install of enthought tools
    from enthought.traits.api import HasStrictTraits, List, Instance, String

import Source.Payette_run as pr
from Viz_ModelData import PayetteModel

class ModelRunner(HasStrictTraits):
    simulation_name = String
    material_models = List(Instance(PayetteModel))

    def RunModels(self):
        for material in self.material_models:
            inputString = self.CreateModelInputString(material)

            self.RunInputString(inputString)


    def RunInputString(self, inputString):
            #output = StringIO.StringIO()

            oldout = sys.stdout
            #sys.stdout = output
            pr.run_payette(["--input-str", inputString])
            sys.stdout = oldout


    def CreateModelInputString(self, material):
        result = (
            "begin simulation %s\n"
            "  begin material\n"
            "    constitutive model %s\n"
        % (self.simulation_name, material.model_name))

        for p in material.parameters:
            result += "    %s = %s\n" % (p.name, p.value)

        result += "  end material\n"

        model_type = str(material.model_type).lower()
        if 'eos' in model_type:
            result += self.CreateEOSBoundaryInput(material)

        elif 'mechanical' in model_type:
            result += self.CreateMechanicalBoundaryInput(material)

        result += "end simulation\n"
        return result

    def CreateEOSBoundaryInput(self, material):
        # XXX Need a less hardcoded way to do this
        T0 = 0.0
        R0 = 0.0
        for p in material.parameters:
            if p.name == "T0":
                T0 = float(p.value)/0.861738573E-4
            elif p.name == "R0":
                R0 = float(p.value)*1000.0

        result = (
            "  begin boundary\n"
            "\n"
            "    nprints 5\n"
            "\n"
            "    input units MKSK\n"
            "    output units MKSK\n"
            "\n"
            "    density range %f %f\n"
            "    temperature range %f %f\n"
            "\n"
            "    surface increments %d\n"
            "\n"
            "    path increments %d\n"
            % (material.eos_boundary.min_density,
               material.eos_boundary.max_density,
               material.eos_boundary.min_temperature,
               material.eos_boundary.max_temperature,
               material.eos_boundary.surface_increments,
               material.eos_boundary.path_increments))

        if material.eos_boundary.isotherm:
            result += "    path isotherm %f %f\n" % (R0, T0)
        if material.eos_boundary.isentrope:
            result += "    path isentrope %f %f\n" % (R0, T0)
        if material.eos_boundary.hugoniot:
            result += "    path hugoniot %f %f\n" % (R0, T0)

        result += "  end boundary\n"

        return result

    def CreateMechanicalBoundaryInput(self, material):
        result = (
            "  begin boundary\n"
            "    kappa = 0.\n"
            "    tstar = %f\n"
            "    fstar = %f\n"
            "    sstar = %f\n"
            "    dstar = %f\n"
            "    estar = %f\n"
            "    efstar = %f\n"
            "    ampl = %f\n"
            "    ratfac = %f\n"
            "    begin legs\n"
            % (material.TSTAR, material.FSTAR, material.SSTAR, material.DSTAR,
               material.ESTAR, material.EFSTAR, material.AMPL, material. RATFAC)
        )

        leg_num = 0
        for leg in material.legs:
            result += "      %d %f %d %s %s\n" % (leg_num, leg.time, leg.nsteps, leg.types, leg.components)
            leg_num += 1

        result += (
            "    end legs\n"
            "  end boundary\n"
        )
        return result




