import StringIO
import sys

from traits.api import HasStrictTraits, List, Instance

import Source.Payette_run as pr
from Viz_ModelData import PayetteModel

class ModelRunner(HasStrictTraits):
    material_models = List(Instance(PayetteModel))

    def RunModels(self):
        for material in self.material_models:
            inputString = self.CreateModelInputString(material)

            output = StringIO.StringIO()

            oldout = sys.stdout
            #sys.stdout = output
            pr.run_payette(["--input-str", inputString])
            sys.stdout = oldout


    def CreateModelInputString(self, material):
        result = """
        begin simulation %s
          begin material
            constitutive model %s
        
        """ % (material.model_name, material.model_name)
            
        for p in material.parameters:
            result += "    %s = %s\n" % (p.name, p.value)

        result += """
          end material
        """

        if 'eos' in material.model_type:
            result += self.CreateEOSBoundaryInput(material)

        if 'mechanical' in material.model_type:
            result += self.CreateMechanicalBoundaryInput(material)

        result += """
        end simulation
        """
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

        result = """
          begin boundary

            nprints 5

            input units MKSK
            output units MKSK

            density range %f %f
            temperature range %f %f

            surface increments %d

            path increments %d
        """ % (material.eos_boundary.min_density,
               material.eos_boundary.max_density,
               material.eos_boundary.min_temperature,
               material.eos_boundary.max_temperature,
               material.eos_boundary.surface_increments,
               material.eos_boundary.path_increments)

        if material.eos_boundary.isotherm:
            result += "    path isotherm %f %f\n" % (R0, T0)
        if material.eos_boundary.isentrope:
            result += "    path isentrope %f %f\n" % (R0, T0)
        if material.eos_boundary.hugoniot:
            result += "    path hugoniot %f %f\n" % (R0, T0)

        result += """
          end boundary
        """

        return result

    def CreateMechanicalBoundaryInput(self, material):
        result = """
          begin boundary
            kappa = 0.
            begin legs
        """

        leg_num = 0
        for leg in material.legs:
            result += "      %d %f %d %s %s\n" % (leg_num, leg.time, leg.nsteps, leg.types, leg.components)
            leg_num += 1

        result += """
            end legs
          end boundary
        """
        return result




