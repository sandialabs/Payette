import StringIO
import sys
import random
import os
import datetime

from enthought.traits.api import HasStrictTraits, List, Instance, String, Interface

import Source.Payette_run as pr
from Viz_ModelData import PayetteModel
from Viz_MetaData import VizMetaData
from Viz_ModelPlot import create_Viz_ModelPlot

class IModelRunnerCallbacks(Interface):
    def RunFinished(self, metadata):
        ''' Called when a run completes or fails. '''

class ModelRunner(HasStrictTraits):
    simulation_name = String
    material_models = List(Instance(PayetteModel))
    callbacks = Instance(IModelRunnerCallbacks)

    def RunModels(self):
        for material in self.material_models:
            inputString = self.CreateModelInputString(material)

            self.RunInputString(inputString, material)

    def RunInputString(self, inputString, material):
        #output = StringIO.StringIO()
        oldout = sys.stdout
        #sys.stdout = output

        # tjf: run_payette can be invoked with disp=1 and then it returns a
        # tjf: dictionary with some extra information. I pass that extra
        # tjf: information to the CreatePlotWindow method.
        cmd = ["--input-str={0}".format(inputString)]
        siminfo = pr.run_payette(cmd, disp=1)[0]
        sys.stdout = oldout

        if self.callbacks is None:
            self.CreatePlotWindow(siminfo)
        else:
            now = datetime.datetime.now()
            index_file = siminfo.get('index file')
            if index_file is None:
                index_file = ''
            else:
                base_dir,index_file = os.path.split(index_file)
            output_file = siminfo.get('output file')
            if output_file is None:
                output_file = ''
            else:
                base_dir,output_file = os.path.split(output_file)

            metadata = VizMetaData(
                name = self.simulation_name,
                base_directory = base_dir,
                index_file = index_file,
                out_file = output_file,
                data_type = 'Simulation',
                model_type = ', '.join(material.model_type),
                created_date = now.date(),
                created_time = now.time(),
                successful = True,
                model = material
            )

            self.callbacks.RunFinished(metadata)

    def CreatePlotWindow(self, siminfo):
        # siminfo is a dictionary containing extra output information from the
        # simulation. Pass it directly to create_Viz_ModelPlot
        create_Viz_ModelPlot(self.simulation_name, **siminfo)

    def CreateModelInputString(self, material):
        result = (
            "begin simulation %s\n"
            "  begin material\n"
            "    constitutive model %s\n"
        % (self.simulation_name, material.model_name))

        for p in material.parameters:
            result += "    %s = %s\n" % (p.name, p.default)

        result += "  end material\n"

        model_type = str(material.model_type).lower()
        if 'eos' in model_type:
            result += self.CreateEOSBoundaryInput(material)

        elif 'mechanical' in model_type:
            result += self.CreateMechanicalBoundaryInput(material)

        result += self.CreatePermutation(material)

        result += "end simulation\n"
        return result

    def CreatePermutation(self, material):
        needsPermutation = False
        for p in material.parameters:
            if p.distribution != 'Specified':
                needsPermutation = True

        if not needsPermutation:
            return ""

        result = ("  begin permutation\n"
                  "    method %s\n"
                 % material.permutation_method.lower())
        for p in material.parameters:
            if p.distribution == '+/-':
                val = float(p.specified)
                mult = p.percent / 100.0
                result += "    permutate %s, sequence = (%s, %s, %s)\n" % (
                    p.name, val - val * mult, val, val + val * mult)
            elif p.distribution == 'Range':
                result += "    permutate %s, range = (%s, %s, %s)\n" % (
                    p.name, p.minimum, p.maximum, p.samples)
            elif p.distribution == 'Uniform':
                vals = []
                for i in range(p.samples):
                    vals.append(random.uniform(p.minimum, p.maximum))
                result += "    permutate %s, sequence = %s\n" % (p.name, str(tuple(vals)))
            elif p.distribution == 'Gaussian':
                vals = []
                for i in range(p.samples):
                    vals.append(random.normalvariate(p.mean, p.std_dev))
                result += "    permutate %s, sequence = %s\n" % (p.name, str(tuple(vals)))
            elif p.distribution == 'AbsGaussian':
                vals = []
                for i in range(p.samples):
                    vals.append(abs(random.normalvariate(p.mean, p.std_dev)))
                result += "    permutate %s, sequence = %s\n" % (p.name, str(tuple(vals)))
            elif p.distribution == 'Weibull':
                vals = []
                for i in range(p.samples):
                    vals.append(random.weibullvariate(p.scale, p.shape))
                result += "    permutate %s, sequence = %s\n" % (p.name, str(tuple(vals)))

        result += "  end permutation\n"

        return result


    def CreateEOSBoundaryInput(self, material):
        # XXX Need a less hardcoded way to do this
        T0 = 0.0
        R0 = 0.0
        for p in material.parameters:
            if p.name == "T0":
                T0 = float(p.default)/0.861738573E-4
            elif p.name == "R0":
                R0 = float(p.default)#*1000.0

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




