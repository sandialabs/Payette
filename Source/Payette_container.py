# The MIT License

# Copyright (c) 2011 Tim Fuller

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


from __future__ import print_function
import os
import sys
import re
import imp
import math
import numpy as np
import time

from Source.Payette_utils import *
from Source.Payette_material import Payette_Constitutive_Models, Material
import Source.Payette_driver as cdriver
from Source.Payette_data_container import DataContainer

def parseError(msg):
    print("ERROR: {0:s}".format(msg))
    sys.exit(4)
    return

class Payette:
    '''
    CLASS NAME
       Payette

    PURPOSE
       main container class for a Payette single element simulation, instantiated in
       the runPayette script. The documentation is currently sparse, but may be
       expanded if time allows.

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self, simname, user_input, opts):

        iam = simname + "__init__(self, simname, user_input, opts)"
        if opts.debug: opts.verbosity = 4
        loglevel = opts.verbosity

        delete = not opts.keep
        basedir = os.getcwd()

        self.name = simname

        outfile = os.path.join(basedir, simname + ".out")
        if delete and os.path.isfile(outfile):
            os.remove(outfile)
        elif os.path.isfile(outfile):
            i = 0
            while True:
                outfile = os.path.join(basedir,'%s.%i.out'%(simname,i))
                if os.path.isfile(outfile):
                    i += 1
                    if i > 100:
                        parseError(r'Come on!  Really, over 100 output files???')
                    else:
                        continue
                else:
                    break
                continue
            pass

        # logfile
        logfile = '%s.log'%os.path.splitext(outfile)[0]
        try: os.remove(logfile)
        except: pass
        setupLogger(logfile,loglevel)
        msg = "setting up simulation %s"%simname
        reportMessage(__file__,msg)

        # file name for the Payette restart file
        rfile = '%s.prf'%os.path.splitext(outfile)[0]

        # set up the simulation data container
        self.simdat = DataContainer(simname)

        # --- register obligatory data
        # plotable data
        self.simdat.registerData("time","Scalar",
                                 init_val = 0.,
                                 plot_key="time")
        self.simdat.registerData("time step","Scalar",
                                 init_val = 0.,
                                 plot_key = "timestep")
        self.simdat.registerData("number of steps","Scalar",
                                 init_val = 0)
        self.simdat.registerData("strain","SymTensor",
                                 init_val = np.zeros(6),
                                 plot_key = "strain")
        self.simdat.registerData("deformation gradient","Tensor",
                                 init_val = "Identity",
                                 plot_key = "F")
        self.simdat.registerData("rate of deformation","SymTensor",
                                 init_val = np.zeros(6),
                                 plot_key = "d")
        self.simdat.registerData("vorticity","Tensor",
                                 init_val = np.zeros(9),
                                 plot_key = "w")
        self.simdat.registerData("equivalent strain","Scalar",
                                 init_val = 0.,
                                 plot_key = "eqveps")
        self.simdat.registerData("permittivity","SymTensor",
                                 init_val = np.zeros(6),
                                 plot_key = "permtv")
        self.simdat.registerData("electric field","Vector",
                                 init_val = np.zeros(3),
                                 plot_key = "efield")
        # non-plotable data
        self.simdat.registerData("prescribed stress","Array",
                                 init_val = np.zeros(6))
        self.simdat.registerData("prescribed stress components","Integer Array",
                                 init_val = np.zeros(6,dtype=int))
        self.simdat.registerData("prescribed strain","SymTensor",
                                 init_val = np.zeros(6))
        self.simdat.registerData("strain rate","SymTensor",
                                 init_val = np.zeros(6))
        self.simdat.registerData("prescribed deformation gradient","Tensor",
                                 init_val = np.zeros(9))
        self.simdat.registerData("deformation gradient rate","Tensor",
                                 init_val = np.zeros(9))
        self.simdat.registerData("rotation","Tensor",
                                 init_val = "Identity")
        self.simdat.registerData("rotation rate","Tensor",
                                 init_val = np.zeros(9))

        # set up simulation from user input
        self.material = None
        self.lcontrol = None
        self.bcontrol = None

        self.boundaryFactory(user_input)
        self.materialFactory(user_input)

        # get mathplot
        mathplot, mathplotid = findBlock(user_input,"mathplot")
        plotable = []
        if mathplot:
            for item in mathplot:
                tmp = item.replace(","," ")
                for repl in [",",";",":"]: item = item.replace(repl," ")
                plotable.extend(item.split())
                continue
            pass

        # register boundary variables
        self.simdat.registerData("leg number","Scalar",
                                 init_val = 0 )

        self.simdat.registerData("leg data","List",
                                 init_val = self.lcontrol)

        # check if user has specified simulation options
        for item in user_input:
            deprication_warning = False
            split_item = item.replace("="," ").replace(","," ").lower().split()

            if split_item[0] == "title": deprication_warning = True

            if deprication_warning:
                msg = ("depricated input file option [{0}] skipped"
                       .format(split_item[0]))
                reportWarning(iam,msg)
                continue

            if len(split_item) == 1: split_item.append("True")
            try: val = eval(split_item[1])
            except: val = str(split_item[1])
            self.simdat.registerOption(split_item[0],val)
            continue

        # register some obligatory options
        self.simdat.registerOption("simname",simname)
        self.simdat.registerOption("outfile",outfile)
        self.simdat.registerOption("restart file",rfile)
        self.simdat.registerOption("verbosity",opts.verbosity)
        self.simdat.registerOption("sqa",opts.sqa)
        self.simdat.registerOption("write vandd table",opts.write_vandd_table)
        self.simdat.registerOption("use table",opts.use_table)
        self.simdat.registerOption("debug",opts.debug)
        self.simdat.registerOption("test restart",opts.testrestart)
        self.simdat.registerOption("write restart",not opts.norestart)
        self.simdat.registerOption("initial time",self.t0)
        self.simdat.registerOption("termination time",self.tf)
        self.simdat.registerOption("kappa",self.bcontrol["kappa"])
        self.simdat.registerOption("emit",self.bcontrol["emit"])
        self.simdat.registerOption("screenout",self.bcontrol["screenout"])
        self.simdat.registerOption("nprints",self.bcontrol["nprints"])
        self.simdat.registerOption("legs",self.lcontrol)
        self.simdat.registerOption("material",self.material)
        self.simdat.registerOption("efield sim",self.material.electricFieldModel())
        self.simdat.registerOption("mathplot vars",[x.upper() for x in plotable])
        self.simdat.registerOption("math1",
                                   os.path.join(basedir, simname + ".math1"))
        self.simdat.registerOption("math2",
                                   os.path.join(basedir, simname + ".math2"))

        if "proportional" not in self.simdat.getAllOptions():
            self.simdat.registerOption("proportional",False)
            pass

        if "strict" not in self.simdat.getAllOptions():
            self.simdat.registerOption("strict",False)
            pass

        # write out properties
        if not opts.nowriteprops: self.writeMaterialParameters()

        pass


    def materialFactory(self,user_input):
        '''
        NAME
           materialFactory

        PURPOSE
           Read material block from input file, parse it, and create material
           object

        AUTHORS
           Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
        '''

        material,mid = findBlock(user_input,'material')

        if not material:
            parseError('Error in material block of input file')
        else:
            material = [x.strip() for x in material]
            pass

        # check for required input
        constitutive_model = None
        f_params = None
        mf_index = None
        user_params = []
        for item in material:
            if 'constitutive model' in item:
                # line defines the constitutive model, get the name of the model
                # to be used
                constitutive_model = item[len('constitutive model'):].strip()
                pass
            elif 'material' in item:
                parseError("material keyword depricated")
                # below is old code for the material input file that is a python
                # file that defines a material based on common names. I might
                # resurrect this in the future, thus I am leaving it.
                m = item[len('material'):].strip()
                fdir,fname = os.path.split(m)
                mname,fext = os.path.splitext(fname)
                if fext:
                    if fext != '.py':
                        parseError('material database file must be a python file, '
                                 'got %s'%m)
                        return 1
                    else: pass
                else: fname = '%s.py'%m

                # material database directories
                lpd = payetteParametersDir()

                # check if material database file is in:
                f = None
                if os.path.isfile(os.path.join(fdir,fname)): # given directory
                    f = os.path.join(fdir,fname)
                    pass
                elif os.path.isfile(os.path.join(os.getcwd(),fname)): # cwd
                    f = os.path.join(os.getcwd(),fname)
                    pass
                elif os.path.isfile(os.path.join(lpd,fname)): #lpd
                    f = os.path.join(lpd,fname)
                    pass
                else:
                    msg = ('%s material file %s not found in [%s, %s, %s]'
                           %(mname,fname,fdir,os.getcwd(),lpd))
                    parseError(msg)
                    return 1
                try: f_params = imp.load_source(mname,f).parameters
                except:
                    parseError('unable to load material parameters from %s'%f)
                    pass

            else:
                user_params.append(item)
                pass

            continue

        # check that the constitutive model is defined in the input file.
        if not constitutive_model:
            # constitutive model not given, exit
            parseError('constitutive model must be specified in material block')
            return

        # constitutive model given, now see if it is available, here we replace
        # spaces with _ in all model names and convert to lower case
        mdlname = constitutive_model.lower().replace(" ","_")
        available_models = Payette_Constitutive_Models
        for key in available_models:
            if mdlname == key or mdlname in available_models[key]["aliases"]:
                constitutive_model = key
                pass
            continue

        # we have done a case insensitive check, replaced spaces with _ and
        # checked aliases, if the model is still not in the available models,
        # bail
        if constitutive_model not in available_models:
            parseError('constitutive model %s not installed'%str(constitutive_model))
            pass

        # instantiate the material object
        self.material = Material(constitutive_model,self.simdat,user_params,f_params)

        return

    def boundaryFactory(self,user_input):
        '''
        NAME
           boundaryFactory

        PURPOSE
           scan the user input for a begin boundary .. end boundary block
           and parse it

        AUTHORS
           Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
        '''
        nodeBoundary_cache = {}
        elementBoundary_cache = {}

        # get boundary block
        boundary,bid = findBlock(user_input,'boundary')
        if not boundary:
            parseError('Error in boundary block of input file')
            return

        # get legs
        legs,lid = findBlock(boundary,'legs')
        if not legs:
            parseError('Error in boundary:leg block of input file')
            return

        # parse boundary
        bcontrol = {'kappa':0.,'estar':1.,'tstar':1.,'ampl':1.,'efstar':1.,
                    'ratfac':1.,'sstar':1.,'fstar':1.,'dstar':1.,
                    'emit':'all','nprints':0,'stepstar':1,"screenout":False}
        for item in boundary:
            item = item.replace('=',' ')
            try: kwd,val = item.split()
            except:
                parseError('boundary control items must be keword = value pairs')
                return 1
            try: kwd,val = kwd.strip(),float(eval(val))
            except: kwd,val = kwd.strip(),str(val)
            if kwd in bcontrol.keys(): bcontrol[kwd] = val
            else: continue
            continue
        if bcontrol['emit'] not in ['all','sparse']:
            parseError('emit must be one of [all,sparse]')
            return 1
        if bcontrol['screenout']: bcontrol['screenout'] = True
        if not isinstance(bcontrol['nprints'], int):
            bcontrol['nprints'] = int(bcontrol['nprints'])
            pass
        if not isinstance(bcontrol['stepstar'], (float,int)):
            bcontrol['stepstar'] = max(0.,float(bcontrol['stepstar']))
            pass
        if bcontrol['stepstar'] <= 0:
            parseError('stepstar must be > 0.')
            return 1

        # the following are from Brannon's MED driver
        # estar is the "unit" of strain
        # sstar is the "unit" of stress
        # fstar is the "unit" of deformation gradient
        # efstar is the "unit" of electric field
        # dstar is the "unit" of electric field
        # tstar is the "unit" of time
        # All strains are multiplied by efac=ampl*estar
        # All stresses are multiplied by sfac=ampl*sstar
        # All deformation gradients are multiplied by ffac=ampl*fstar
        # All electric fields are multiplied by effac=ampl*efstar
        # All displacements are multiplied by dfac=ampl*dstar
        # All times are multiplied by tfac=abs(ampl)*tstar/ratfac
        # From these formulas, note that AMPL may be used to increase or decrease
        # the peak strain without changing the strain rate. ratfac is the
        # multiplier on strain rate and stress rate.
        bcontrol['tfac'] = abs(bcontrol['ampl'])*bcontrol['tstar']/bcontrol['ratfac']
        bcontrol['efac'] = bcontrol['ampl']*bcontrol['estar']
        bcontrol['sfac'] = bcontrol['ampl']*bcontrol['sstar']
        bcontrol['ffac'] = bcontrol['ampl']*bcontrol['fstar']
        bcontrol['effac'] = bcontrol['ampl']*bcontrol['efstar']
        bcontrol['dfac'] = bcontrol['ampl']*bcontrol['dstar']

        # parse legs
        lcontrol = []
        sigc = False
        for ileg,leg in enumerate(legs):
            leg = [x.strip() for x in leg.replace(',',' ').split(' ') if x]
            if len(leg) < 7:
                parseError('leg %s input must be of form: \n'
                         '       leg number, time, steps, type, c[ij]'%leg[0])
                pass
            leg_no = int(float(leg[0]))
            leg_t = bcontrol['tfac']*float(leg[1])
            leg_steps = int(bcontrol['stepstar']*float(leg[2]))
            if ileg != 0 and leg_steps == 0:
                parseError("leg number {0} has no steps".format(leg_no))
                pass
            control = leg[3].strip()
            c = [float(eval(y)) for y in leg[4:]]
            # control should be a group of letters describing what type of
            # control type the leg is. valid options are:
            #  1: strain rate control
            #  2: strain control
            #  3: stress rate control
            #  4: stress control
            #  5: deformation gradient control
            #  6: electric field
            #  8: displacement
            allwd_cntrl = '1234568'
            if [x for x in control if x not in allwd_cntrl]:
                parseError('leg control parameters can only be one of '
                         '[%s] got %s for leg number %i'
                         %(allwd_cntrl,control,leg_no))
                pass

            if not sigc: sigc = bool([x for x in control if x == '3' or x == '4'])

            lcntrl = [int(x) for x in list(control)]

            if len(lcntrl) != len(c):
                parseError('length of leg control != number of control items '
                         'in leg %i'%leg_no)
                pass

            # separate out electric fields from deformations
            ef,hold,efcntrl = [], [], []
            for i,j in enumerate(lcntrl):
                if j == 6:
                    ef.append(c[i])
                    hold.append(i)
                    efcntrl.append(j)
                    pass
                continue
#            if ef and sigc: parseError("Stress and electric field control not "
#                                     "allowed simulataneously")
            ef.extend([0.]*(3-len(ef)))
            efcntrl.extend([6]*(3-len(efcntrl)))
            c = [i for j, i in enumerate(c) if j not in hold]
            lcntrl = [i for j, i in enumerate(lcntrl) if j not in hold]

            if len(lcntrl) != len(c):
                parseError('final length of leg control != number of control items '
                         'in leg %i'%leg_no)
                pass

            reduced_lcntrl = list(set(lcntrl))
            if 5 in reduced_lcntrl:
                # deformation gradient control check
                if len(reduced_lcntrl) != 1:
                    parseError('only components of deformation gradient are allowed '
                             'with deformation gradient control in leg %i, got %s'
                             %(leg_no,control))
                    return 1
                elif len(c) != 9:
                    parseError('all 9 components of deformation gradient must '
                             'be specified for leg %i'%leg_no)
                    return 1
                else:
                    # check for valid deformation
                    F = np.array([[c[0],c[1],c[2]],
                                  [c[3],c[4],c[5]],
                                  [c[6],c[7],c[8]]])
                    J = np.linalg.det(F)
                    if J <= 0:
                        parseError('inadmissible deformation gradient in leg %i '
                                 'gave a Jacobian of %f'%(leg_no,J))
                        return 1
                    # convert F to strain E with associated rotation given by
                    # axis of rotation x and angle of rotation theta
                    R,V = np.linalg.qr(F)
                    U = np.dot(R.T,F)
                    if np.max(np.abs(R - np.eye(3))) > epsilon():
                        parseError('deformations with rotation encountered in leg %i. '
                                 'rotations are not yet supported'%leg_no)
                        return 1

            elif 8 in reduced_lcntrl:
                # displacement control check
                if len(reduced_lcntrl) != 1:
                    parseError('only components of displacment are allowed '
                             'with displacment control in leg %i, got %s'
                             %(leg_no,control))
                    return 1
                elif len(c) != 3:
                    parseError('all 3 components of displacement must '
                             'be specified for leg %i'%leg_no)
                    return 1

                # convert displacments to strains
                kappa,dfac = bcontrol["kappa"],bcontrol["dfac"]
                # Seth-Hill generalized strain is defined
                # strain = (1/kappa)*[(stretch)^kappa - 1]
                # and
                # stretch = displacement + 1

                # In the limit as kappa->0, the Seth-Hill strain becomes
                # strain = ln(stretch).
                for j in range(3):
                    stretch = dfac*c[j] + 1
                    if kappa != 0: c[j] = 1/kappa*(stretch**kappa - 1.)
                    else: c[j] = math.log(stretch)
                    continue

                # displacements now converted to strains
                lcntrl = [2,2,2]
                pass

            i = 0
            for j in lcntrl:
                if j == 1 or j == 3: c[i] = bcontrol['ratfac']*c[i]
                elif j == 2: c[i] = bcontrol['efac']*c[i]
                elif j == 4: c[i] = bcontrol['sfac']*c[i]
                elif j == 5: c[i] = bcontrol['ffac']*c[i]
                elif j == 6: c[i] = bcontrol['effac']*c[i]
                i += 1
                continue

            # fill in c and lcntrl so that their lengths are always 9
            c.extend([0.]*(9-len(c)))
            lcntrl.extend([0]*(9-len(lcntrl)))

            # append leg control
            # the electric field control is added to the end of lcntrl
            lcontrol.append([leg_no,leg_t,leg_steps,lcntrl + efcntrl,np.array(c + ef)])
            continue

        if sigc:
            # stress and or stress rate is used to control this leg. For
            # these cases, kappa is set to 0. globally.
            if bcontrol['kappa'] != 0.:
                reportWarning(__file__,'WARNING: stress control boundary conditions '
                                 'only compatible with kappa=0. kappa is being '
                                 'reset to 0. from %f\n'%bcontrol['kappa'])
                bcontrol['kappa'] = 0.
                pass

        # check that time is monotonic in lcontrol
        i = -0.001
        self.t0 = None
        for leg in lcontrol:
            if self.t0 == None: self.t0 = leg[1]
            if leg[1] < i:
                print(leg[1],i)
                print(leg)
                parseError('time must be monotonic in from %i to %i'
                         %(int(leg[0]-1),int(leg[0])))
                pass
            i = leg[1]
            self.tf = leg[1]
            continue
        self.lcontrol = lcontrol
        self.bcontrol = bcontrol
        return

    def finish(self):
        closeFiles()
        del self.simdat
        return

    def simulationData(self):
        return self.simdat

    def writeMaterialParameters(self):
        with open( self.simdat.simname + ".props", "w" ) as f:
            matdat = self.material.materialData()
            for item in matdat.parameter_table:
                key = item["name"]
                val = item["adjusted value"]
                f.write("{0:s} = {1:12.5E}\n".format(key,val))
                continue
            pass
        return

    def setupRestart(self):
        setupLogger(self.simdat.logfile,self.simdat.loglevel)
        msg = "setting up simulation %s"%self.simdat.simname
        reportMessage(__file__,msg)

if __name__ == "__main__":
    sys.exit("Payette_container.py must be called by runPayette")
