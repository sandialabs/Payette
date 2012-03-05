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
import scipy.linalg
import time

from Source.Payette_utils import *
import Source.Payette_material as cmtl
import Source.Payette_driver as cdriver

def parseError(msg):
    print("ERROR: {0:s}".format(msg))
    sys.exit()
    return

class Payette:
    '''
    CLASS NAME
       Payette

    PURPOSE
       main container class for a Payette single element simulation, instantiated in
       the runPayette script. The documentation is currently sparse, but may be
       expanded if time allows.

    OBJECT DATA
       Relevant data, this list is not comprehensive.

         variable           type      size    description
         ------------       --------  ----    --------------------------------------
         mt                 instance  1       material object
         sym_vel_grad       array     (2,6)   symmetic part of velocity gradient d
                                                d[0,:] at time n
                                                d[1,:] (to be) at time n + 1
         strain             array     (2,6)   strain E
                                                E[0,:] at time n
                                                E[1,:] (to be) at time n + 1
         rate_of_strain     array     (6)     rate of strain dEdt
         prescribed_strain  array     (3,6)   prescribed strain Epres
                                                Epres[0,:] at tleg[0]
                                                Epres[1,:] at tleg[1]
                                                Epres[2,:] at t (interpolated)
         defgrad            array     (2,9)   deformation gradient F
                                                F[0,:] at time n
                                                F[1,:] (to be) at time n + 1
         rate_of_defgrad    array     (9)     rate of deformation gradient dFdt
         prescribed_defgrad array     (3,9)   prescribed deformation gradient Fpres
                                                Fpres[0,:] at tleg[0]
                                                Fpres[1,:] at tleg[1]
                                                Fpres[2,:] at t (interpolated)
         kappa              scalar    1       Seth-Hill parameter
         lcontrol           list      var     input deformation leg.  size varies by
                                              prescribed [deformation,stress] type.
         stress             array     (2,6)   stress P conjugate to E
         rate_of_stress     array     (6)     rate of stress dPdt
         stress_dum         array     (2,6)   dummy holder for prescribed stress Pdum
                                                Pdum[0,:] at tleg[0]
                                                Pdum[1,:] at tleg[1]
         prescribed_stress  array     (3,?)   array Ppres containing only those
                                              components of Pdum for which stresses
                                              were actually prescribed
                                                Ppres[0,:] at tleg[0]
                                                Ppres[1,:] at tleg[1]
                                                Ppres[2,:] at t (interpolated)
         rotation           array     (2,9)   rotation R (=I for now)
                                                R[0,:] at time n
                                                R[1,:] (to be) at time n + 1
         rate_of_rotation   array     (9)     rate of rotation dRdt
         prescribed_rotation array    (3,9)   prescribed rotation Rpres (not used)
                                                Rpres[0,:] at tleg[0]
                                                Rpres[1,:] at tleg[1]
                                                Rpres[2,:] at t (interpolated)
         t                  scalar    1       current time
         tleg*              array     (2)     leg time
                                                tleg[0]: time at beginning of leg
                                                tleg[1]: time at end of leg
         v                  array     var     vector subscript array containing the
                                              components for which stresses
                                              (or stress rates) are prescribed
         skew_vel_grad      array     (2,9)   skew part of velocity gradient w
                                                w[0,:] at time n
                                                w[1,:] (to be) at time n + 1

    AUTHORS
       Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
    '''

    def __init__(self, simname, user_input, opts):

        self.opts = opts
        if self.opts.debug: self.opts.verbosity = 4
        self.loglevel = self.opts.verbosity
        self.write_restart = not self.opts.norestart
        writeprops = not self.opts.nowriteprops
        self.test_restart = self.opts.testrestart

        delete = not opts.keep
        basedir = os.getcwd()
        self.simname = simname
        self.name = simname
        self.class_name = "Payette"

        self.math1 = os.path.join(basedir, self.simname + ".math1")
        self.math2 = os.path.join(basedir, self.simname + ".math2")

        # all material data
        self.material_data = {}

        self.extra_vars_registered = False

        self.outfile = os.path.join(basedir, self.simname + ".out")
        if delete and os.path.isfile(self.outfile):
            os.remove(self.outfile)
        elif os.path.isfile(self.outfile):
            i = 0
            while True:
                self.outfile = os.path.join(basedir,'%s.%i.out'%(self.simname,i))
                if os.path.isfile(self.outfile):
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
        self.logfile = '%s.log'%os.path.splitext(self.outfile)[0]
        try: os.remove(self.logfile)
        except: pass
        setupLogger(self.logfile,self.loglevel)
        msg = "setting up simulation %s"%self.simname
        reportMessage(__file__,msg)


        # file name for the Payette restart file
        self.rfile = '%s.prf'%os.path.splitext(self.outfile)[0]

        self.mtl = None
        self.lcontrol = None
        self.bcontrol = None

        # set up simulation from user input
        self.boundaryFactory(user_input)
        self.materialFactory(user_input)

        if not self.diagonal and opts.write_vandd_table:
            parseError("cannot create displacement tables from nondiagonal strain")
            return 1
        self.write_vandd_table = opts.write_vandd_table

        # material parameter keys
        self.paramkeys = [None]*self.mtl.cm.nprop
        for param in self.mtl.cm.parameter_table:
            i = self.mtl.cm.parameter_table[param]['ui pos']
            self.paramkeys[i] = param

        # write out properties
        if writeprops: self.writeMaterialParameters()

        # get mathplot
        mathplot, mathplotid = findBlock(user_input,"mathplot")
        self.plotable = []
        if mathplot:
            for item in mathplot:
                tmp = item.replace(","," ")
                for repl in [",",";",":"]: item = item.replace(repl," ")
                self.plotable.extend(item.split())
                continue
            pass

        # ----- Initialize model variables

        # timing
        self.ileg = 0
        self.tleg = np.zeros(2)
        self.t = 0.
        self.dt = 0.

        # strain
        self.strain = np.zeros((2,6))
        self.prescribed_strain = np.zeros((3,6))
        self.rate_of_strain = np.zeros(6)

        # velocity gradient
        self.sym_vel_grad = np.zeros((2,6))
        self.skew_vel_grad = np.zeros((2,9))

        # stress
        self.stress = np.zeros((2,6))
        self.prescribed_stress = np.zeros((3,6))
        self.rate_of_stress = np.zeros(6)
        self.stress_dum = np.zeros((2,6))
        self.stress_dum_comp = np.zeros(6,dtype=int)

        I9 = np.array([1.,1.,1.,0.,0.,0.,0.,0.,0.],dtype='double')
        # rotation
        self.rotation = np.array([I9,I9])
        self.prescribed_rotation = np.array([I9,I9,I9])
        self.rate_of_rotation = np.zeros(9)

        # deformation gradient
        self.defgrad = np.array([I9,I9])
        self.prescribed_defgrad = np.array([I9,I9,I9])
        self.rate_of_defgrad = np.zeros(9)

        # electric field
        self.efield = np.zeros((2,3))
        self.prescribed_efield = np.zeros((3,3))

        # electric displacement
        self.electric_displacement = np.zeros((2,3))

        # polarization
        self.polarization = np.zeros((2,3))

        # failure
        self.failed = False

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
        available_models = cmtl.Payette_Constitutive_Models
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

        # instantiate the constitutive model
        cmod = cmtl.Payette_Constitutive_Models[constitutive_model]["class name"]
        self.constitutive_model = cmod()
        self.electric_field_simulation = self.constitutive_model.electric_field_model

        # check if the model was successfully imported
        if not self.constitutive_model.imported:
            msg = ("Error importing the {0} material model.\n"
                   "If the material model is a fortran extension library, "
                   "it probably was not built correctly.\nTo check, go to "
                   "{1}/Source/Materials/Library\nand try importing "
                   "the material's extension module directly in a python "
                   "session.\nIf it does not import, you will need to rebuild "
                   "the extension module.\n"
                   "If rebuilding Payette does not fix the problem, "
                   "please contact the Payette\ndevelopers."
                   .format(self.constitutive_model.name,Payette_Materials_Library))
            reportError(__file__,msg)
            pass

        # initialize material
        user_params = self.constitutive_model.parseParameters(user_params,f_params)

        # register common parameters
        self.registerCommonData()

        # finish set up
        self.constitutive_model.setUp(self,user_params)
        self.constitutive_model.checkSetUp()
        self.constitutive_model.intializeState()

        # instantiate a material object
        self.mtl = cmtl.Material(self.constitutive_model)

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
        self.isdiag = True
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
            if self.isdiag: self.isdiag = not [x for x in c[3:] if x > 0.]
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
                    F = np.array([[c[0],c[3],c[5]],
                                  [c[6],c[1],c[4]],
                                  [c[8],c[7],c[2]]])
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

    def getPayetteState(self):
        return (
            # timing
            self.ileg, self.lcontrol[self.ileg:],self.tleg, self.t, self.dt,

            # strain
            self.strain, self.prescribed_strain, self.rate_of_strain,

            # stress
            self.stress, self.prescribed_stress, self.rate_of_stress,

            # stress control
            self.stress_dum, self.stress_dum_comp,

            # rotation
            self.rotation, self.prescribed_rotation, self.rate_of_rotation,

            # deformation gradient
            self.defgrad, self.prescribed_defgrad, self.rate_of_defgrad,

            # velocity gradient
            self.sym_vel_grad, self.skew_vel_grad,

            # electric field
            self.efield, self.prescribed_efield,

            # electric displacement
            self.electric_displacement,

            # polarization
            self.polarization,

            # failure
            self.failed)


    def savePayetteState(self,ileg,tleg,t,dt,e,epres,dedt,s,spres,dsdt,sdum,v,
                         r,rpres,drdt,f,fpres,dfdt,d,w,
                         ef,efpres,ed,polrzn,failed):
        # timing
        self.ileg,self.tleg,self.t,self.dt, = ileg,tleg,t,dt

        # strain
        self.strain,self.prescribed_strain,self.rate_of_strain = e,epres,dedt

        # stress
        self.stress,self.prescribed_stress,self.rate_of_stress = s,spres,dsdt

        # stress control
        self.stress_dum,self.stress_dum_comp = sdum,v

        # rotation
        self.rotation,self.prescribed_rotation,self.rate_of_rotation = r,rpres,drdt

        # deformation gradient
        self.defgrad,self.prescribed_defgrad,self.rate_of_defgrad = f,fpres,dfdt

        # velocity gradient
        self.sym_vel_grad,self.skew_vel_grad = d,w

        # electric field
        self.efield, self.prescribed_efield = ef, efpres

        # electric displacement
        self.electric_displacement = ed

        # polarization
        self.polarization = polrzn

        self.failed = failed
        return None

    def registerCommonData(self):

        self.material_data_idx = 0

        # register obligatory variables
        self.registerMaterialData("time","time",0.,typ="Scalar")
        self.registerMaterialData("stress","sig",np.zeros(6),typ="SymTensor")
        self.registerMaterialData("time rate of stress","dsigdt",
                                  np.zeros(6),typ="SymTensor")
        self.registerMaterialData("strain","eps",np.zeros(6),typ="SymTensor")
        self.registerMaterialData("time rate of strain","d",
                                  np.zeros(6),typ="SymTensor")
        self.registerMaterialData("deformation gradient","F",np.zeros(9),
                                  typ="Tensor")
        self.registerMaterialData("equivalent strain","eqveps",0.,typ="Scalar")

        # electric field items
        if self.electric_field_simulation:
            self.registerMaterialData("electric field","efield",np.zeros(3),
                                      typ="Vector")
            self.registerMaterialData("polarization","polrzn",np.zeros(3),
                                      typ="Vector")
            self.registerMaterialData("electric displacement","edisp",np.zeros(3),
                                      typ="Vector")
            pass

        pass

    def registerExtraVariables(self,nsv,names,keys,values):
        """
           method used by material models to register extra variables with payette
        """
        iam = self.class_name+".registerExtraVariables(self,nsv,names,keys,values)"
        if self.extra_vars_registered:
            reporteError(iam,"extra variables can only be registered once")

        self.extra_vars_registered = True
        self.num_extra = nsv
        self.extra_vars_map = {}

        for i in range(nsv):
            name = names[i]
            key = keys[i]
            value = values[i]
            self.registerMaterialData(name,key,value,typ="scalar")
            self.extra_vars_map[i] = name
            continue

        if len(self.material_data) != self.material_data_idx:
            reportError(iam,"duplicate extra variable names")
        pass

    def registerMaterialData(self,name,key,value,typ):
        """
            register the material data for Payette to track
        """
        iam = self.class_name + ".registerMaterialData"
        typ = typ.lower()

        if typ == "symtensor":
            if not isinstance(value,(list,np.ndarray)):
                reportError(iam,("SymTensor material data {0} must be a list "
                                 "or numpy.ndarray").format(name))
            elif len(value) != 6:
                reportError(iam,"initial value of SymTensor material data {0} != 6"
                            .format(name))
                pass

        elif typ == "tensor":
            if not isinstance(value,(list,np.ndarray)):
                reportError(iam,("Tensor material data {0} must be a list "
                                 "or numpy.ndarray").format(name))
            elif len(value) != 9:
                reportError(iam,"initial value of Tensor material data {0} != 9"
                            .format(name))
                pass

        elif typ == "scalar":
            if isinstance(value,(list,np.ndarray)):
                reportError(iam,("Scalar material data {0} must not be a list "
                                 "or numpy.ndarray").format(name))
                pass

        elif typ == "vector":
            if not isinstance(value,(list,np.ndarray)):
                reportError(iam,("Vector material data {0} must be a list "
                                 "or numpy.ndarray").format(name))
            elif len(value) != 3:
                reportError(iam,"initial value of Vector material data {0} != 9"
                            .format(name))
                pass

        else:
            reportError(iam,"unrecognized material data type: {0}".format(typ))
            pass

        # register the data
        self.material_data[name] = { "name":name,
                                     "plot key":key,
                                     "idx":self.material_data_idx,
                                     "type":typ,
                                     "value":value }
        self.material_data_idx += 1
        pass

    def updateMaterialData(self,**kwargs):
        """
        """
        for key, val in kwargs.items():

            if key == "extra variables":
                for isv, sv in enumerate(val):
                    nkey = self.extra_vars_map[isv]
                    self.material_data[nkey]["value"] = sv
                    continue
                continue

            if ( key in ["electric field","electric displacement","polarization"]
                 and not self.electric_field_simulation ):
                continue

            self.material_data[key]["value"] = val

            if key == "strain":
                self.material_data["equivalent strain"]["value"] = (
                    np.sqrt( 2./3.*( sum(val[:3]**2) + 2.*sum(val[3:]**2))) )
                pass

            continue
        pass

    def plotKeys(self):
        iam = self.class_name + ".plotKeys(self)"
        # return a list of plot keys in the order registered
        plot_keys = [None]*len(self.material_data)
        for key,val in self.material_data.items():
            typ = val["type"]
            key = val["plot key"]
            idx = val["idx"]
            if typ == "scalar":
                plot_keys[idx] = [key]

            elif typ == "vector":
                plot_keys[idx] = [key + "1", key + "2", key + "3"]

            elif typ == "symtensor":
                plot_keys[idx] = [key + "11", key + "22", key + "33",
                                  key + "12", key + "23", key + "13" ]

            elif typ == "tensor":
                plot_keys[idx] = [key + "11", key + "22", key + "33",
                                  key + "12", key + "23", key + "13",
                                  key + "21", key + "32", key + "31" ]

            else:
                reportError(iam,"unrecognized material data type {0}".format(typ))
                pass

            continue
        # flatten plot keys to just a list
        plot_keys = [x for y in plot_keys for x in y]

        if not any(plot_keys): reportError(iam,"non empty plot key")
        return plot_keys


    def plotableData(self):
        iam = self.class_name + ".plotableData(self)"

        # return a list of the current values of the material data, in the order
        # registered
        plot_data = [0.]*len(self.material_data)
        for key, val in self.material_data.items():
            typ = val["type"]
            value = val["value"]
            idx = val["idx"]
            if typ == "scalar":
                plot_data[idx] = [value]

            elif typ == "vector":
                plot_data[idx] = value

            elif typ == "symtensor":
                plot_data[idx] = value

            elif typ == "tensor":
                plot_data[idx] = value

            else:
                reportError(iam,"unrecognized material data type {0}".format(typ))
                pass

            continue

        # flatten plot_data to just a list
        plot_data = [x for y in plot_data for x in y]

        return plot_data




    def legs(self):
        return self.lcontrol

    def initialTime(self):
        return self.t0

    def terminationTime(self):
        return self.tf

    def kappa(self):
        return self.bcontrol['kappa']

    def emit(self):
        return self.bcontrol['emit']

    def screenout(self):
        return self.bcontrol['screenout']

    def nprints(self):
        return self.bcontrol['nprints']

    def amplitude(self):
        return self.bcontrol['amplitude']

    def material(self):
        return self.mtl

    def constitutiveModel(self):
        return self.constitutive_model

    def diagonal(self):
        return self.isdiag

    def finish(self):
        closeFiles()
        del self.constitutive_model
        del self.material_data
        return

    def debug(self):
        return self.opts.debug

    def strict(self):
        return self.opts.strict

    def proportional(self):
        return self.opts.proportional

    def sqa(self):
        return self.opts.sqa

    def verbosity(self):
        return self.opts.verbosity

    def useTableVals(self):
        return self.opts.use_table

    def writeMaterialParameters(self):
        with open( self.simname + ".props", "w" ) as f:
            for i, key in enumerate(self.paramkeys):
                f.write("{0:s} = {1:12.5E}\n".format(key,self.mtl.cm.ui[i]))
                continue
            pass
        return

    def setupRestart(self):
        setupLogger(self.logfile,self.loglevel)
        msg = "setting up simulation %s"%self.simname
        reportMessage(__file__,msg)

if __name__ == "__main__":
    sys.exit("Payette_container.py must be called by runPayette")
