
########################
Installing New Materials
########################

*Payette* was born from the need for an environment in which material
constitutive models could be rapidly developed, tested, deployed, and maintained,
independent of host finite element code implementation. Important prerequisites
in the design of *Payette* were ease of model installation and support for
constitutive routines written in Fortran. This requirement is met by providing a
simple API with which a model developer can install a material in *Payette* as
a new Python class and use `f2py <cens.ioc.ee/projects/f2py2e/>`_ to compile
Python extension modules from the material's Fortran source (if applicable). In
this section, the required elements of the constitutive model interface and
instructions on compiling the material's Fortran source (if applicable) are
provided.


Material File Naming Conventions
================================

Each material model must provide its Python class definition in an "interface"
file in the :file:`PAYETTE_ROOT/Source/Materials`. The accepted naming convention
for the material's interface file is::

    PAYETTE_ROOT/Source/Materials/Payette_material_name.py

A material's Fortran source code (if applicable) is stored in its own directory
in the :file:`PAYETTE_ROOT/Source/Materials/Fortran` directory. The accepted
naming convention for the material's Fortran directory is::

    PAYETTE_ROOT/Source/Materials/Fortran/MaterialName

There is no defined naming convention for Fortran source code within the
materials :file:`Fortran/MaterialName` directory.

As an example, suppose you were developing a new material model "Porous Rock".
The source files and directories would be named::

    PAYETTE_ROOT/Source/Materials/Payette_porous_rock.py
    PAYETTE_ROOT/Source/Materials/Fortran/PorousRock


.. note::

   The *Payette* project is an open source project, without restrictions on how
   the source code is used and/or distributed. However, many material models do
   have restrictive access controls and cannot, therefore, include their source
   files in *Payette*. This is the case for many materials currently being
   developed with *Payette*. For these materials, rather than include the source
   code in the material's Fortran directory, the Fortran build script used by
   *Payette* to build the material's Python extension module is used to direct
   *Payette* to the location of the material's source files, elsewhere in the
   developer's file system. The contents of the Fortran build script are
   discussed later in this document in the section on building Fortran source
   code in *Payette*.


Interface File Required Attributes
==================================

The interface file must provide an ``attributes`` with the following keys

.. data:: attributes["payette material"]

   Boolean.  Does the material represent an interface file, or not.

.. data:: attributes["name"]

   String.  The material name.

.. data:: attributes["fortran source"]

   Boolean.  Does the material have additional Fortran source code.

.. data:: attributes["build script"]

   String.  Absolute path to Fortran build script, if applicable.

.. data:: attributes["aliases"]

   List.  List of optional names.

.. data:: attributes["material type"]

   List. List of keyword descriptors of material type. Examples are "mechanical",
   "electro-mechanical".

.. note::

   Using the ``attributes`` dictionary outside of the material's class definition
   allows *Payette* to scan :file:`PAYETTE_ROOT/Source/Materials/` directory to
   find and import only the *Payette* material interface files during the build
   process.


Constitutive Model API: Required Elements
=========================================

*Payette* provides a simple interface for interacting with material models
through the Python class structure. Material models are installed as separate
Python classes, derived from the ``ConstitutiveModelPrototype`` base class.


Inheritance From Base Class
---------------------------

A new material model ``MaterialModel`` is only recognized as a material model by
Payette if it inherits from the ``ConstitutiveModelPrototype`` base class::

  class MaterialModel(ConstitutiveModelPrototype):


Required Data
-------------

.. data:: MaterialModel.aliases

   The aliases by which the constitutive model can be called (case insensitive).

.. data:: MaterialModel.bulk_modulus

   The bulk modulus.  Used for determining the material's Jacobian matrix

.. data:: MaterialModel.imported

   Boolean indicating whether the material's extension library (if applicable)
   was imported.

.. data:: MaterialModel.name

   The name by which users can invoke the constituve model from the input file
   (case insensitive).

.. data:: MaterialModel.nprop

   The number of required parameters for the model.

.. data:: MaterialModel.shear_modulus

   The shear modulus.  Used for determining the material's Jacobian matrix


Required Functions
------------------

.. function:: MaterialModel.__init__()

   Instantiate the material model.  Register parameters with *Payette*.


.. function:: MaterialModel.setUp(simdat,matdat,user_params,f_params)

   Check user inputs and register extra variables with *Payette*. *simdat* and
   *matdat* are the simulation and material data containers, respectively,
   *user_params* are the parameters read in from the input file, and *f_params*
   are parameters from a parameters file.

.. function:: MaterialModel.updateState(simdat,matdat)

   Update the material state to the end of the current time step. *simdat* and
   *matdat* are the simulation and material data containers, respectively.


Example: Elastic Material Model Interface File
==============================================

The required elements of the material's interface file described above are now
demonstrated by an annotated version of the elastic material's interface.

**View the source code:**
:download:`Payette_elastic.py <../../../Source/Materials/Payette_elastic.py>`

::

  import sys
  import os
  import numpy as np

  from Source.Payette_utils import *
  from Source.Payette_constitutive_model import ConstitutiveModelPrototype

.. note::

   The ``Source.Payette_utils`` module contains public methods for interfacing
   with *Payette*.

::

  attributes = {
      "payette material":True,
      "name":"elastic",
      "fortran source":True,
      "build script":os.path.join(Payette_Materials_Fortran,"Elastic/build.py"),
      "aliases":["hooke","elasticity"],
      "material type":["mechanical"]
      }

  try:
      import Source.Materials.Library.elastic as mtllib
      imported = True
  except:
      imported = False
      pass

.. note::

   We don't raise an exception just yet if the material's extension library is
   not importable. This allows users to run simulations even if all materials
   were not imported. Of course, if you try to run a simulation with a material
   that is not imported, an exception is raised.

::

  class Elastic(ConstitutiveModelPrototype):
      """
      CLASS NAME
         Elastic

      PURPOSE
         Constitutive model for an elastic material. When instantiated, the Elastic
         material initializes itself by first checking the user input
         (_check_props) and then initializing any internal state variables
         (_set_field). Then, at each timestep, the driver update the Material state
         by calling updateState.

      METHODS
         Private:
           _check_props
           _set_field

	 Public:
           setUp
           updateState

      FORTRAN
         The core code for the Elastic material is contained in
         Fortran/Elastic/elastic.f.  The module Library/elastic is created by f2py.
         elastic.f defines the following public subroutines

            hookechk: fortran data check routine called by _check_props
            hookerxv: fortran field initialization  routine called by _set_field
            hooke_incremental: fortran stress update called by updateState

         See the documentation in elastic.f for more information.

      AUTHORS
         Tim Fuller, Sandia National Laboratories, tjfulle@sandia.gov
      """
      def __init__(self):
          ConstitutiveModelPrototype.__init__(self)

.. note::

   The base ConstitutiveModelPrototype class must be initialized.

::

          self.name = attributes["name"]
          self.aliases = attributes["aliases"]
          self.imported = imported

.. note::

   The required elastic material data ``name``, ``aliases``, and ``imported`` are
   assigned from the interface files ``attributes`` dictionary and the file scope
   variable ``imported``.

.. note::

   Below, the elastic material's parameters are registered with *Payette*
   through the ``registerParameter`` function:

   .. function:: self.registerParameter(name, ui_loc, aliases=[])

      Register the parameter *name* with *Payette*. *ui_loc* is the integer
      location (starting at 0) of the parameter in the material's user input array.
      *aliases* are aliases by which the parameter can be specified in the input
      file.

::

          # register parameters
          self.registerParameter("LAM",0,aliases=[])
          self.registerParameter("G",1,aliases=['SHMOD'])
          self.registerParameter("E",2,aliases=['YMOD'])
          self.registerParameter("NU",3,aliases=['POISSONS'])
          self.registerParameter("K",4,aliases=['BKMOD'])
          self.registerParameter("H",5,aliases=[])
          self.registerParameter("KO",6,aliases=[])
          self.registerParameter("CL",7,aliases=[])
          self.registerParameter("CT",8,aliases=[])
          self.registerParameter("CO",9,aliases=[])
          self.registerParameter("CR",10,aliases=[])
          self.registerParameter("RHO",11,aliases=[])
          self.nprop = len(self.parameter_table.keys())
          self.ndc = 0
          pass

.. note::

   ``self.ndc`` is the number of derived constants.  This model has none.

::

    # Public methods
    def setUp(self,simdat,matdat,user_params,f_params):
        iam = self.name + ".setUp(self,material,props)"

        if not imported: return

        # parse parameters
        self.parseParameters(user_params,f_params)

.. note::

   ``parseParameters`` passes the user input read from the input file to the
   initial user input array ``self.UI0``. There is not return value.

::

        # check parameters
        self.dc = np.zeros(self.ndc)
        self.ui = self._check_props()
        self.nsv,namea,keya,sv,rdim,iadvct,itype = self._set_field()

.. note::

   ``_check_props`` and ``_set_field`` are private functions that check the user
   input and assign initial values to extra variables.

::

        namea = parseToken(self.nsv,namea)
        keya = parseToken(self.nsv,keya)

        # register the extra variables with the payette object
        matdat.registerExtraVariables(self.nsv,namea,keya,sv)

.. note::

   Above, the elastic material model registers its extra variables with the
   material data container.

   .. function:: DataContainer.registerExtraVariables(nxv,namea,keya,exinit)

      Register extra varaibles with the data container. *nxv* is the number of
      extra variables, *namea* and *keya* are ordered lists of extra variable
      names and plot keys, respectively, and *exinit* is a list of initial
      values.

::

        self.bulk_modulus,self.shear_modulus = self.ui[4],self.ui[1]
        pass


.. note::

   By default, *Payette* computes the material's Jacobian matrix numerically
   through a central difference algorithm. For some materials, like this elastic
   model, the Jacobian is constant. Here, we redefine the Jacobian to return the
   intial value.

::

    # redefine Jacobian to return initial jacobian
    def jacobian(self,simdat,matdat):
        if not imported: return
        v = simdat.getData("prescribed stress components")
        return self.J0[[[x] for x in v],v]

    def updateState(self,simdat,matdat):
        """
           update the material state based on current state and strain increment
        """
        if not imported: return

.. note::

   The *simdat* and *matdat* data containers contain all current data. Data is
   accessed by the ``DataContainer.getData(name)`` method.

::

        dt = simdat.getData("time step")
        d = simdat.getData("rate of deformation")
        sigold = matdat.getData("stress")
        svold = matdat.getData("extra variables")

        a = [dt,self.ui,sigold,d,svold,migError,migMessage]
        if not Payette_F2Py_Callback: a = a[:-2]
        sig, sv, usm = mtllib.hooke_incremental(*a)

.. note::

   ``hooke_incremental(*a)`` is a Fortran subroutine that performs the actual
   physics. Below, we store the update values of the extra variables and the
   stress.

::

        matdat.storeData("extra variables",sv)
        matdat.storeData("stress",sig)

        return


Building Material Fortran Extension Modules in *Payette*
==========================================================

.. note::

   This is not an exhaustive tutorial for how to link Python programs with
   compiled source code. Instead, it demonstrates through an annotated example
   the strategy that *Payette* uses to build and link with material models
   written in Fortran.

The strategy used in *Payette* to build and link to material models written in
Fortran is to use *f2py* to compile the Fortran source in to a shared object
library recognized by Python. The same task can be accomplished through Python's
built in `ctypes <http://docs.python.org/library/ctypes.html>`_, `weave
<http://www.scipy.org/Weave>`_\, or other methods. We have found that *f2py*
offers the most robust and easy to use solution. For more detailed examples of
how to use compiled libraries with Python see `Using Python as glue
<http://docs.scipy.org/doc/numpy/user/c-info.python-as-glue.html>`_ at the SciPy
website or `Using Compiled Code Interactively
<http://www.sagemath.org/doc/numerical_sage/using_compiled_code_iteractively.html>`_
on Sage's website.

Rather than provide an exhaustive tutorial on linking Python programs to compiled
libraries, we demonstrate how the ``elastic`` material model accomplishes this
task through annotated examples.


Creating the Elastic Material Signature File
--------------------------------------------

First, a Python signature file for the ``elatic`` material's Fortran source must
be created. A signature file is a Fortran 90 file that contains all of the
information that is needed to construct Python bindings to Fortran (or C)
functions.

For the elastic model, change to
:file:`PAYETTE_ROOT/Source/Materials/Fortran/Elastic` and execute

::

  % f2py -m elastic -h elastic.signature.pyf elastic.F

which will create the :file:`elastic.signature.pyf` signature file.

f2py will create a signature for every function in :file:`elastic.F`. However,
only three public functions need to be bound to our Python program. So, after
creating the signature file, all of the signatures for the private functions can
safely be removed.

The signature file can be modified even further. See the above links on how to
specialize your signature file for maximum speed and efficiency.

**View the elastic.signature.pyf file:** :download:`elastic.signauture.pyf
<../../../Source/Materials/Fortran/Elastic/elastic.signature.pyf>`


Elastic Material Build Script
-----------------------------

Materials are built by *f2py* through the ``MaterialBuilder`` class from which
each material derives its ``Build`` class. The ``Build`` class must provide a
``build_extension_module`` function, as shown below in the elastic material's
build script.

**View the elastic material build script:** :download:`build.py
<../../../Source/Materials/Fortran/Elastic/build.py>`

::

  import os,sys

  from Payette_config import *
  from Source.Payette_utils import BuildError
  from Source.Materials.Payette_build_material import MaterialBuilder

  class Build(MaterialBuilder):

      def __init__(self,name,libname,compiler_info):

          fdir,fnam = os.path.split(os.path.realpath(__file__))
          self.fdir, self.fnam = fdir, fnam

          # initialize base class
          MaterialBuilder.__init__(self,name,libname,fdir,compiler_info)

          pass

      def build_extension_module(self):

          # fortran files
          self.source_files = [os.path.join(self.fdir,x) for x in os.listdir(self.fdir)
                               if x.endswith(".F")]

          self.build_extension_module_with_f2py()

          return 0

.. note::

   For the elastic material, the ``build_extension_module`` function defines the
   Fortran source files and the calls the base class's
   ``build_extension_module_with_f2py`` function.

