
########################
Installing New Materials
########################

*Payette* was born from the need for an environment in which material models
could be rapidly developed, tested, deployed, and maintained, independent of
host finite element code implementation. Important prerequisites in the design
of *Payette* were ease of model installation and support for constitutive
routines written in Fortran. This requirement is met by providing a simple API
with which a model developer can install a material in *Payette* as a new
Python class and use `f2py <http://www.scipy.org/F2py>`_ to compile Python
extension modules from the material's Fortran source (if applicable). In this
section, the required elements of the constitutive model interface and
instructions on compiling the material's Fortran source (if applicable) are
provided.


Build Process
=============

The ``buildPayette`` script, described in :ref:`installation`, scans the
``PAYETTE_ROOT/Source/Materials`` directory for *material interface module*
and, when found, processes the build directives contained therein. A
material's interface module is the link between it and *Payette*.

.. note::

   *Payette* is an open source project, without restrictions on how the source
   code is used and/or distributed. However, many material models have
   restrictive access controls and cannot, therefore, be included in
   *Payette*. This is the case for many materials currently being developed
   with *Payette*. For these materials, *Payette* provides two methods for
   extending the list of directories it scans for material interface modules:

   #) ``PAYETTE_MTLDIR`` environment variable. A colon separated list of
      directories containing *Payette* interface modules.

   #) ``--mtldir`` :file:`configure.py` option. At the configuration step,
      directories containing *Payette* interface modules can be passed to
      *Payette* through the ``--mtldir=/path/to/material/directory`` option.


Naming Convention
=================

In general, a material provides, in addition to its interface module, a build
module and, if applicable, source and source header files. For default
materials included with *Payette*, the following naming convention has been
adopted for a material "material model":

Interface file
  :file:`PAYETTE_ROOT/Materials/Payette_material_model.py`

Build script
  :file:`PAYETTE_ROOT/Materials/Build_material_model.py`

Fortran source
  :file:`PAYETTE_ROOT/Materials/Fortran/material_model.f`

Header file
  :file:`PAYETTE_ROOT/Materials/Include/material_model.h`


Material Interface Module
=========================

Each material model must provide an interface module used by *Payette* to
interact with that material. The interface module must provide 1) a
``attributes`` attribute and 2) a material class derived from the
``ConstitutiveModelPrototype`` base class.

``attributes`` Attribute
------------------------

``attributes`` is a dictionary a dictionary used by *Payette* during the build
step and contains the following ``key:value`` pairs:

| ``attributes["`` **payette material** ``"]``
|   Boolean.  Required.  Does the material represent an interface file, or not.
|
| ``attributes["`` **name** ``"]``
|   String.  Required.  The material name.
|
| ``attributes["`` **fortran source** ``"]``
|   Boolean.  Optional.  Does the material have additional Fortran source code.
|   Default: ``False``
|
| ``attributes["`` **build script** ``"]``
|   String.  Optional.  Absolute path to Fortran build script, if applicable.
|   Default: ``None``
|
| ``attributes["`` **aliases** ``"]``
|   List.  Optional.  List of aliases.
|   Default: ``[]``
|
| ``attributes["`` **material type** ``"]``
|   List.  Optional.  List of keyword descriptors of material type. Examples are
|   "mechanical", "electro-mechanical".
|   Default: ``["mechanical"]``



Material Class
--------------

*Payette* provides a simple interface for interacting with material models
through the Python class structure. Material models are installed as separate
Python classes, derived from the ``ConstitutiveModelPrototype`` base class.


Inheritance From Base Class
"""""""""""""""""""""""""""

A new material model "material model" is only recognized as a material model
by Payette if it inherits from the ``ConstitutiveModelPrototype`` base class::

  class MaterialModel(ConstitutiveModelPrototype):


The ``ConstitutiveModelPrototype`` base class provides several methods in its
API for material models to communicate with *Payette*. Minimally, the material
model must provide the following data: ``aliases``, ``bulk_modulus``,
``imported``, ``name``, ``nprop``, and ``shear_modulus``, and methods:
``__init__``, ``setUp``, and ``updateState``.

Required Data
"""""""""""""

.. data:: MaterialModel.aliases

   List. The aliases by which the constitutive model can be called (case
   insensitive).

.. data:: MaterialModel.bulk_modulus

   Float. The bulk modulus. Used for determining the material's Jacobian matrix

.. data:: MaterialModel.imported

   Boolean. Indicator of whether the material's extension library (if
   applicable) was imported.

.. data:: MaterialModel.name

   String. The name by which users can invoke the constituve model from the
   input file (case insensitive).

.. data:: MaterialModel.nprop

   Int. The number of required parameters for the model.

.. data:: MaterialModel.shear_modulus

   Float. The shear modulus. Used for determining the material's Jacobian
   matrix


Required Functions
------------------

``__init__(self)``

   Instantiate the material model. Register parameters with *Payette*.
   Parameters are registered by the ``registerParameter`` method

   ::

     registerParameter(self, name, ui_loc, aliases=[])
         """Register the parameter name with Payette.

         ui_loc is the integer location (starting at 0) of the parameter in
         the material's user input array. aliases are aliases by which the
         parameter can be specified in the input file."""

``setUp(self, simdat, matdat, user_params, f_params)``

   Check user inputs and register extra variables with *Payette*. *simdat* and
   *matdat* are the simulation and material data containers, respectively,
   *user_params* are the parameters read in from the input file, and *f_params*
   are parameters from a parameters file.

``updateState(self, simdat, matdat)``

   Update the material state to the end of the current time step. *simdat* and
   *matdat* are the simulation and material data containers, respectively.


Example: Elastic Material Model Interface File
==============================================

The required elements of the material's interface file described above are now
demonstrated by an annotated version of the elastic material's interface.

**View the source code:** :download:`Payette_elastic.py
<../../../Source/Materials/Models/Elastic/Payette_elastic.py>`

::

  import sys
  import os
  import numpy as np

  from Source.Payette_utils import *
  from Source.Payette_constitutive_model import ConstitutiveModelPrototype

  try:
      import Source.Materials.Library.elastic as mtllib
      imported = True
  except ImportError:
      imported = False

  from Payette_config import PC_MTLS_FORTRAN, PC_F2PY_CALLBACK

  THIS_DIR = os.path.dirname(os.path.realpath(__file__))
  attributes = {
      "payette material": True,
      "name": "elastic",
      "fortran source": True,
      "build script": os.path.join(THIS_DIR, "Build_elastic.py"),
      "aliases": ["hooke", "elasticity", "linear elastic"],
      "material type": ["mechanical"],
      "default material": True,
      }

  class Elastic(ConstitutiveModelPrototype):
      def __init__(self):
          super(Elastic, self).__init__()

          # register parameters
          self.registerParameter("LAM", 0, aliases=[])
          self.registerParameter("G", 1, aliases=["SHMOD"])
          self.registerParameter("E", 2, aliases=["YMOD"])
          self.registerParameter("NU", 3, aliases=["POISSONS"])
          self.registerParameter("K", 4, aliases=["BKMOD"])
          self.registerParameter("H", 5, aliases=[])
          self.registerParameter("KO", 6, aliases=[])
          self.registerParameter("CL", 7, aliases=[])
          self.registerParameter("CT", 8, aliases=[])
          self.registerParameter("CO", 9, aliases=[])
          self.registerParameter("CR", 10, aliases=[])
          self.registerParameter("RHO", 11, aliases=[])
          self.nprop = len(self.parameter_table.keys())
          self.ndc = 0
          pass


      # Public methods
      def setUp(self, simdat, matdat, user_params, f_params):
          iam = self.name + ".setUp(self, material, props)"

          if not imported:
	      return

          # parse parameters
          self.parseParameters(user_params, f_params)

          # check parameters
          self.dc = np.zeros(self.ndc)
          self.ui = self._check_props()
          self.nsv,namea,keya,sv,rdim,iadvct,itype = self._set_field()

          namea = parseToken(self.nsv, namea)
          keya = parseToken(self.nsv, keya)

          # register the extra variables with the payette object
          matdat.registerExtraVariables(self.nsv, namea, keya, sv)

          self.bulk_modulus, self.shear_modulus = self.ui[4], self.ui[1]
          pass


      # redefine Jacobian to return initial jacobian
      def jacobian(self, simdat, matdat):
          if not imported:
	      return
          v = simdat.getData("prescribed stress components")
          return self.J0[[[x] for x in v], v]

      def updateState(self, simdat, matdat):
          """
             update the material state based on current state and strain increment
          """
          if not imported:
	      return

          dt = simdat.getData("time step")
          d = simdat.getData("rate of deformation")
          sigold = matdat.getData("stress")
          svold = matdat.getData("extra variables")

          a = [dt, self. ui, sigold, d, svold, migError, migMessage]
          if not Payette_F2Py_Callback:
	      a = a[:-2]
          sig, sv, usm = mtllib.hooke_incremental(*a)


          matdat.storeData("extra variables", sv)
          matdat.storeData("stress", sig)

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
:file:`PAYETTE_ROOT/Source/Materials/Fortran` and execute

::

  % f2py -m elastic -h Payette_elastic.pyf elastic.F

which will create the :file:`Payette_elastic.pyf` signature file.

f2py will create a signature for every function in :file:`elastic.F`. However,
only three public functions need to be bound to our Python program. So, after
creating the signature file, all of the signatures for the private functions can
safely be removed.

The signature file can be modified even further. See the above links on how to
specialize your signature file for maximum speed and efficiency.

**View the Payette_elastic.pyf file:** :download:`Payette_elastic.pyf
<../../../Source/Materials/Models/Elastic/Payette_elastic.pyf>`


Elastic Material Build Script
-----------------------------

Materials are built by *f2py* through the ``MaterialBuilder`` class from which
each material derives its ``Build`` class. The ``Build`` class must provide a
``build_extension_module`` function, as shown below in the elastic material's
build script.

**View the elastic material build script:** :download:`Build_elastic.py
<../../../Source/Materials/Models/Elastic/Build_elastic.py>`

::

  import os,sys

  from Payette_config import *
  from Source.Payette_utils import BuildError
  from Source.Materials.Payette_build_material import MaterialBuilder

  class Build(MaterialBuilder):

      def __init__(self, name, libname, compiler_info):

          fdir,fnam = os.path.split(os.path.realpath(__file__))
          self.fdir, self.fnam = fdir, fnam

          # initialize base class
          srcd = os.path.join(fdir, "Fortran")
          sigf = os.path.join(fdir, "Payette_elastic.pyf")
          MaterialBuilder.__init__(
              self, name, libname, srcd, compiler_info, sigf=sigf)


      def build_extension_module(self):

          # fortran files
          srcs = ["elastic.F"]
          self.source_files = [os.path.join(self.source_directory, x)
          for x in srcs]
          self.build_extension_module_with_f2py()

          return 0

.. note::

   For the elastic material, the ``build_extension_module`` function defines the
   Fortran source files and the calls the base class's
   ``build_extension_module_with_f2py`` function.

