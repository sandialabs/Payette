
############
Introduction
############

`Payette <http://www.eng.utah.edu/~timothyf>`_ is an object oriented
material model driver written in `Python <http://www.python.org>`_ designed
for rapid development and testing of material models. The core of the
*Payette* code base is written in Python, with the exception of many material
models and optimization routines that are written in Fortran and wrapped by
`*f2py* <http://www.scipy.org/F2py>`_.

*Payette* is free software released under the
`MIT License <http://www.wikipedia.org/wiki/MIT_License>`_


Why a Single Element Driver?
============================

Due to their complexity, it is often over kill to use a finite element code for
constitutive model development. In addition, features such as artificial
viscosity can mask the actual material response from constitutive model
development. Single element drivers allow the constituive model developer to
concentrate on model development and not the finite element response. Other
advantages of the *Payette* (or, more generally, of any stand-alone constitutive
model driver) are

  * *Payette* is a very small, special purpose, code. Thus, maintaining and
    adding new features to *Payette* is very easy.

  * Simulations are not affected by irrelevant artifacts such as artificial
    viscosity or uncertainty in the handling of boundary conditions.

  * It is straightforward to produce supplemental output for deep analysis of the
    results that would otherwise constitute an unnecessary overhead in a finite
    element code.

  * Specific material benchmarks may be developed and automatically run quickly
    any time the model is changed.

  * Specific features of a material model may be exercised easily by the model
    developer by prescribing strains, strain rates, stresses, stress rates, and
    deformation gradients as functions of time.

Why Python?
===========

Python is an interpreted, high level object oriented language. It allows for
writing programs rapidly and, because it is an interpreted language, does not
require a compiling step. While this might make programs written in python slower
than those written in a compiled language, modern packages and computers make the
speed up difference between python and a compiled language for single element
problems almost insignificant.

For numeric computations, the `NumPy <http://www.numpy.org>`_ and `SciPy
<http://www.scipy.org>`_ modules allow programs written in Python to leverage
a large set of numerical routines provided by ``LAPACK``, ``BLASPACK``,
``EIGPACK``, etc. Python's APIs also allow for calling subroutines written in
C or Fortran (in addition to a number of other languages), a prerequisite for
model development as most legacy material models are written in Fortran. In
fact, most modern material models are still written in Fortran to this day.

Python's object oriented nature allows for rapid installation of new material
models.


Historical Background
=====================

*Payette* is an outgrowth of Tom Pucick's *MMD* and `Rebecca Brannon's
<http://www.mech.utah.edu/~brannon/>`_ *MED* drivers. Both these other drivers
are written in Fortran.


Simulation Approach
===================

*Payette* exercises a material model directly by "driving" it through user
specified mechanical and electrical inputs.


Supported Drivers
-----------------

Mechanical
^^^^^^^^^^

**Direct**

* Strain rate
* Strain
* Deformation gradient
* Velocity
* Displacement

**Inverse**

* Stress
* Stress rate

Electrical
^^^^^^^^^^

**Direct**

* Electric field

