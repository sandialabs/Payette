.. highlight:: rst

########
Building
########

Payette is an object oriented material driver. The majority of the source code is
written in Python and requires no additional building. Many of the material
models, however, are written in Fortran and require a seperate compile step.


System Requirements
===================

Payette has been built and tested extensively on several versions linux and the
Apple Mac OSX 10.6 operating systems. It is unknown whether it will run on
Windows.


Required Software
=================

Payette requires the following software installed for your platform:

#) `Python 2.6 <http://www.python.org/>`_ or newer

#) `NumPy 1.5 <http://www.numpy.org/>`_ or newer

#) `SciPy 0.1 <http://www.scipy.org/>`_ or newer

The required software may be obtained in one of several ways:

    * *Install from your systems package manager*

      ``apt-get`` on Ubuntu, ``yum`` on Fedora, ``rpm`` on Red Hat, etc. On the
      Mac, we have had great success using software installed by `MacPorts
      <http://www.macports.org>`_.

      .. warning::

         We have had varying degrees of success using software installed by
         package managers on various linux distributions and have had to resort
         to building from source or using Sage, as described below.

    * *Build each from source*

      An involved, but fairly straight forward process.

    * *Use Python, ``NumPy``, and ``SciPy`` installed with*
      `Sage 4.7 <http://www.sagemath.org/>`_ *or newer*

      This option is perhaps the easiest, since Sage provides prebuilt binaries
      for the Mac and many linux platforms. Their build process has also proven
      to be very robust on many platforms.

      .. note::

         * Some features are disabled when using Sage. In particular, callback
      	   functions from Fortran subroutines to Python functions.

	 * Using Sage is not as extensively tested as using a seperate Python
           installation.



Installation
============

#) Make sure that all Payette prerequisites are installed and working properly.

#) Add ``PAYETTE_ROOT`` to your ``PYTHONPATH`` environment variable

#) Add ``PAYETTE_ROOT/Toolset`` to your ``PATH`` environment variable

#) Change to ``PAYETTE_ROOT`` and run::

        % PYTHON configure.py

   where ``PYTHON`` is the python interpreter that has ``NumPy`` and ``SciPy``
   installed. If using Sage, replace ``PYTHON`` with ``sage -python``.

   :file:`configure.py` will write the Payette configuration file and the following
   executable scripts::

       PAYETTE_ROOT
         Payette_config.py
         Toolset/
           buildPayette
           cleanPayette
           runPayette
           testPayette

5) execute::

	% buildPayette

   which will build the Payette material libraries and create a configuration
   file of all built and installed materials in
   :file:`PAYETTE_ROOT/Source/Materials/Payette_installed_materials.py`


Testing the Installation
========================

To test Payette after installation, execute::

	% testPayette -k regression -k fast

which will run the "fast" "regression" tests. To run the full test suite execute::

	% testPayette -j4

Please note that running all of the tests takes several minutes.


Known Issues
============

#) *callbacks*

   A callback function is a Python function accessible by Fortran subroutines
   through a callback mechanism provided by ``f2py``. The ``NumPy`` and ``SciPy``
   packages distributed by many of the different linux distributions package
   managers have broken dependencies. In particular, callback functions seem to
   be broken on many linux systems. The easy work around is to configure Payette
   with --no-callback. Another work around is passing different fortran compilers
   to configure.py (--f77exec= , --f90exec= ) and seeing if that makes a
   difference.

#) *segfault*

   A segfault error is usually a result of a broken callback.  See 0) above.

#) *Unable to build*

   Difficulty building Payette is usually the result of broken ``NumPy`` and
   ``SciPy`` installations and the workaround involves reinstalling all software
   packages from sourc. If you are uncomfortable installing these software
   packages from source, consider using Sage to build and run Payette.


Troubleshooting
===============

If you experience problems when building/installing/testing Payette, you can ask
help from `Tim Fuller <tjfulle@sandia.gov>`_ or `Scot Swan <mswan@sandia.gov>`_.
Please include the following information in your message:

#) Are you using Sage, or not

#) Platform information OS, its distribution name and version information etc.::

        % PYTHON -c 'import os,sys;print os.name,sys.platform'
	% uname -a


#) Information about C,C++,Fortran compilers/linkers as reported by
   the compilers when requesting their version information, e.g.,
   the output of::

        % gcc -v
        % gfortran --version

#) Python version::

        % PYTHON -c 'import sys;print sys.version'

#) ``NumPy`` version::

        % PYTHON -c 'import numpy;print numpy.__version__'

#) ``SciPy`` version::

        % PYTHON -c 'import scipy;print scipy.__version__'

#) The contents of the :file:`PAYETTE_ROOT/Payette_config.py` file

#) Feel free to add any other relevant information.

