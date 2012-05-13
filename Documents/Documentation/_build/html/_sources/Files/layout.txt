
###############
Developer Guide
###############

*Payette* Directory Structure
=============================

The ``Payette`` project has the following directory structure

::

  PAYETTE_ROOT/
      configure.py
    COPYING
    __init__.py
    README

    Aux/
      Inputs/

    Benchmarks/
      Materials/
      Regression/

    MaterialsDatabase/

    Documents/
      Documentation/
      Presentations/

    Examples/

    Source/
      Fortran/
      Materials/
        Fortran/
        Library/

    Toolset/

Coding Style Guide
==================

All new code in *Payette* should adhere sctrictly to the style described in
Python's `PEP-0008 <http://www.python.org/dev/peps/pep-0008/>`_ and docstrings
should follow NumPy's `docstring guide
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_. One
exception to the Python style guide is the naming convention of files in
*Payette*: all file names begin with :file:`Payette_`.

It is highly recommended to use tools such as *pylint* and *pep8* to perform
code analysis on all new and existing code.

.. note:: Many of the original files in *Payette* were not written to the
   Python style guide. As time allows, these files are being modified to
   adhere to the standards.
