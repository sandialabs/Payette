
#################
Running *Payette*
#################

Getting Started
===============

Interacting with *Payette* is done through the :file:`runPayette` script and
properly formatted input files. The basic usage of :file:`runPayette` is::

  % runPayette input_file

For a complete list of options for :file:`runPayette` execute::

  % runPayette -h


Simulation Output
=================

For a simulation titled ``simnam``, the following output is created by
:file:`runPayette`::

  simnam.log      (ascii log file)
  simnam.out      (ascii space delimited output file)
  simnam.math1    (ascii Mathematica auxiliary postprocessing file)
  simnam.math2    (ascii Mathematica auxiliary postprocessing file)
  simnam.prf      (binary restart file)
  simnam.props    (ascii list of checked material parameters)

