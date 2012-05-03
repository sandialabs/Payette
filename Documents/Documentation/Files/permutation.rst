###########
Permutation
###########

*Payette* has the ability to assist in understanding the sensitivity of a
material model to user inputs by permutating selecting user inputs and running
a simulation with the permutated inputs. Below is a summary of the
permutation block.

Permutation Block
=================

Input for the permutation problem is done in the ``permutation`` block. We
specify every available option showing its default value in brackets, along
with other available options in braces, if applicable.


Example 1
---------

Zip a range of bulk and shear moduli.

::

  begin permutation

    method zip

    permutate K, range = (125.e9, 150.e9, 10) # parameter to permutate
    permutate G, range = (45.e9, 57.e9, 10)  # parameter to permutate

  end permutation

Example 2
---------

Combine an unequal length range of bulk and shear moduli.

::

  begin permutation

    method combination

    permutate K, range = (125.e9, 150.e9, 5) # parameter to permutate
    permutate G, range = (45.e9, 57.e9, 10)  # parameter to permutate

  end permutation

Example 3
---------

Zip a sequence of bulk and shear moduli.

::

  begin permutation

    method zip

    permutate K, sequence = (125.e9, 145.e9, 150.e9)
    permutate G, sequence = (45.e9, 50.e9, 57.e9)

  end permutation

Example 4
---------

Combine sequence of bulk and shear moduli.

::

  begin permutation

    method combination

    permutate K, sequence = (125.e9, 150.e9)
    permutate G, sequence = (45.e9, 50.e9, 57.e9)

  end permutation


Methods
-------

**zip** (default)
  Zip together ranges and/or sequences of permutated parameters. Length of
  each parameter's permutation list must be the same.

**combination**
  Combine ranges and/or sequences of permutated parameters in to all possible
  combinations.



