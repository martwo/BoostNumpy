.. highlight:: c

.. _BoostNumpy_dstream_wiring:

Wiring
======

Wiring is the process that connects the input and output values of the
to-be-exposed C++ (member) function to the core shape values of the input and
output arrays of the Python function. The wiring has to be implemented through a
so called *wiring model*. A wiring model loops over the loop dimensions and
implements the fundamental operation on the core dimensions of all the input and
output arrays.

A wiring model depends on a particular mapping definition and the return and
argument types of the to-be-exposed C++ (member) function.

In general, the user needs to provide (i.e. implement) a wiring model, but the
dstream library provides some default wiring models which should be suitable for
most cases (see section
:ref:`BoostNumpy_dstream_wiring_default_wiring_models`).


.. toctree::
   :maxdepth: 2

   default_wiring_models
