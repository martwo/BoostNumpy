.. highlight:: c

.. _BoostNumpy_dstream:

dstream: Generalized Universal Function Library
===============================================

The dstream library provides a facility to automatically expose C++ (member)
functions to Python as generalized universal functions (GUFs).

GUFs were proposed by the numpy community [#guf]_. They introduce the concept of
core and loop dimensions for a numpy multi-dimensional array (ndarray). The core
dimensions are always the inner-most dimensions of a ndarray

The fundamental GUF operation is defined on the core dimensions of the input
and output arrays of that function. The operation is then performed for all
elements of the loop dimensions.

A GUF has always a mapping (aka signature) that defines the core dimensions of
all input and output arrays. boost::numpy::dstream is able to generate the
mapping for a given C++ (member) function automatically, based on the argument
and return types of that C++ function (see section
:ref:`BoostNumpy_dstream_mapping`).

After the mapping has been fixed, a *wiring model* has to be chosen. The wiring
model implements the loop of the loop dimensions and the fundamental operation
on the core dimensions of all the input and output arrays.

In all the examples below, the following namespace conventions are used::

    namespace ds = boost::numpy::dstream;

.. toctree::
   :maxdepth: 3

   mapping/index
   wiring/index
   exposing
   threading

.. [#guf] http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
