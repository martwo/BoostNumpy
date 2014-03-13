.. highlight:: c

.. _BoostNumpy_dstream_wiring_default_wiring_models:

Default Wiring Models
=====================

The dstream library provides default wiring models for certain classes of
mapping definitions and C++ (member) function signatures.

scalars_to_scalar_callable
--------------------------

**Mapping definition prerequisites:**

    * All input arrays are scalar arrays.
    * The output is a scalar array or is none.

**Function signature prerequisites:**

    * All argument types are scalar types, i.e. ``boost::is_scalar<T>`` derives
      from ``boost::mpl::true_`` for type ``T``.
    * The return type is either a scalar type or ``void``.

This wiring model does a one-to-one wiring of the input and output values
between the C++ function and the Python GUF.

scalars_to_vector_of_scalar_callable
------------------------------------

**Mapping definition prerequisites:**

    * All input arrays are scalar arrays.
    * The output mapping is either one 1-dimensional array or multiple scalar
      arrays.

**Function signature prerequisites:**

    * All argument types are scalar types.
    * The return type is a ``std::vector`` of a scalar type.

This wiring model does a one-to-one wiring of the input values. If the output
mapping consists of one 1-dimensional array, all values of the returned vector
will be put into that array. If the output mapping consists of multiple scalar
arrays, each value of the returned vector will be put into the corresponding
scalar array and the GUF returns a Python tuple of all these scalar arrays.

.. note::

    Due to the fact, that ``std::vector`` does not define its length at compile
    time, always a mapping definition for the to-be-exposed C++ (member)
    function needs to be specified explicitly by the user.
