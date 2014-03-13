.. highlight:: c

.. _BoostNumpy_dstream_mapping:

Mapping
=======

Mapping defines the core shapes of all input and output arrays of a generalized
universal function (GUF).

Possible mapping definitions could look like::

    namespace dim = ds::dim;

    // Scalar to scalar mapping.
    ( ds::scalar() >> ds::scalar() )

    // Inner product mapping.
    ( (ds::array<dim::N>(), ds::array<dim::N>()) >> ds::scalar() )

    // Matrix multiplication.
    ( (ds::array<dim::M, dim::N>(), ds::array<dim::N, dim::P>()) >> ds::array<dim::M, dim::P>() )

As shown above, the dstream library provides the scalar class and the array
template for specifying the mapping definition of a GUF. The array template
takes the names (actually integers) of the core dimensions of the input/output
array. Dimensions with the same name must have the same length or must be
broadcast-able to each other. The dimension names ``A`` to ``Z`` are defined in
namespace ``ds::dim``.

Fixed-length dimensions are also possible by specifying a positive integer for
a particular array core dimension::

    ( ds::array<3>() >> ds::scalar() )

Automatic mapping is performed, when no mapping definition has been specified
by the user. In such cases, the mapping is determined from the argument and
return types of the to-be-exposed C++ (member) function. The concept of those
converters are described in detail in section
:ref:`BoostNumpy_dstream_mapping_converter`.

.. toctree::
   :maxdepth: 3

   converter/index
