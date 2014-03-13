.. highlight:: c

.. _BoostNumpy_dstream_mapping_converter_return_type_to_out_mapping:

return_type_to_out_mapping converter
====================================

Namespace conventions::

    namespace mapping = boost::numpy::dstream::mapping;
    namespace converter = boost::numpy::dstream::mapping::converter;

The ``converter::return_type_to_out_mapping`` converter is a
MPL function and converts a function's return type to an output mapping type,
i.e. a specialization of the ``out<ND>::core_shapes`` template.
The converter is used to automatically determine the output mapping
for a given function's return type if no mapping definition had been specified
for the to-be-exposed C++ function.

Pre-defined converters
----------------------

For some return types converters already exists:

- ``converter::detail::void_to_out_mapping``

    Converts ``void`` type to ``out<0>::core_shapes<>``, i.e. no output.

- ``converter::detail::scalar_to_out_mapping``

    Converts scalar types to
    ``mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<0>::shape<> >``

- ``converter::detail::std_vector_of_scalar_to_out_mapping``

    Converts ``std::vector< SCALAR_TYPE >`` types to
    ``mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<1>::shape< dim::I > >``,
    i.e. to a 1D core shape with dimension name ``I``.

User defined converters
-----------------------

It is possible to define user defined ``return_type_to_out_mapping`` converter
meta-functions for a particular return type, or a class of types. The example
below illustrates how to convert a ``std::vector< std::vector<double> >`` return
type to an output mapping with one 2-dimensional MxN output array. ::

    #include <boost/numpy/dstream/mapping/converter/return_type_to_out_mapping_fwd.hpp>

    namespace boost {
    namespace numpy {
    namespace dstream {
    namespace mapping {
    namespace converter {

    template <class T>
    struct return_type_to_out_mapping<T, typename enable_if< is_same< T, std::vector< std::vector<double> > > >::type>
      : detail::return_type_to_out_mapping_type
    {
        typedef mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<2>::shape< dim::M, dim::N > >
                type;
    };

    }// namespace converter
    }// namespace mapping
    }// namespace dstream
    }// namespace numpy
    }// namespace boost
