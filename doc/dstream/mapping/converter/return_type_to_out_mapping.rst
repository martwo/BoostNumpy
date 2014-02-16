.. highlight:: c

.. _BoostNumpy_dstream_mapping_converter_return_type_to_out_mapping:

return_type_to_out_mapping converter
====================================

**namespace**: ``boost::numpy::dstream::mapping``

The ``converter::return_type_to_out_mapping`` converter is a
MPL function and converts a function's return type to an output mapping type,
i.e. a specialization of the ``out<ND>::core_shapes``
template. The converter is used to automatically determine the output mapping
for a given function's return type if no mapping definition had been specified
for the to-be-exposed C++ function.

Pre-defined converters
----------------------

For some return types converters already exists:

- ``converter::detail::void_to_out_mapping``

    Converts ``void`` type to ``out<0>::core_shapes<>``, i.e. no output.

- ``converter::detail::scalar_to_out_mapping``

    Converts scalar types to
    ``out<1>::core_shapes< dstream::detail::core_shape::nd<0>::shape<> >``

- ``converter::detail::std_vector_of_scalar_to_out_mapping``

    Converts ``std::vector< SCALAR_TYPE >`` types to
    ``out<1>::core_shapes< dstream::detail::core_shape::nd<1>::shape< dstream::detail::core_shape::dim::I > >``,
    i.e. to a 1D core shape with dimension I.

User defined converters
-----------------------

It is possible to define user defined ``return_type_to_out_mapping`` converter
MPL functions for a particular return type, or a class of types. The example
below illustrates how to convert a ``std::vector< std::vector<double> >`` return
type to an output mapping with one 2D MxN output array. ::

    #include <boost/numpy/dstream/mapping/converter/return_type_to_out_mapping_fwd.hpp>

    namespace boost {
    namespace numpy {
    namespace dstream {
    namespace mapping {
    namespace converter {

    namespace core_shape = dstream::detail::core_shape;
    namespace dim = dstream::detail::core_shape::dim;

    template <class T>
    struct return_type_to_out_mapping<T, typename enable_if< is_same< T, std::vector< std::vector<double> > > >::type>
      :
    {
        typedef out<1>::core_shapes< core_shape::nd<2>::shape<dim::M,dim::N> >
                type;
    };

    }// namespace converter
    }// namespace mapping
    }// namespace dstream
    }// namespace numpy
    }// namespace boost
