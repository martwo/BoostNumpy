.. highlight:: c

.. _BoostNumpy_dstream_mapping_converter_arg_type_to_core_shape:

arg_type_to_core_shape converter
================================

**namespace**: ``boost::numpy::dstream::mapping``

The ``converter::arg_type_to_core_shape`` converter is a MPL
function for converting a function's argument type to a core shape type, i.e.
a specialization of the ``dstream::detail::core_shape::nd<ND>::shape`` template.
The converter is used to automatically determine the core shape of one of the
input arrays based on the type of the input argument of the to-be-exposed C++
function, if no mapping definition had been specified for that function.

Pre-defined converters
----------------------

For some argument types converters already exists:

- ``detail::scalar_to_core_shape``

    Converts scalar types to ``dstream::detail::core_shape::nd<0>::shape<>``,
    i.e. a scalar array.

- ``detail::std_vector_of_scalar_to_core_shape``

    Converts ``std::vector< SCALAR_TYPE >`` types to
    ``dstream::detail::core_shape::nd<1>::shape< dstream::detail::core_shape::dim::I >``,
    i.e. to a 1D array with dimension I.

User defined converters
-----------------------

It is possible to define user defined ``arg_type_to_core_shape`` converter MPL
functions for a particular argument type, or a class of types. The example
below illustrates how to convert a ``std::vector< std::vector<double> >``
argument type to a 2D MxN core shape. ::

    #include <boost/numpy/dstream/mapping/converter/arg_type_to_core_shape_fwd.hpp>

    namespace boost {
    namespace numpy {
    namespace dstream {
    namespace mapping {
    namespace converter {

    namespace core_shape = dstream::detail::core_shape;
    namespace dim = dstream::detail::core_shape::dim;

    template <class T>
    struct arg_type_to_core_shape<T, typename enable_if< is_same< T, std::vector< std::vector<double> > > >::type>
    {
        typedef core_shape::nd<2>::shape<dim::M,dim::N>
                type;
    };

    }// namespace converter
    }// namespace mapping
    }// namespace dstream
    }// namespace numpy
    }// namespace boost
