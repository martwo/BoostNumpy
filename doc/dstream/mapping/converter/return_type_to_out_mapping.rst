.. highlight:: c

.. _BoostNumpy_dstream_mapping_converter_return_type_to_out_mapping:

return_type_to_out_mapping converter
====================================

The ``dstream::mapping::converter::return_type_to_out_mapping`` converter is a
MPL function and converts a function's return type to an output mapping type,
i.e. a specialization of the ``dstream::mapping::out<ND>::core_shapes``
template. The converter is used to automatically determine the output mapping
for a given function's return type if no mapping definition has been specified
for the to-be-exposed function.

User defined converters
-----------------------

It is possible to define user defined ``return_type_to_out_mapping`` converter
MPL functions for special return types T. The example below illustrates how to
convert a ``std::vector< std::vector<double> >`` return type to an output mapping
with one 2D MxN output array. ::

    #include <boost/numpy/dstream/mapping/converter/return_type_to_out_mapping_fwd.hpp>

    namespace boost {
    namespace numpy {
    namespace dstream {
    namespace mapping {
    namespace converter {

    namespace core_shape = dstream::detail::core_shape;
    namespace dim = dstream::detail::core_shape::dim;

    template <class T>
    struct return_type_to_out_mapping< T, typename enable_if< is_same< T, std::vector< std::vector<double> > > >::type >
    {
        typedef out<1>::core_shapes< core_shape::nd<2>::shape<dim::M,dim::N> >
                type;
    };

    }// namespace converter
    }// namespace mapping
    }// namespace dstream
    }// namespace numpy
    }// namespace boost
