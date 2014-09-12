/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/return_to_core_shape_data.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the return_to_core_shape_data template that should
 *        put a function's return value into the output arrays defined by the
 *        out mapping type and its core shapes.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_WIRING_RETURN_TO_CORE_SHAPE_DATA_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_RETURN_TO_CORE_SHAPE_DATA_HPP_INCLUDED

#include <stdint.h>

#include <iostream>
#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/comparison/greater.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <boost/assert.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/python/refcount.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/detail/utils.hpp>
#include <boost/numpy/dstream/mapping/detail/definition.hpp>
#include <boost/numpy/dstream/wiring/detail/iter_data_ptr.hpp>
#include <boost/numpy/dstream/wiring/detail/nd_accessor.hpp>
#include <boost/numpy/dstream/wiring/detail/utilities.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace converter {

template <class WiringModelAPI, class OutMapping, class RT, class Enable=void>
struct return_to_core_shape_data
{
    typedef return_to_core_shape_data<WiringModelAPI, OutMapping, RT, Enable>
            type;

    // The return_to_core_shape_data needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_return_to_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_WIRING_MODEL_API_AND_FUNCTION_RETURN_TYPE_RT_AND_OUT_MAPPING_TYPE_OutMapping, (WiringModelAPI, RT, OutMapping));
};

namespace detail {

template <class VectorT, unsigned axis>
struct multidim_std_vector_has_fixed_length_axis;

// Define specializations for axis = 1 .. 17 for multi-dimensional vectors of
// up to 18 dimensions, i.e. dim::I to dim::Z.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, 17, <boost/numpy/dstream/wiring/converter/return_to_core_shape_data.hpp>, 1))
#include BOOST_PP_ITERATE()

//------------------------------------------------------------------------------
// The scalar_return_to_core_shape_data_impl template is used to put the
// function's scalar result data into the one and only output array.
// This makes only sense for scalar out mappings, which is ensured already by
// the select_scalar_return_to_core_shape_data_impl metafunction.
template <class WiringModelAPI, class OutMapping, class RT>
struct scalar_return_to_core_shape_data_impl
{
    typedef scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT>
            type;

    typedef typename WiringModelAPI::template out_arr_value_type<0>::type
            out_arr_value_t;

    scalar_return_to_core_shape_data_impl(
        numpy::detail::iter &                        iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
      : iter_(iter)
    {
        BOOST_ASSERT((out_core_shapes.size()    == 1 &&
                      out_core_shapes[0].size() == 0));
    }

    inline
    bool
    operator()(RT result)
    {
        out_arr_value_t & out_arr_value = *reinterpret_cast<out_arr_value_t *>(iter_.get_data(0));
        out_arr_value = out_arr_value_t(result);

        return true;
    }

    numpy::detail::iter & iter_;
};

template <class WiringModelAPI, class OutMapping, class RT>
struct select_scalar_return_to_core_shape_data_impl
{
    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    // Check if the output arity is 1.
    typedef typename out_mapping_utils::template arity_is_equal_to<1>::type
            is_unary_out_mapping;

    // Check if the output array has a scalar core shape.
    typedef typename out_mapping_utils::template array<0>::is_scalar::type
            is_scalar_out_array;

    // Check if the output array has a scalar data holding type.
    typedef typename is_scalar<typename WiringModelAPI::template out_arr_value_type<0>::type>::type
            is_scalar_out_array_data_type;

    typedef typename boost::mpl::if_<
              typename boost::mpl::and_<
                is_unary_out_mapping
              , is_scalar_out_array
              , is_scalar_out_array_data_type
              >::type
            , scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT>

            , numpy::mpl::unspecified
            >::type
            type;
};

//------------------------------------------------------------------------------
template <class WiringModelAPI, class OutMapping, class RT>
struct bp_object_return_to_core_shape_data_impl
{
    typedef bp_object_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT>
            type;

    bp_object_return_to_core_shape_data_impl(
        numpy::detail::iter &                        iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
      : iter_(iter)
    {
        BOOST_ASSERT((out_core_shapes.size()    == 1 &&
                      out_core_shapes[0].size() == 0));
    }

    inline
    bool
    operator()(RT const & obj)
    {
        uintptr_t * ptr_value_ptr = reinterpret_cast<uintptr_t *>(iter_.get_data(0));
        // Increment the reference counter for the bp object so it does not get
        // destroyed when the bp::object object gets out of scope.
        *ptr_value_ptr = reinterpret_cast<uintptr_t>(python::xincref<PyObject>(obj.ptr()));

        return true;
    }

    numpy::detail::iter & iter_;
};

template <class WiringModelAPI, class OutMapping, class RT>
struct select_bp_object_return_to_core_shape_data_impl
{
    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    // Check if the output arity is 1.
    typedef typename out_mapping_utils::template arity_is_equal_to<1>::type
            is_unary_out_mapping;

    // Check if the output array has a scalar core shape.
    typedef typename out_mapping_utils::template array<0>::is_scalar::type
            is_scalar_out_array;

    // Check if the output array has a bp::object data holding type.
    typedef typename is_same<typename WiringModelAPI::template out_arr_value_type<0>::type, python::object>::type
            is_bp_object_out_array_data_type;

    typedef typename boost::mpl::if_<
              typename boost::mpl::and_<
                is_unary_out_mapping
              , is_scalar_out_array
              , is_bp_object_out_array_data_type
              >::type
            , bp_object_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT>

            , numpy::mpl::unspecified
            >::type
            type;
};

//------------------------------------------------------------------------------
template <class VectorT, unsigned nd>
struct get_multidim_std_vector_shape;

// The std_vector_of_scalar_return_to_core_shape_data_impl template is used to
// put the function's result data from a n-dimensional std::vector of scalar
// type into out_arity output arrays.
template <class WiringModelAPI, class OutMapping, class RT, class VectorValueT, unsigned nd, unsigned out_arity>
struct std_vector_of_scalar_return_to_core_shape_data_impl;

// Define specializations for dimensions I to Z, i.e. 18 dimensions for
// out_arity = 1.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, 18, <boost/numpy/dstream/wiring/converter/return_to_core_shape_data.hpp>, 2))
#include BOOST_PP_ITERATE()

// Define specializations for dimensions I to Z, i.e. 18 dimensions for
// out_arity >= 2.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (2, BOOST_NUMPY_LIMIT_OUTPUT_ARITY, <boost/numpy/dstream/wiring/converter/return_to_core_shape_data.hpp>, 3))
#include BOOST_PP_ITERATE()

template <class WiringModelAPI, class OutMapping, class RT, class VectorValueT, unsigned nd, unsigned out_arity>
struct select_std_vector_of_scalar_return_to_core_shape_data_impl
{
    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    // At this point we know that RT is a nd-dimensional std::vector of scalar
    // and that out_arity is greater than 1 (because out_arity=1 is specialized
    // below). Thus, we will distribute the first dimension of the
    // nd-dimensional vector over the different output arrays.

    // First, we need to check if all the output arrays have the same
    // dimensionality of nd-1, because we will distribute the first axis of
    // the nd-dimensional result vector to the out_arity output arrays.
    typedef typename out_mapping_utils::template all_arrays_have_dim<nd-1>::type
            all_arrays_have_correct_dim;

    // Second, we need to check if all the output arrays have a scalar data
    // holding type.
    typedef typename wiring::detail::utilities<WiringModelAPI>::template all_out_arr_value_types<boost::is_scalar>::type
            all_out_arr_value_types_are_scalars;

    typedef typename boost::mpl::if_<
              typename boost::mpl::and_<
                all_arrays_have_correct_dim
              , all_out_arr_value_types_are_scalars
              >::type
            , std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, VectorValueT, nd, out_arity>

            , numpy::mpl::unspecified
            >::type
            type;
};

// Specialization for out_arity = 1.
template <class WiringModelAPI, class OutMapping, class RT, class VectorValueT, unsigned nd>
struct select_std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, VectorValueT, nd, 1>
{
    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    // Check if the one-and-only output array has a core dimensionality equal
    // to nd.
    typedef typename out_mapping_utils::template array<0>::template has_dim<nd>::type
            has_correct_dim;

    // Check if the one-and-only output array has a scalar data holding type.
    typedef typename is_scalar<typename WiringModelAPI::template out_arr_value_type<0>::type>::type
            has_scalar_array_data_holding_type;

    typedef typename boost::mpl::if_<
              typename boost::mpl::and_<
                has_correct_dim
              , has_scalar_array_data_holding_type
              >::type
            , std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, VectorValueT, nd, 1>

            , numpy::mpl::unspecified
            >::type
            type;
};

template <class WiringModelAPI, class OutMapping, class RT, class NestedRT, unsigned ND>
struct std_vector_return_to_core_shape_data
{
    typedef typename remove_reference<NestedRT>::type
            vector_t;
    typedef typename vector_t::value_type
            vector_value_t;
    typedef typename remove_reference<vector_value_t>::type
            vector_bare_value_t;

    typedef typename boost::mpl::if_<
              typename is_scalar<vector_bare_value_t>::type
            , typename select_std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, vector_value_t, ND, OutMapping::arity>::type

              // TODO: Add check for bp::object vector value type.
            // Check if the std::vector's value type is a std::vector again, and
            // if so, keep track of the number of dimensions.
            , typename boost::mpl::eval_if<
                typename numpy::mpl::is_std_vector<vector_value_t>::type
              , std_vector_return_to_core_shape_data<WiringModelAPI, OutMapping, RT, vector_value_t, ND+1>

              , numpy::mpl::unspecified
              >::type
            >::type
            type;
};

template <class WiringModelAPI, class OutMapping, class RT>
struct select_return_to_core_shape_data_converter
{
    typedef typename remove_reference<RT>::type
            bare_rt;

    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    typedef typename boost::mpl::if_<
              typename is_scalar<bare_rt>::type
            , select_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT>

            // TODO: Add bp::object types.
            , typename boost::mpl::if_<
                typename is_same<bare_rt, python::object>::type
              , select_bp_object_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT>

              , typename boost::mpl::eval_if<
                  typename numpy::mpl::is_std_vector<bare_rt>::type
                , std_vector_return_to_core_shape_data<WiringModelAPI, OutMapping, RT, RT, 1>

                , numpy::mpl::unspecified
                >::type
              >::type
            >::type
            type;
};

template <class WiringModelAPI, class OutMapping, class RT>
struct return_to_core_shape_data_converter
{
    typedef typename select_return_to_core_shape_data_converter<WiringModelAPI, OutMapping, RT>::type
            builtin_converter_selector;
    typedef typename boost::mpl::eval_if<
              is_same<typename builtin_converter_selector::type, numpy::mpl::unspecified>
            , ::boost::numpy::dstream::wiring::converter::return_to_core_shape_data<WiringModelAPI, OutMapping, RT>
            , builtin_converter_selector
            >::type
            type;
};

}// namespace detail
}// namespace converter
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_WIRING_RETURN_TO_CORE_SHAPE_DATA_HPP_INCLUDED
#else
#if (BOOST_PP_ITERATION_DEPTH() == 1) && (BOOST_PP_ITERATION_FLAGS() == 1)

#define AXIS BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_null_item(z, n, data) [0]
#define BOOST_NUMPY_DSTREAM_trailing_dim(_axis) \
    BOOST_PP_REPEAT(BOOST_PP_SUB(_axis,1), BOOST_NUMPY_DSTREAM_null_item, ~)

// Assumes, that the vector has at least AXIS+1 dimensions.
template <class VectorT>
struct multidim_std_vector_has_fixed_length_axis<VectorT, AXIS>
{
    static
    bool
    apply(VectorT const & v)
    {
        intptr_t const naxes = v BOOST_NUMPY_DSTREAM_trailing_dim(AXIS) .size();

        for(int i=1; i<naxes; ++i)
        {
            if(v BOOST_NUMPY_DSTREAM_trailing_dim(AXIS) [i].size !=
               v BOOST_NUMPY_DSTREAM_trailing_dim(AXIS) [0].size()
              )
            { return false; }
        }

        return true;
    }
};

#undef BOOST_NUMPY_DSTREAM_trailing_dim
#undef BOOST_NUMPY_DSTREAM_null_item

#undef AXIS

#else
#if (BOOST_PP_ITERATION_DEPTH() == 1) && (BOOST_PP_ITERATION_FLAGS() == 2)

#define ND BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_null_item(z, n, data) [0]
#define BOOST_NUMPY_DSTREAM_trailing_dim(_nd) \
    BOOST_PP_REPEAT(BOOST_PP_SUB(_nd,1), BOOST_NUMPY_DSTREAM_null_item, ~)

#define BOOST_NUMPY_DSTREAM_fill_shape(z, n, data) \
    shape [n] = v BOOST_NUMPY_DSTREAM_trailing_dim( BOOST_PP_ADD(n,1) ) .size();

#define BOOST_NUMPY_DSTREAM_check_for_fixed_length_axis(z, _axis, data) \
    typedef multidim_std_vector_has_fixed_length_axis< VectorT, _axis > BOOST_PP_CAT(flc_t,_axis) ; \
    assert( BOOST_PP_CAT(flc_t,_axis)::apply(v) );

template <class VectorT>
struct get_multidim_std_vector_shape<VectorT, ND>
{
    static
    std::vector<intptr_t>
    apply(VectorT const & v)
    {
        BOOST_PP_REPEAT_FROM_TO(1, ND, BOOST_NUMPY_DSTREAM_check_for_fixed_length_axis, ~)

        std::vector<intptr_t> shape(ND);
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_fill_shape, ~)

        return shape;
    }
};

#undef BOOST_NUMPY_DSTREAM_check_for_fixed_length_axis
#undef BOOST_NUMPY_DSTREAM_fill_shape
#undef BOOST_NUMPY_DSTREAM_trailing_dim
#undef BOOST_NUMPY_DSTREAM_null_item



#define BOOST_NUMPY_DSTREAM_for_dim_begin(z, n, data) \
    for(dim_indices_[n] = 0; dim_indices_[n] < out_core_shapes_[0][n]; ++dim_indices_[n]) {

#define BOOST_NUMPY_DSTREAM_for_dim_end(z, n, data) \
    }

template <class WiringModelAPI, class OutMapping, class RT, class VectorValueT>
struct std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, VectorValueT, ND, 1>
{
    // This implementation is used to put the ND-dimensional result into the
    // one and only output array (i.e. nd=ND and out_arity=1).
    typedef std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, VectorValueT, ND, 1>
            type;

    typedef typename remove_reference<RT>::type
            vector_t;

    typedef typename WiringModelAPI::template out_arr_value_type<0>::type
            out_arr_value_t;

    std_vector_of_scalar_return_to_core_shape_data_impl(
        numpy::detail::iter &                        iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
      : iter_(iter)
      , out_core_shapes_(out_core_shapes)
      , dim_indices_(std::vector<intptr_t>(ND))
      , op_strides_(iter_.get_operand(0).get_strides_vector())
      , iter_data_ptr_(iter_, 0, dim_indices_, op_strides_)
      , nd_accessor_(wiring::detail::nd_accessor<RT, VectorValueT, ND>(dim_indices_))
      , check_result_shape_(true)
    {
        BOOST_ASSERT((out_core_shapes_.size() == 1 && out_core_shapes_[0].size() == ND));
    }

    inline
    bool
    operator()(vector_t const & result)
    {
        // Check if the shape of the function result matches the shape of the
        // output array.
        if(check_result_shape_)
        {
            std::vector<intptr_t> const result_shape = get_multidim_std_vector_shape<vector_t, ND>::apply(result);
            if(result_shape != out_core_shapes_[0])
            {
                std::cerr << "The shape "
                          << numpy::detail::shape_vector_to_string<intptr_t>(result_shape)
                          << " of the function's "<<ND<<"-dimensional result"
                          << " vector must be "
                          << numpy::detail::shape_vector_to_string<intptr_t>(out_core_shapes_[0])
                          << "!"
                          << std::endl;
                return false;
            }
            check_result_shape_ = false;
        }

        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_for_dim_begin, ND)
        out_arr_value_t & out_arr_value = *reinterpret_cast<out_arr_value_t *>( iter_data_ptr_() );
        out_arr_value = out_arr_value_t( nd_accessor_(result) );
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_for_dim_end, ND)

        return true;
    }

    numpy::detail::iter &                             iter_;
    std::vector< std::vector<intptr_t> > const &      out_core_shapes_;
    std::vector<intptr_t>                             dim_indices_;
    std::vector<intptr_t> const                       op_strides_;
    wiring::detail::iter_data_ptr<ND, 0>              iter_data_ptr_;
    wiring::detail::nd_accessor<RT, VectorValueT, ND> nd_accessor_;
    bool                                              check_result_shape_;
};

#undef BOOST_NUMPY_DSTREAM_for_dim_end
#undef BOOST_NUMPY_DSTREAM_for_dim_begin

#undef ND

#else
#if (BOOST_PP_ITERATION_DEPTH() == 1) && (BOOST_PP_ITERATION_FLAGS() == 3)

// Loop over ND.
#define BOOST_PP_ITERATION_PARAMS_2 \
    (4, (1, 18, <boost/numpy/dstream/wiring/converter/return_to_core_shape_data.hpp>, 1))
#include BOOST_PP_ITERATE()

#else
#if (BOOST_PP_ITERATION_DEPTH() == 2) && (BOOST_PP_ITERATION_FLAGS() == 1)

#define OUT_ARITY BOOST_PP_RELATIVE_ITERATION(1)
#define ND BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_out_arr_value_type(z, n, data)                     \
    typedef typename WiringModelAPI::template out_arr_value_type<n>::type      \
            BOOST_PP_CAT(out_arr_value_t,n);

#define BOOST_NUMPY_DSTREAM_strides_def(z, n, data)                            \
    std::vector<intptr_t> const BOOST_PP_CAT(op_strides_,n);

#define BOOST_NUMPY_DSTREAM_iter_data_ptr_def(z, n, _nd)                       \
    wiring::detail::iter_data_ptr<_nd, 1> BOOST_PP_CAT(iter_data_ptr_,n);

#define BOOST_NUMPY_DSTREAM_iter_data_ptr_init(z, n, _nd)                      \
    BOOST_PP_COMMA() BOOST_PP_CAT(iter_data_ptr_,n)( wiring::detail::iter_data_ptr<_nd, 1>( iter_, n, dim_indices_, BOOST_PP_CAT(op_strides_,n) ) )

#define BOOST_NUMPY_DSTREAM_strides(z, n, data)                                \
    BOOST_PP_COMMA() BOOST_PP_CAT(op_strides_,n)( iter_.get_operand(n).get_strides_vector() )

#define BOOST_NUMPY_DSTREAM_for_dim_begin(z, n, data)                          \
    for(dim_indices_[ BOOST_PP_ADD(n,1) ] = 0; dim_indices_[ BOOST_PP_ADD(n,1) ] < out_core_shapes_[0][n]; ++ dim_indices_[ BOOST_PP_ADD(n,1) ] ) {

#define BOOST_NUMPY_DSTREAM_for_dim_end(z, n, data)                            \
    }

#define BOOST_NUMPY_DSTREAM_out_arr_value_set(z, n, _nd)                       \
    dim_indices_[0] = n;                                                       \
    BOOST_PP_CAT(out_arr_value_t,n) & BOOST_PP_CAT(out_arr_value,n) =          \
        *reinterpret_cast<BOOST_PP_CAT(out_arr_value_t,n) *>( BOOST_PP_CAT(iter_data_ptr_,n)() ); \
    BOOST_PP_CAT(out_arr_value,n) = BOOST_PP_CAT(out_arr_value_t,n)( nd_accessor_(result) );

template <class WiringModelAPI, class OutMapping, class RT, class VectorValueT>
struct std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, VectorValueT, ND, OUT_ARITY>
{
    // This implementation is used to put the ND-dimensional result into the
    // OUT_ARITY (ND-1)-dimensional output arrays by distributing the first
    // dimension of the function's result vector accross the output arrays.

    typedef std_vector_of_scalar_return_to_core_shape_data_impl<WiringModelAPI, OutMapping, RT, VectorValueT, ND, OUT_ARITY>
            type;

    typedef typename remove_reference<RT>::type
            vector_t;

    BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_out_arr_value_type, ~)

    std_vector_of_scalar_return_to_core_shape_data_impl(
        numpy::detail::iter &                        iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
      : iter_(iter)
      , out_core_shapes_(out_core_shapes)
      , dim_indices_(std::vector<intptr_t>(ND))
      , nd_accessor_(dim_indices_)
      BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_strides, ~)
      BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_iter_data_ptr_init, ND)
    {}

    inline
    bool
    operator()(vector_t const & result)
    {
        // Check if the size of the result's vector first dimension matches the
        // output arity.
        if(result.size() != OUT_ARITY)
        {
            std::cerr << "The size of the first dimension of the function's "
                      << "return vector ("<< result.size() <<") must match the "
                      << "output arity ("<< OUT_ARITY <<")!" << std::endl;
            return false;
        }

        BOOST_PP_REPEAT(BOOST_PP_SUB(ND,1), BOOST_NUMPY_DSTREAM_for_dim_begin, ~)
        BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_out_arr_value_set, ND)
        BOOST_PP_REPEAT(BOOST_PP_SUB(ND,1), BOOST_NUMPY_DSTREAM_for_dim_end, ~)

        return true;
    }

    numpy::detail::iter &                             iter_;
    std::vector< std::vector<intptr_t> > const &      out_core_shapes_;
    std::vector<intptr_t>                             dim_indices_;
    wiring::detail::nd_accessor<RT, VectorValueT, ND> nd_accessor_;
    BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_strides_def, ~)
    BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_iter_data_ptr_def, ND)
};

#undef BOOST_NUMPY_DSTREAM_out_arr_value_set
#undef BOOST_NUMPY_DSTREAM_for_dim_end
#undef BOOST_NUMPY_DSTREAM_for_dim_begin
#undef BOOST_NUMPY_DSTREAM_strides
#undef BOOST_NUMPY_DSTREAM_strides_def
#undef BOOST_NUMPY_DSTREAM_iter_data_ptr_init
#undef BOOST_NUMPY_DSTREAM_iter_data_ptr_def
#undef BOOST_NUMPY_DSTREAM_out_arr_value_type

#undef ND
#undef OUT_ARITY

#endif // (BOOST_PP_ITERATION_DEPTH() == 2) && (BOOST_PP_ITERATION_FLAGS() == 1)

#endif // (BOOST_PP_ITERATION_DEPTH() == 1) && (BOOST_PP_ITERATION_FLAGS() == 3)
#endif // (BOOST_PP_ITERATION_DEPTH() == 1) && (BOOST_PP_ITERATION_FLAGS() == 2)
#endif // (BOOST_PP_ITERATION_DEPTH() == 1) && (BOOST_PP_ITERATION_FLAGS() == 1)

#endif // BOOST_PP_IS_ITERATING
