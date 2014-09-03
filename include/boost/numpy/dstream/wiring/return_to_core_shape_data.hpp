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
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/assert.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>

#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/dstream/mapping/detail/definition.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace converter {

template <class OutMapping, class RT, class Enable=void>
struct return_to_core_shape_data
{
    typedef return_to_core_shape_data<OutMapping, RT, Enable>
            type;

    // The return_to_core_shape_data needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_return_to_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_RETURN_TYPE_RT_AND_OUT_MAPPING_TYPE_OutMapping, (RT, OutMapping));
};

namespace detail {

//------------------------------------------------------------------------------
// The scalar_return_to_core_shape_data template is used to put the function's
// scalar result data into the one and only output array. This makes only sense
// for scalar out mappings, which is ensured already by the
// select_return_to_core_shape_data_converter metafunction.
template <class OutMapping, class RT>
struct scalar_return_to_core_shape_data
{
    typedef scalar_return_to_core_shape_data<OutMapping, RT>
            type;

    template <class WiringModelAPI>
    static
    bool
    apply(
        RT result
      , numpy::detail::iter & iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
    {
        BOOST_ASSERT((out_core_shapes.size() == 1 && out_core_shapes[0].size() == 1 && out_core_shapes[0][0] == 1));

        typedef typename WiringModelAPI::template out_arr_value_type<0>::type
                out_arr_value_t;

        out_arr_value_t & out_arr_value = *reinterpret_cast<out_arr_value_t *>(iter.get_data(0));
        out_arr_value = out_arr_value_t(result);

        return true;
    }
};

//------------------------------------------------------------------------------
// The std_vector_of_scalar_return_to_core_shape_data template is used to put
// the function's result data from a std::vector<scalar> type into OutArity
// output arrays, (e.g. ND=1).
template <class OutMapping, class RT, unsigned OutArity>
struct std_vector_of_scalar_return_to_core_shape_data;

template <class OutMapping, class RT>
struct std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, 1>
{
    // This implementation is used to put the 1-dimensional result into the
    // one and only output array (i.e. OutArity = 1).
    typedef std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, 1>
            type;

    template <class WiringModelAPI>
    static
    bool
    apply(
        RT result
      , numpy::detail::iter & iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
    {
        BOOST_ASSERT((out_core_shapes.size() == 1 && out_core_shapes[0].size() == 1));

        typedef typename WiringModelAPI::template out_arr_value_type<0>::type
                out_arr_value_t;

        intptr_t const out_op_value_stride = iter.get_item_size(0);
        intptr_t const N = out_core_shapes[0][0];
        if(result.size() != N)
        {
            std::cerr << "The length of the result vector "
                      << "("<<result.size()<<") must be " << N << "!"
                      << std::endl;
            return false;
        }
        for(intptr_t i=0; i<N; ++i)
        {
            out_arr_value_t & out_arr_value = *reinterpret_cast<out_arr_value_t *>(iter.get_data(0) + i*out_op_value_stride);
            out_arr_value = out_arr_value_t(result[i]);
        }
        return true;
    }
};

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (2, BOOST_NUMPY_LIMIT_OUTPUT_ARITY, <boost/numpy/dstream/wiring/return_to_core_shape_data.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class OutMapping, class RT>
struct select_std_vector_of_scalar_return_to_core_shape_data
{
    // At this point we know that RT is std::vector<scalar_type>, i.e. the
    // function's result is 1-dimensional. Now we need to distribute the result
    // values according to the out mapping arity.
    //
    // If the output arity is 1, the core shape of the output array must be
    // 1-dimensional, otherwise all output arrays have to have a scalar core
    // shape, otherwise it's not intuitive how to distribute the scalar values.
    // Thus, the user needs to provide a converter in such cases.
    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    typedef typename boost::mpl::if_<
              typename out_mapping_utils::template arity_is_equal_to<1>::type
            , typename boost::mpl::if_<
                typename out_mapping_utils::template array<0>::is_1d::type
              , std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, 1>

              , numpy::mpl::unspecified
              >::type

            , typename boost::mpl::if_<
                typename out_mapping_utils::all_arrays_are_scalars::type
              , std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, OutMapping::arity>

              , numpy::mpl::unspecified
              >::type
            >::type
            type;
};

//------------------------------------------------------------------------------
// The nested_std_vector_of_scalar_return_to_core_shape_data template is used
// to put the function's result data from a ND-nested std::vector of
// std::vector's of scalar type into the OutArity output arrays. The NestedRT
// type is the inner most std::vector<scalar> type of the nested RT type.
template <class OutMapping, class RT, class NestedRT, unsigned ND, unsigned OutArity>
struct nested_std_vector_of_scalar_return_to_core_shape_data;

template <class OutMapping, class RT, class NestedRT, unsigned ND>
struct nested_std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, NestedRT, ND, 1>
{
    // This implementation is used to put the ND-dimensional result into the
    // one and only output array. Thus the number of core shape dimensions must
    // match ND.
    typedef nested_std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, NestedRT, ND, 1>
            type;

    template <class WiringModelAPI>
    static
    bool
    apply(
        RT result
      , numpy::detail::iter & iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
    {
        // FIXME
        return false;
    }
};

// FIXME: Here come all the OutArity specializations for the ND-dimensional
//        function result.

template <class OutMapping, class RT, class NestedRT, unsigned ND>
struct nested_std_vector_return_to_core_shape_data
{
    typedef typename NestedRT::value_type
            vector_value_type;
    typedef typename remove_reference<vector_value_type>::type
            vector_bare_value_type;

    typedef typename boost::mpl::if_<
              typename is_scalar<vector_bare_value_type>::type
              // At this point we know that RT is
              // std::vector< std::vector< ... scalar ... > >.
            , nested_std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, NestedRT, ND, OutMapping::arity>

            , typename boost::mpl::eval_if<
                typename numpy::mpl::is_std_vector<vector_value_type>::type
              , nested_std_vector_return_to_core_shape_data<OutMapping, RT, vector_value_type, ND+1>

              , numpy::mpl::unspecified
              >::type
            >::type
            type;
};

template <class OutMapping, class RT>
struct std_vector_return_to_core_shape_data
{
    typedef typename RT::value_type
            vector_value_type;
    typedef typename remove_reference<vector_value_type>::type
            vector_bare_value_type;

    typedef typename boost::mpl::if_<
              typename is_scalar<vector_bare_value_type>::type
            , typename select_std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT>::type

            // Check if the std::vector's value type is a std::vector again, and
            // if so, keep track of the number of dimensions.
            , typename boost::mpl::if_<
                typename numpy::mpl::is_std_vector<vector_value_type>::type
              , nested_std_vector_return_to_core_shape_data<OutMapping, RT, vector_value_type, 2>

              , numpy::mpl::unspecified
              >::type
            >::type
            type;
};

template <class OutMapping, class RT>
struct select_return_to_core_shape_data_converter
{
    typedef typename remove_reference<RT>::type
            bare_rt;

    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    typedef typename boost::mpl::if_<
              typename boost::mpl::and_<
                         typename is_scalar<bare_rt>::type
                       , typename out_mapping_utils::template arity_is_equal_to<1>::type
                       , typename out_mapping_utils::template array<0>::is_scalar::type
                       >::type
            , scalar_return_to_core_shape_data<OutMapping, RT>

            , typename boost::mpl::if_<
                typename numpy::mpl::is_std_vector<bare_rt>::type
              , std_vector_return_to_core_shape_data<OutMapping, RT>

              , numpy::mpl::unspecified
              >::type
            >::type
            type;
};

template <class OutMapping, class RT>
struct return_to_core_shape_data_converter
{
    typedef typename select_return_to_core_shape_data_converter<OutMapping, RT>::type
            builtin_converter_selector;
    typedef typename boost::mpl::eval_if<
              is_same<typename builtin_converter_selector::type, numpy::mpl::unspecified>
            , ::boost::numpy::dstream::wiring::converter::return_to_core_shape_data<OutMapping, RT>
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

#if BOOST_PP_ITERATION_FLAGS() == 1

#define OUT_ARITY BOOST_PP_ITERATION()

template <class OutMapping, class RT>
struct std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, OUT_ARITY>
{
    // This implementation is used to put the 1-dimensional result into the
    // OUT_ARITY scalar output arrays.

    typedef std_vector_of_scalar_return_to_core_shape_data<OutMapping, RT, OUT_ARITY>
            type;

    template <class WiringModelAPI>
    static
    bool
    apply(
        RT result
      , numpy::detail::iter & iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
    {
        // Check if the number of scalar values match the output arity.
        if(result.size() != OUT_ARITY)
        {
            std::cerr << "The size of the return vector "
                      << "("<< result.size() <<") does not match the output "
                      << "arity ("<< OUT_ARITY <<")!" << std::endl;
            return false;
        }

        #define BOOST_NUMPY_DSTREAM_DEF(z, n, data)                                     \
            typedef typename WiringModelAPI::template out_arr_value_type<n>::type       \
                    BOOST_PP_CAT(out_arr_value_t,n);                                    \
            BOOST_PP_CAT(out_arr_value_t,n) & BOOST_PP_CAT(out_arr_value,n) =           \
                *reinterpret_cast<BOOST_PP_CAT(out_arr_value_t,n) *>(iter.get_data(n)); \
            BOOST_PP_CAT(out_arr_value,n) = BOOST_PP_CAT(out_arr_value_t,n)(result[n]);
        BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DSTREAM_DEF, ~)
        #undef BOOST_NUMPY_DSTREAM_DEF
        return true;
    }
};

#undef OUT_ARITY

#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
