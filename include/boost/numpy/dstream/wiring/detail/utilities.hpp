/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/detail/utilities.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines templates for wiring utility meta functions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_WIRING_DETAIL_UTILITIES_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_DETAIL_UTILITIES_HPP_INCLUDED

#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/mpl/bool.hpp>

#include <boost/numpy/limits.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace detail {

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
  , unsigned out_arity
>
struct all_out_arr_value_types_arity;

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
>
struct all_out_arr_value_types_arity<WiringModelAPI, UnaryMetaFunction, 0>
{
    typedef boost::mpl::bool_<false>
            type;
};

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
>
struct all_out_arr_value_types_arity<WiringModelAPI, UnaryMetaFunction, 1>
{
    typedef typename UnaryMetaFunction< typename WiringModelAPI::template out_arr_value_type<0>::type >::type
            type;
};

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
  , unsigned out_arity
>
struct any_out_arr_value_type_arity;

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
>
struct any_out_arr_value_type_arity<WiringModelAPI, UnaryMetaFunction, 0>
{
    typedef boost::mpl::bool_<false>
            type;
};

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
>
struct any_out_arr_value_type_arity<WiringModelAPI, UnaryMetaFunction, 1>
{
    typedef typename UnaryMetaFunction< typename WiringModelAPI::template out_arr_value_type<0>::type >::type
            type;
};

#define BOOST_PP_ITERATION_PARAMS_1 \
    (4, (2, BOOST_NUMPY_LIMIT_OUTPUT_ARITY, <boost/numpy/dstream/wiring/detail/utilities.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class WiringModelAPI>
struct utilities
{
    template <
        template<class T> class UnaryMetaFunction
    >
    struct all_out_arr_value_types
    {
        typedef typename all_out_arr_value_types_arity<WiringModelAPI, UnaryMetaFunction, WiringModelAPI::mapping_definition_t::out::arity>::type
                type;
    };

    template <
        template<class T> class UnaryMetaFunction
    >
    struct any_out_arr_value_type
    {
        typedef typename any_out_arr_value_type_arity<WiringModelAPI, UnaryMetaFunction, WiringModelAPI::mapping_definition_t::out::arity>::type
                type;
    };
};

}// namespace detail
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_DETAIL_UTILITIES_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define OUT_ARITY BOOST_PP_ITERATION()

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
>
struct all_out_arr_value_types_arity<WiringModelAPI, UnaryMetaFunction, OUT_ARITY>
{
    #define BOOST_NUMPY_DEF_pre_and(z, n, data) \
        typename boost::mpl::and_<
    #define BOOST_NUMPY_DEF_unary_metafunction_result(n) \
        typename UnaryMetaFunction< typename WiringModelAPI::template out_arr_value_type<n>::type >::type
    #define BOOST_NUMPY_DEF_post_and(z, n, data) \
        BOOST_PP_COMMA() BOOST_NUMPY_DEF_unary_metafunction_result(BOOST_PP_ADD(n,1)) >::type

    typedef BOOST_PP_REPEAT(BOOST_PP_SUB(OUT_ARITY,1), BOOST_NUMPY_DEF_pre_and, ~)
            BOOST_NUMPY_DEF_unary_metafunction_result(0)
            BOOST_PP_REPEAT(BOOST_PP_SUB(OUT_ARITY,1), BOOST_NUMPY_DEF_post_and, ~)
            type;

    #undef BOOST_NUMPY_DEF_post_and
    #undef BOOST_NUMPY_DEF_unary_metafunction_result
    #undef BOOST_NUMPY_DEF_pre_and
};

template <
    class WiringModelAPI
  , template <class T> class UnaryMetaFunction
>
struct any_out_arr_value_type_arity<WiringModelAPI, UnaryMetaFunction, OUT_ARITY>
{
    #define BOOST_NUMPY_DEF_pre_or(z, n, data) \
        typename boost::mpl::or_<
    #define BOOST_NUMPY_DEF_unary_metafunction_result(n) \
        typename UnaryMetaFunction< typename WiringModelAPI::template out_arr_value_type<n>::type >::type
    #define BOOST_NUMPY_DEF_post_or(z, n, data) \
        BOOST_PP_COMMA() BOOST_NUMPY_DEF_unary_metafunction_result(BOOST_PP_ADD(n,1)) >::type

    typedef BOOST_PP_REPEAT(BOOST_PP_SUB(OUT_ARITY,1), BOOST_NUMPY_DEF_pre_or, ~)
            BOOST_NUMPY_DEF_unary_metafunction_result(0)
            BOOST_PP_REPEAT(BOOST_PP_SUB(OUT_ARITY,1), BOOST_NUMPY_DEF_post_or, ~)
            type;

    #undef BOOST_NUMPY_DEF_post_or
    #undef BOOST_NUMPY_DEF_unary_metafunction_result
    #undef BOOST_NUMPY_DEF_pre_or
};

#undef OUT_ARITY

#endif // BOOST_PP_ITERATION_FLAGS == 1

#endif // BOOST_PP_IS_ITERATING
