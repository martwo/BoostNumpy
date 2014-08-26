/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/detail/definition.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines templates for mapping definitions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_DEFINITION_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_DEFINITION_HPP_INCLUDED

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/greater.hpp>
#include <boost/mpl/greater_equal.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/integral_c.hpp>

#include <boost/type_traits/is_same.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/dstream/mapping/detail/in.hpp>
#include <boost/numpy/dstream/mapping/detail/out.hpp>
#include <boost/numpy/dstream/mapping/detail/core_shape.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {

struct mapping_definition_type
{};

namespace detail {

template <
      class OutMapping
    , class InMapping
>
struct definition
  : mapping_definition_type
{
    typedef OutMapping out;
    typedef InMapping in;

    typedef definition<OutMapping, InMapping> type;

    typedef typename boost::mpl::if_c<
              out::arity
            , boost::mpl::false_
            , boost::mpl::true_
            >::type
            maps_to_void_t;
    BOOST_STATIC_CONSTANT(bool, maps_to_void = maps_to_void_t::value);
};

template <class OutCoreShapeTuple, class InCoreShapeTuple>
struct make_definition
{
    typedef typename make_out_mapping<OutCoreShapeTuple::len::value>::template impl<OutCoreShapeTuple>::type
            out_mapping_t;
    typedef typename make_in_mapping<InCoreShapeTuple::len::value>::template impl<InCoreShapeTuple>::type
            in_mapping_t;
    typedef definition<out_mapping_t, in_mapping_t>
            type;
};

// Construct a mapping definition from two core_shape_tuple types.
template <class InCoreShapeTuple, class OutCoreShapeTuple>
typename boost::lazy_enable_if<
    boost::mpl::and_< is_core_shape_tuple<InCoreShapeTuple>
                    , is_core_shape_tuple<OutCoreShapeTuple>
    >
  , make_definition<
        OutCoreShapeTuple
      , InCoreShapeTuple
    >
>::type
operator>>(InCoreShapeTuple const &, OutCoreShapeTuple const &)
{
    return typename make_definition<
               OutCoreShapeTuple
             , InCoreShapeTuple
           >::type();
}

// Construct a mapping definition from a core_shape_tuple and a core_shape type.
template <class InCoreShapeTuple, class OutCoreShape>
typename boost::lazy_enable_if<
    boost::mpl::and_< is_core_shape_tuple<InCoreShapeTuple>
                    , is_core_shape<OutCoreShape>
    >
  , make_definition<
        core_shape_tuple<1>::template core_shapes<OutCoreShape>
      , InCoreShapeTuple
    >
>::type
operator>>(InCoreShapeTuple const &, OutCoreShape const &)
{
    return typename make_definition<
               core_shape_tuple<1>::template core_shapes<OutCoreShape>
             , InCoreShapeTuple
           >::type();
}

// Construct a mapping definition from a core_shape and a core_shape_tuple type.
template <class InCoreShape, class OutCoreShapeTuple>
typename boost::lazy_enable_if<
    boost::mpl::and_< is_core_shape<InCoreShape>
                    , is_core_shape_tuple<OutCoreShapeTuple>
    >
  , make_definition<
        OutCoreShapeTuple
      , core_shape_tuple<1>::template core_shapes<InCoreShape>
    >
>::type
operator>>(InCoreShape const &, OutCoreShapeTuple const &)
{
    return typename make_definition<
               OutCoreShapeTuple
             , core_shape_tuple<1>::template core_shapes<InCoreShape>
           >::type();
}

// Construct a mapping definition from a core_shape and a core_shape type.
template <class InCoreShape, class OutCoreShape>
typename boost::lazy_enable_if<
    boost::mpl::and_< is_core_shape<InCoreShape>
                    , is_core_shape<OutCoreShape>
    >
  , make_definition<
        core_shape_tuple<1>::template core_shapes<OutCoreShape>
      , core_shape_tuple<1>::template core_shapes<InCoreShape>
    >
>::type
operator>>(InCoreShape const &, OutCoreShape const &)
{
    return typename make_definition<
               core_shape_tuple<1>::template core_shapes<OutCoreShape>
             , core_shape_tuple<1>::template core_shapes<InCoreShape>
           >::type();
}

template <unsigned arity, class Mapping>
struct all_mapping_arrays_are_scalars_arity;

template <class Mapping>
struct all_mapping_arrays_are_scalars_arity<0, Mapping>
{
    typedef boost::mpl::bool_<true>
            type;
};

template <class Mapping>
struct all_mapping_arrays_are_scalars_arity<1, Mapping>
{
    typedef boost::is_same< typename Mapping::core_shape_t0, core_shape<0>::shape<> >
            type;
};

#define BOOST_PP_ITERATION_PARAMS_1 \
    (4, (2, BOOST_NUMPY_LIMIT_MAX_INPUT_OUTPUT_ARITY, <boost/numpy/dstream/mapping/detail/definition.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class OutMapping, unsigned Idx>
struct mapping_array_select;

#define BOOST_PP_ITERATION_PARAMS_1 \
    (4, (0, BOOST_NUMPY_LIMIT_MAX_INPUT_OUTPUT_ARITY, <boost/numpy/dstream/mapping/detail/definition.hpp>, 2))
#include BOOST_PP_ITERATE()

template <class InMapping>
struct in_mapping
{
    struct all_arrays_are_scalars
    {
        typedef typename all_mapping_arrays_are_scalars_arity<InMapping::arity, InMapping>::type
                type;
    };
};

template <class OutMapping>
struct out_mapping
{
    template <unsigned n>
    struct arity_is_equal_to
    {
        typedef typename boost::mpl::equal_to<
                      boost::mpl::integral_c<unsigned, OutMapping::arity>
                    , boost::mpl::integral_c<unsigned, n>
                >::type
                type;
    };

    template <unsigned n>
    struct arity_is_greater
    {
        typedef typename boost::mpl::greater<
                      boost::mpl::integral_c<unsigned, OutMapping::arity>
                    , boost::mpl::integral_c<unsigned, n>
                >::type
                type;
    };

    template <unsigned n>
    struct arity_is_greater_equal
    {
        typedef typename boost::mpl::greater_equal<
                      boost::mpl::integral_c<unsigned, OutMapping::arity>
                    , boost::mpl::integral_c<unsigned, n>
                >::type
                type;
    };

    struct all_arrays_are_scalars
    {
        typedef typename all_mapping_arrays_are_scalars_arity<OutMapping::arity, OutMapping>::type
                type;
    };

    template <unsigned Idx>
    struct array
    {
        typedef typename mapping_array_select<OutMapping, Idx>::type
                array_type;

        struct is_scalar
        {
            typedef typename numpy::dstream::mapping::detail::is_scalar<array_type>::type
                    type;
        };

        struct is_1d
        {
            typedef typename numpy::dstream::mapping::detail::is_1d<array_type>::type
                    type;
        };
    };
};

}// namespace detail
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_DEFINITION_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define N BOOST_PP_ITERATION()

template <class Mapping>
struct all_mapping_arrays_are_scalars_arity<N, Mapping>
{
    // By default, boost::mpl::and_ has only a maximal arity of 5, so we have
    // to construct a sequence of boost::mpl::and_<.,.> with always, two
    // arguments.
    #define BOOST_NUMPY_DEF_is_same(n) \
        boost::is_same< typename Mapping:: BOOST_PP_CAT(core_shape_t,n) , core_shape<0>::shape<> >
    #define BOOST_NUMPY_DEF_pre_and(z, n, data) \
        typename boost::mpl::and_<
    #define BOOST_NUMPY_DEF_post_and(z, n, data) \
        BOOST_PP_COMMA() BOOST_NUMPY_DEF_is_same(BOOST_PP_ADD(n,1)) >::type

    typedef BOOST_PP_REPEAT(BOOST_PP_SUB(N,1), BOOST_NUMPY_DEF_pre_and, ~)
            BOOST_NUMPY_DEF_is_same(0)
            BOOST_PP_REPEAT(BOOST_PP_SUB(N,1), BOOST_NUMPY_DEF_post_and, ~)
            type;

    #undef BOOST_NUMPY_DEF_post_and
    #undef BOOST_NUMPY_DEF_pre_and
    #undef BOOST_NUMPY_DEF_is_same
};

#undef N

#else
#if BOOST_PP_ITERATION_FLAGS() == 2

#define N BOOST_PP_ITERATION()

// In case the output mapping has arity 0, any array of the output is
// unspecified.
template <>
struct mapping_array_select<out<0>::core_shapes<numpy::mpl::unspecified>, N>
{
    typedef numpy::mpl::unspecified
            type;
};

template <class Mapping>
struct mapping_array_select<Mapping, N>
{
    typedef typename Mapping::BOOST_PP_CAT(core_shape_t,N)
            type;
};

#undef N

#endif // BOOST_PP_ITERATION_FLAGS == 2
#endif // BOOST_PP_ITERATION_FLAGS == 1

#endif // BOOST_PP_IS_ITERATING
