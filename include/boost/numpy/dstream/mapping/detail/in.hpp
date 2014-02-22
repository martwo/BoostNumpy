/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/detail/in.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines templates for input mapping.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_IN_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_IN_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace detail {

template <int InArity>
struct in;

template <int InArity>
struct make_in_mapping;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/mapping/detail/in.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_IN_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

template <>
struct in<N>
{
    template < BOOST_PP_ENUM_PARAMS_Z(1, N, class CoreShape) >
    struct core_shapes
    {
        BOOST_STATIC_CONSTANT(unsigned, arity = N);

        // List all the core shape types as core_shape_tI.
        #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(N, 1))
        #define BOOST_PP_LOCAL_MACRO(n)                                        \
            typedef typename BOOST_PP_CAT(CoreShape, n)::type BOOST_PP_CAT(core_shape_t,n);
        #include BOOST_PP_LOCAL_ITERATE()
        #undef BOOST_PP_LOCAL_MACRO
        #undef BOOST_PP_LOCAL_LIMITS
    };
};

template <>
struct make_in_mapping<N>
{
    template <class CoreShapeTuple>
    struct impl
    {
        typedef in<N>::core_shapes<BOOST_PP_ENUM_PARAMS_Z(1, N, typename CoreShapeTuple::core_shape_type_)>
                type;
    };
};

#endif // BOOST_PP_ITERATION_FLAGS

#undef N

#endif // BOOST_PP_IS_ITERATING
