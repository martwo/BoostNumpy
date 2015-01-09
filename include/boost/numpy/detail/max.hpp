/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/detail/max.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the max template function for a variadic number of
 *        values using the C++03 standard.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_MAX_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_MAX_HPP_INCLUDED

#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <boost/numpy/limits.hpp>

namespace boost {
namespace numpy {
namespace detail {

template <class T>
T max(T x)
{
    return x;
}

template <class T>
T max(T x, T y)
{
    return x > y ? x : y;
}

#define BOOST_NUMPY_DEF(z, n, data)                                            \
    template <class T>                                                         \
    T max(T x, BOOST_PP_ENUM_PARAMS(n, T x))                                   \
    {                                                                          \
        return max(x, max(BOOST_PP_ENUM_PARAMS(n, x)));                        \
    }
BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_ADD(BOOST_PP_ADD(BOOST_NUMPY_LIMIT_INPUT_ARITY, BOOST_NUMPY_LIMIT_OUTPUT_ARITY), 2), BOOST_NUMPY_DEF, ~)
#undef BOOST_NUMPY_DEF

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DETAIL_MAX_HPP_INCLUDED
