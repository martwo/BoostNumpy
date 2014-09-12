/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/detail/nd_accessor.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the nd_accessor template to access a value
 *        of a multi-dimensional object which implements the []-operator for
 *        each dimension.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_WIRING_ND_ACCESSOR_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_ND_ACCESSOR_HPP_INCLUDED

#include <stdint.h>

#include <vector>

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace detail {

template <class T, class ValueT, unsigned nd>
struct nd_accessor;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, 18, <boost/numpy/dstream/wiring/detail/nd_accessor.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_WIRING_ND_ACCESSOR_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define ND BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_dim_indices_def(z, n, data)                        \
    intptr_t & BOOST_PP_CAT(dim_index,n);

#define BOOST_NUMPY_DSTREAM_dim_indices_init(z, n, data)                       \
    BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(dim_index,n)(dim_indices[n])

#define BOOST_NUMPY_DSTREAM_access(z, n, data)                                 \
    [ BOOST_PP_CAT(dim_index,n) ]

template <class T, class ValueT>
struct nd_accessor<T, ValueT, ND>
{
    nd_accessor(std::vector<intptr_t> & dim_indices)
      : BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_dim_indices_init, ~)
    {}

    inline
    ValueT
    operator()(T const & nd_obj)
    {
        return nd_obj BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_access, ~) ;
    }

    BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_dim_indices_def, ~)
};

#undef BOOST_NUMPY_DSTREAM_access
#undef BOOST_NUMPY_DSTREAM_dim_indices_init
#undef BOOST_NUMPY_DSTREAM_dim_indices_def

#undef ND

#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
