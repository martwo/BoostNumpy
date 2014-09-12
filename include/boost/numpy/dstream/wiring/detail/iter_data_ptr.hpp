/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/detail/iter_data_ptr.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the iter_data_ptr template to generate a pointer
 *        value for a particular data inside a numpy iterator operand.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_WIRING_ITER_DATA_PTR_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_ITER_DATA_PTR_HPP_INCLUDED

#include <stdint.h>

#include <vector>

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/numpy/detail/iter.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace detail {

template <unsigned nd, unsigned dim_offset>
struct iter_data_ptr;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, 18, <boost/numpy/dstream/wiring/detail/iter_data_ptr.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_DATA_PTR_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define ND BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_dim_index_ref_def(z, n, data) \
    intptr_t & BOOST_PP_CAT(dim_index,n);

#define BOOST_NUMPY_DSTREAM_op_stride_ref_def(z, n, data) \
    intptr_t const & BOOST_PP_CAT(op_stride,n);

#define BOOST_NUMPY_DSTREAM_dim_index_ref_init(z, n, _dim_offset) \
    BOOST_PP_COMMA() BOOST_PP_CAT(dim_index,n)( dim_indices [_dim_offset + n] )

#define BOOST_NUMPY_DSTREAM_op_stride_ref_init(z, n, _dim_offset) \
    BOOST_PP_COMMA() BOOST_PP_CAT(op_stride,n)( op_strides [op_nd_ - BOOST_PP_SUB(ND,n) + _dim_offset])

#define BOOST_NUMPY_DSTREAM_value_offset(z, n, _dim_offset)                    \
    + BOOST_PP_CAT(dim_index,n) * BOOST_PP_CAT(op_stride,n)

template <>
struct iter_data_ptr<ND, 0>
{
    iter_data_ptr(
        numpy::detail::iter &         iter
      , intptr_t const                op_idx
      , std::vector<intptr_t>       & dim_indices
      , std::vector<intptr_t> const & op_strides
    )
      : iter_(iter)
      , op_idx_(op_idx)
      , op_nd_(op_strides.size())
      BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_dim_index_ref_init, 0)
      BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_op_stride_ref_init, 0)
    {}

    char*
    operator()()
    {
        return iter_.get_data(op_idx_) BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_value_offset, 0);
    }

    numpy::detail::iter & iter_;
    size_t const          op_idx_;
    intptr_t const        op_nd_;
    BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_dim_index_ref_def, ~)
    BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_op_stride_ref_def, ~)
};

template <>
struct iter_data_ptr<ND, 1>
{
    iter_data_ptr(
        numpy::detail::iter & iter
      , intptr_t const op_idx
      , std::vector<intptr_t>       & dim_indices
      , std::vector<intptr_t> const & op_strides
    )
      : iter_(iter)
      , op_idx_(op_idx)
      , op_nd_(op_strides.size())
      BOOST_PP_REPEAT(BOOST_PP_SUB(ND,1), BOOST_NUMPY_DSTREAM_dim_index_ref_init, 1)
      BOOST_PP_REPEAT(BOOST_PP_SUB(ND,1), BOOST_NUMPY_DSTREAM_op_stride_ref_init, 1)
    {}

    char*
    operator()()
    {
        return iter_.get_data(op_idx_) BOOST_PP_REPEAT(BOOST_PP_SUB(ND,1), BOOST_NUMPY_DSTREAM_value_offset, 1);
    }

    numpy::detail::iter & iter_;
    size_t const          op_idx_;
    intptr_t const        op_nd_;
    BOOST_PP_REPEAT(BOOST_PP_SUB(ND,1), BOOST_NUMPY_DSTREAM_dim_index_ref_def, ~)
    BOOST_PP_REPEAT(BOOST_PP_SUB(ND,1), BOOST_NUMPY_DSTREAM_op_stride_ref_def, ~)
};

#undef BOOST_NUMPY_DSTREAM_value_offset
#undef BOOST_NUMPY_DSTREAM_op_stride_ref_init
#undef BOOST_NUMPY_DSTREAM_dim_index_ref_init
#undef BOOST_NUMPY_DSTREAM_op_stride_ref_def
#undef BOOST_NUMPY_DSTREAM_dim_index_ref_def

#undef ND

#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
