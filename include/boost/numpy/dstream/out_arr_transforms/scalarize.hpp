/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/out_arr_transforms/scalarize.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines an output array transformation template for
 *        scalarizing the output array of a data stream operation.
 *        It just calls ndarray.scalarize() on the output array. So if the
 *        output array is a null-dimensional array or an one-dimensional array
 *        with only one element, it will be transformed into a numpy scalar.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SCALARIZE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SCALARIZE_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/dstream/out_arr_transform.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace out_arr_transforms {

namespace detail {

template <int InArity, class MappingModel>
struct scalarize_base;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/out_arr_transforms/scalarize.hpp>))
#include BOOST_PP_ITERATE()

template <class MappingModel>
struct scalarize
  : scalarize_base<MappingModel::in_arity, MappingModel>
{
    typedef scalarize<MappingModel>
            type;
};

}// namespace detail

struct scalarize
  : out_arr_transform_selector_type
{
    template <class MappingModel>
    struct out_arr_transform
    {
        typedef detail::scalarize<MappingModel>
                type;
    };
};

}// namespace out_arr_transforms
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_NONE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

// Partial specialization for input arity N.
template <class MappingModel>
struct scalarize_base<N, MappingModel>
  : out_arr_transform_base<MappingModel>
{
    inline static int
    apply(ndarray & out_arr, BOOST_PP_ENUM_PARAMS(N, ndarray const & in_arr_))
    {
        out_arr = out_arr.scalarize();
        return 1;
    }
};

#undef N

#endif // BOOST_PP_IS_ITERATING
