/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/out_arr_transforms/squeeze_first_axis_if_single_input.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines an output array transformation template for
 *     squeezing the first axis of the output array if it contains only one
 *     element and if all the input arrays have a shape matching their data
 *     shape, i.e. only single input values.
 *     This will reduce the dimension of the output array by one.
 *
 *     This file is distributed under the Boost Software License,
 *     Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *     http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT_HPP_INCLUDED

#include <stdint.h>

#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/mpl/as_std_vector.hpp>
#include <boost/numpy/dstream/out_arr_transform.hpp>
#include <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace out_arr_transforms {

namespace detail {

template <int InArity, class MappingModel>
struct squeeze_first_axis_if_single_input_base;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis_if_single_input.hpp>))
#include BOOST_PP_ITERATE()

template <class MappingModel>
struct squeeze_first_axis_if_single_input
  : detail::squeeze_first_axis_if_single_input_base<MappingModel::in_arity, MappingModel>
{
    typedef squeeze_first_axis_if_single_input<MappingModel>
            type;
};

}// namespace detail

struct squeeze_first_axis_if_single_input
  : out_arr_transform_selector_type
{
    template <class MappingModel>
    struct out_arr_transform
    {
        typedef detail::squeeze_first_axis_if_single_input<MappingModel>
                type;
    };
};

}// namespace out_arr_transforms
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

// Partial specialization for input arity N.
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT__in_arr_dshape(z, n, data) \
    std::vector<intptr_t> BOOST_PP_CAT(in_arr_dshape_,n) = MappingModel::BOOST_PP_CAT(in_arr_dshape_,n)::template as_std_vector<intptr_t>();

#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT__has_dshape_shape(z, n, data) \
    && BOOST_PP_CAT(in_arr_,n).has_shape(BOOST_PP_CAT(in_arr_dshape_,n))

template <class MappingModel>
struct squeeze_first_axis_if_single_input_base<N, MappingModel>
  : out_arr_transform_base<MappingModel>
{
    inline static int
    apply(ndarray & out_arr, BOOST_PP_ENUM_PARAMS(N, ndarray const & in_arr_))
    {
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT__in_arr_dshape, ~)
        if(true BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT__has_dshape_shape, ~))
        {
            return squeeze_first_axis<MappingModel>::apply(out_arr, BOOST_PP_ENUM_PARAMS(N, in_arr_));
        }
        return 0;
    }
};

#undef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT__has_dshape_shape
#undef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_IF_SINGLE_INPUT__in_arr_dshape

#undef N

#endif // BOOST_PP_IS_ITERATING
