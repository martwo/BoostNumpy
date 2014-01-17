/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/out_arr_transforms/squeeze_first_axis_and_scalarize.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines an out_arr_transform template for squeezing the
 *     first axis of the output array if it contains only one element, and
 *     scalarizing it if the result output array is zero-dimensional or
 *     one-dimensional with only one element.
 *     This operation will be done regardless of the shapes of the input arrays!
 *
 *     This file is distributed under the Boost Software License,
 *     Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *     http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_AND_SCALARIZE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_AND_SCALARIZE_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/dstream/out_arr_transform.hpp>
#include <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis.hpp>
#include <boost/numpy/dstream/out_arr_transforms/scalarize.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace out_arr_transforms {

//==============================================================================
template <int InArity, class MappingModel>
struct squeeze_first_axis_and_scalarize;

// Partially specialize the class template for different input arities.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis_and_scalarize.hpp>))
#include BOOST_PP_ITERATE()

}/*namespace out_arr_transforms*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_AND_SCALARIZE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

template <class MappingModel>
struct squeeze_first_axis_and_scalarize<N, MappingModel>
  : out_arr_transform_base<N, MappingModel>
{
    typedef squeeze_first_axis_and_scalarize<N, MappingModel>
            type;

    inline static int
    apply(ndarray & out_arr, BOOST_PP_ENUM_PARAMS(N, ndarray const & in_arr_))
    {
        if(squeeze_first_axis<N, MappingModel>::apply(out_arr, BOOST_PP_ENUM_PARAMS(N, in_arr_)))
        {
            scalarize<N, MappingModel>::apply(out_arr, BOOST_PP_ENUM_PARAMS(N, in_arr_));
            return 1;
        }
        return 0;
    }
};

#undef N

#endif // BOOST_PP_IS_ITERATING
