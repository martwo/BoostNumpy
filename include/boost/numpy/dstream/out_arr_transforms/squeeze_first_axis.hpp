/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/out_arr_transforms/squeeze_first_axis.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines an out_arr_transform template for squeezing the
 *     first axis of the output arrays if it contains only one element. This
 *     will reduce the dimension of the array by one.
 *     This operation will be done regardless of the shapes of the input arrays!
 *
 *     This file is distributed under the Boost Software License,
 *     Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *     http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/dstream/out_arr_transform.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace out_arr_transforms {

//==============================================================================
template <int InArity, class MappingModel>
struct squeeze_first_axis;

// Partially specialize the class template for different input arity.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis.hpp>))
#include BOOST_PP_ITERATE()

}/*namespace out_arr_transforms*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_SQUEEZE_FIRST_AXIS_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

template <class MappingModel>
struct squeeze_first_axis<N, MappingModel>
  : out_arr_transform_base<N, MappingModel>
{
    typedef squeeze_first_axis<N, MappingModel>
            type;

    inline static int
    apply(ndarray & out_arr, BOOST_PP_ENUM_PARAMS(N, ndarray const & in_arr_))
    {
        int const nd = out_arr.get_nd();
        if(nd > 0 && out_arr.shape(0) == 1)
        {
            // There is only one element in the first axis, so we can
            // just reshape the array to exclude the first axis.
            python::list shape;
            for(int i=1; i<nd; ++i) {
                shape.append(out_arr.shape(i));
            }
            out_arr = out_arr.reshape(shape);
            return 1;
        }
        return 0;
    }
};

#undef N

#endif // BOOST_PP_IS_ITERATING
