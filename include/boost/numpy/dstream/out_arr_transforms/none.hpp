/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/out_arr_transforms/none.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines an out_arr_transform template for not transforming
 *        the output array of a data stream operation at all.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_NONE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_NONE_HPP_INCLUDED

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
struct none_base;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/out_arr_transforms/none.hpp>))
#include BOOST_PP_ITERATE()

}// namespace detail

//==============================================================================
template <class MappingModel>
struct none
  : detail::none_base<MappingModel::in_arity, MappingModel>
{
    typedef none<MappingModel>
            type;
};

}/*namespace out_arr_transforms*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_NONE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

// Partial specialization of none_base for input arity N.
template <class MappingModel>
struct none_base<N, MappingModel>
  : out_arr_transform_base<MappingModel>
{
    inline static int
    apply(ndarray & out_arr, BOOST_PP_ENUM_PARAMS(N, ndarray const & in_arr_))
    {
        return 0;
    }
};

#undef N

#endif // BOOST_PP_IS_ITERATING
