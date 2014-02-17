/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines templates for data stream mapping functionalty.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_HPP_INCLUDED

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/pp.hpp>
#include <boost/numpy/mpl/unspecified.hpp>

#include <boost/numpy/dstream/dim.hpp>
#include <boost/numpy/dstream/mapping/detail/definition.hpp>
#include <boost/numpy/dstream/mapping/detail/core_shape.hpp>

namespace boost {
namespace numpy {
namespace dstream {

mapping::detail::core_shape<0>::shape<>
scalar()
{
    return mapping::detail::core_shape<0>::shape<>();
}

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_CORE_SHAPE_ND, <boost/numpy/dstream/mapping.hpp>))
#include BOOST_PP_ITERATE()

namespace mapping {

struct mapping_model_selector_type
{};

// We derive the mapping_model_type from the mapping_model_selector_type because
// a mapping model can of course always select itself as mapping model.
// This allows to specify either a mapping model selector or a mapping model
// to the def/classdef functions.
struct mapping_model_type
  : mapping_model_selector_type
{};

//==============================================================================
/**
 * \brief The boost::numpy::dstream::mapping::base_mapping_model template
 *     provides a base class for a particular mapping model.
 *     A mapping model defines the data shape (dshape) of the output
 *     ndarray, the data shape of all the input ndarrays, and the broadcasting
 *     rules for the input arrays that can be used for the
 *     boost::numpy::detail::iter class.
 */
template <
      int InArity
    , class OutArrDShape
    , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, class InArrDShape_, = BOOST_NUMPY_PP_MPL_VOID BOOST_PP_INTERCEPT)
>
struct base_mapping_model
  : mapping_model_type
{
    BOOST_STATIC_CONSTANT(int, in_arity = InArity);

    typedef OutArrDShape out_arr_dshape;

    //--------------------------------------------------------------------------
    // List all the in array data shape types.
    #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(BOOST_NUMPY_LIMIT_INPUT_ARITY, 1))
    #define BOOST_PP_LOCAL_MACRO(n)                                            \
        typedef BOOST_PP_CAT(InArrDShape_, n) BOOST_PP_CAT(in_arr_dshape_,n);
    #include BOOST_PP_LOCAL_ITERATE()
    //--------------------------------------------------------------------------
};

}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

template <BOOST_PP_ENUM_PARAMS_Z(1, N, int D)>
mapping::detail::core_shape<N>::shape<BOOST_PP_ENUM_PARAMS_Z(1, N, D)>
array()
{
    return mapping::detail::core_shape<N>::shape<BOOST_PP_ENUM_PARAMS_Z(1, N, D)>();
}

#undef N

#endif // BOOST_PP_IS_ITERATING
