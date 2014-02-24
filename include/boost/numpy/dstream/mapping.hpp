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

struct mapping_definition_selector_type
{};

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
