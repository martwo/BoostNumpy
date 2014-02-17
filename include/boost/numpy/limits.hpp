/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/limits.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the limits for certain library components.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_LIMITS_HPP_INCLUDED
#define BOOST_NUMPY_LIMITS_HPP_INCLUDED

#include <boost/mpl/limits/vector.hpp>

#ifndef BOOST_NUMPY_LIMIT_INPUT_ARITY
    #define BOOST_NUMPY_LIMIT_INPUT_ARITY 5//10
#endif

#ifndef BOOST_NUMPY_LIMIT_OUTPUT_ARITY
    #define BOOST_NUMPY_LIMIT_OUTPUT_ARITY 5//10
#endif

#define BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY \
    BOOST_NUMPY_LIMIT_INPUT_ARITY + BOOST_NUMPY_LIMIT_OUTPUT_ARITY

#define BOOST_NUMPY_LIMIT_CORE_SHAPE_ND \
    BOOST_MPL_LIMIT_VECTOR_SIZE

#endif // !BOOST_NUMPY_LIMITS_HPP_INCLUDED
