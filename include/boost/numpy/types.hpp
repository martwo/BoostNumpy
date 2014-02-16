/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/types.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the boost::numpy types which are usually just
 *        typedefs of existing numpy types in order to allow a uniform type name
 *        scheme.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_TYPES_HPP_INCLUDED
#define BOOST_NUMPY_TYPES_HPP_INCLUDED

#include <boost/numpy/numpy_c_api.hpp>

namespace boost {
namespace numpy {

typedef enum {
    ANYORDER     = NPY_ANYORDER,
    CORDER       = NPY_CORDER,
    FORTRANORDER = NPY_FORTRANORDER,
    KEEPORDER    = NPY_KEEPORDER
} order_t;

typedef enum {
    NO_CASTING        = NPY_NO_CASTING,
    EQUIV_CASTING     = NPY_EQUIV_CASTING,
    SAFE_CASTING      = NPY_SAFE_CASTING,
    SAME_KIND_CASTING = NPY_SAME_KIND_CASTING,
    UNSAFE_CASTING    = NPY_UNSAFE_CASTING
} casting_t;

}// namespace numpy
}// namespace boost

#endif// !BOOST_NUMPY_TYPES_HPP_INCLUDED
