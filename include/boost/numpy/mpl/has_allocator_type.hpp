/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/mpl/has_allocator_type.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the MPL template
 *        boost::numpy::mpl::has_allocator_type for
 *        checking if a type T has a sub type named "allocator_type". It uses
 *        boost introspection to do so.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_MPL_HAS_ALLOCATOR_TYPE_HPP_INCLUDED
#define BOOST_NUMPY_MPL_HAS_ALLOCATOR_TYPE_HPP_INCLUDED

#include <boost/mpl/has_xxx.hpp>

namespace boost {
namespace numpy {
namespace mpl {

BOOST_MPL_HAS_XXX_TRAIT_DEF(allocator_type)

}// namespace mpl
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_MPL_HAS_ALLOCATOR_TYPE_HPP_INCLUDED
