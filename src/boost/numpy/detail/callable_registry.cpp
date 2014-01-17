/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/detail/callable_registry.cpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file implements the callable registry for storing callable
 *        objects. The callable objects are the bridges between the C++
 *        functions and the Python functions with numpy support.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#define BOOST_NUMPY_INTERNAL_IMPL
#include <boost/numpy/internal_impl.hpp>

#include <vector>

#include <boost/shared_ptr.hpp>

namespace boost {
namespace numpy {
namespace detail {

//std::vector< boost::shared_ptr<void> > callable_registry_vector;

}/*namespace detail*/
}/*namespace numpy*/
}/*namespace boost*/
