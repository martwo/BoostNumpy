/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @file    boost/numpy/detail/pre_boost_python_hpp_includes.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @brief This file includes boost::numpy header files, that need to be included
 *        before the boost/python.hpp header file is included.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_PRE_BOOST_PYTHON_HPP_INCLUDES_INCLUDED
#define BOOST_NUMPY_DETAIL_PRE_BOOST_PYTHON_HPP_INCLUDES_INCLUDED

// The boost/python/detail/prefix.hpp header includes Python.h.
#include <boost/python/detail/prefix.hpp>
#include <boost/python/object_fwd.hpp>

#include <boost/numpy/detail/invoke_extension/invoke_tag.hpp>
#include <boost/numpy/detail/invoke_extension/invoke.hpp>

#endif // !BOOST_NUMPY_DETAIL_PRE_BOOST_PYTHON_HPP_INCLUDES_INCLUDED
