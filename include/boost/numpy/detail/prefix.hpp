/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @file    boost/numpy/detail/prefix.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @brief This file makes sure, that the include order of boost::python and
 *        boost::numpy header files is correct. It includes the boost/python.hpp
 *        header file.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_PREFIX_INCLUDED
#define BOOST_NUMPY_DETAIL_PREFIX_INCLUDED

#include <boost/numpy/detail/pre_boost_python_hpp_includes.hpp>
#include <boost/python.hpp>

#endif // !BOOST_NUMPY_DETAIL_PREFIX_INCLUDED
