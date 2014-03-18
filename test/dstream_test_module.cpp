/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file test/dstream_test_module.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>
 * @brief This file implements a Python module for testing the
 *        boost::numpy::dstream library.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#include <boost/python.hpp>

#define NDEBUG

#include <boost/numpy.hpp>
#include <boost/numpy/dstream.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;
namespace ds = boost::numpy::dstream;

namespace test {

template <typename T>
static
void
unary_to_void(T)
{}

template <typename T>
static
T
unary_to_T_squared(T v)
{
    return v*v;
}

}// namespace test

BOOST_PYTHON_MODULE(dstream_test_module)
{
    bn::initialize();

    ds::def("unary_to_void__double", &test::unary_to_void<double>, bp::arg("v"));
    ds::def("unary_to_T_squared__double", &test::unary_to_T_squared<double>, bp::arg("v"));
}
