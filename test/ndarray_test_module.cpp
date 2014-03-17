/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 * 2010-2012
 *     Jim Bosch, Ankit Daftery
 *
 * @file test/ndarray_test_module.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>
 * @brief This file implements a Python module for basic tests of the
 *        boost::numpy::ndarray class.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;

namespace test {

static np::ndarray zeros(bp::tuple shape, np::dtype dt)
{ return np::zeros(shape, dt); }

static np::ndarray array1(bp::object obj)
{ return np::array(obj); }

static np::ndarray array2(bp::object obj, np::dtype dt)
{ return np::array(obj, dt); }

static np::ndarray empty(bp::tuple shape, np::dtype dt)
{ return np::empty(shape, dt);}

static np::ndarray transpose(np::ndarray arr)
{ return arr.transpose(); }

static np::ndarray squeeze(np::ndarray arr)
{ return arr.squeeze(); }

static np::ndarray reshape(np::ndarray arr, bp::tuple shape)
{ return arr.reshape(shape);}

}// namespace test

BOOST_PYTHON_MODULE(ndarray_test_module)
{
    np::initialize();
    bp::def("zeros",           &test::zeros);
    bp::def("zeros_as_matrix", &test::zeros, np::as_matrix<>());
    bp::def("array",           &test::array1);
    bp::def("array",           &test::array2);
    bp::def("empty",           &test::empty);
    bp::def("transpose",       &test::transpose);
    bp::def("squeeze",         &test::squeeze);
    bp::def("reshape",         &test::reshape);
}
