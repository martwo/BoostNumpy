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
namespace bn = boost::numpy;

namespace test {

static bn::ndarray zeros(bp::tuple shape, bn::dtype dt)
{ return bn::zeros(shape, dt); }

static bn::ndarray array1(bp::object obj)
{ return bn::array(obj); }

static bn::ndarray array2(bp::object obj, bn::dtype dt)
{ return bn::array(obj, dt); }

static bn::ndarray empty(bp::tuple shape, bn::dtype dt)
{ return bn::empty(shape, dt);}

static bn::ndarray transpose(bn::ndarray arr)
{ return arr.transpose(); }

static bn::ndarray squeeze(bn::ndarray arr)
{ return arr.squeeze(); }

static bn::ndarray reshape(bn::ndarray arr, bp::tuple shape)
{ return arr.reshape(shape);}

}// namespace test

BOOST_PYTHON_MODULE(ndarray_test_module)
{
    bn::initialize();
    bp::def("zeros",           &test::zeros);
    bp::def("zeros_as_matrix", &test::zeros, bn::as_matrix<>());
    bp::def("array",           &test::array1);
    bp::def("array",           &test::array2);
    bp::def("empty",           &test::empty);
    bp::def("transpose",       &test::transpose);
    bp::def("squeeze",         &test::squeeze);
    bp::def("reshape",         &test::reshape);
}
