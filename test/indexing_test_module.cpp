/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 * 2010-2012
 *     Jim Bosch, Ankit Daftery
 *
 * @file test/indexing_test_module.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>
 * @brief This file implements a Python module for indexing tests of the
 *        boost::numpy::ndarray class.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/slice.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::numpy;

namespace test {

static bp::object single(np::ndarray arr, int i)
{ return arr[i]; }

static bp::object slice(np::ndarray arr, bp::slice sl)
{ return arr[sl]; }

static bp::object index_array(np::ndarray arr, np::ndarray idxarr)
{ return arr[idxarr]; }

static bp::object index_array_2d(np::ndarray arr, np::ndarray idxarr1, np::ndarray idxarr2)
{ return arr[bp::make_tuple(idxarr1, idxarr2)]; }

static bp::object index_array_slice(np::ndarray arr, np::ndarray idxarr, bp::slice sl)
{ return arr[bp::make_tuple(idxarr, sl)]; }

}// namespace test

BOOST_PYTHON_MODULE(indexing_test_module)
{
    np::initialize();
    bp::def("single",            &test::single);
    bp::def("slice",             &test::slice);
    bp::def("index_array",       &test::index_array);
    bp::def("index_array_2d",    &test::index_array_2d);
    bp::def("index_array_slice", &test::index_array_slice);
}
