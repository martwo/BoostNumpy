/**
 * $Id$
 *
 * Copyright (C)
 * 2015
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file boost/numpy/utilities.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>
 * @brief This file contains the implementation of the BoostNumpy utility
 *        functions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#define BOOST_NUMPY_INTERNAL_IMPL
#include <boost/numpy/internal_impl.hpp>

#include <boost/python.hpp>
#include <boost/python/import.hpp>

#include <boost/numpy/utilities.hpp>

namespace bp = boost::python;

namespace boost {
namespace numpy {

bp::object
all(ndarray const & a, int axis)
{
    bp::object out(bp::detail::new_reference(PyArray_All((PyArrayObject*)a.ptr(), axis, NULL)));
    return out;
}

void
all(ndarray const & a, int axis, ndarray & out)
{
    // The PyArray_All function returns a new reference on the None object, so
    // we need to create a bp::object to decrement the refcount of None after
    // returning from this all function.
    bp::object ret(bp::detail::new_reference(PyArray_All((PyArrayObject*)a.ptr(), axis, (PyArrayObject*)out.ptr())));
}

ndarray
equal(ndarray const & x1, ndarray const & x2)
{
    bp::object np = bp::import("numpy");
    bp::object out_obj = np.attr("equal")(x1, x2);
    ndarray out(python::detail::borrowed_reference(out_obj.ptr()));
    return out;
}

void
equal(ndarray const & x1, ndarray const & x2, ndarray & out)
{
    bp::object np = bp::import("numpy");
    np.attr("equal")(x1, x2, out);
}

}// namespace numpy
}// namespace boost
