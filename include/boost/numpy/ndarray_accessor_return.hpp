/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/ndarray_accessor_return.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::ndarray_accessor_return call
 *        policy. It can be used when a member function returns a
 *        boost::numpy::ndarray whose data is owned by the member function's
 *        class instance. This call policy will ensure, that the data owner of
 *        the returned ndarray object is set to the class instance, and that
 *        the class instance lives as long as the returned ndarray is alive.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_NDARRAY_ACCESSOR_RETURN_HPP_INCLUDED
#define BOOST_NUMPY_NDARRAY_ACCESSOR_RETURN_HPP_INCLUDED

#include <boost/python/default_call_policies.hpp>

#include <boost/numpy/ndarray.hpp>

namespace boost {
namespace numpy {

struct ndarray_accessor_return
  : python::default_call_policies
{
    template <class ArgumentPackage>
    static PyObject* postcall(ArgumentPackage const& args_, PyObject* result)
    {
        PyObject* py_cls_instance = python::detail::get_prev<1>::execute(args_, result);
        python::object cls_instance = python::object(python::detail::borrowed_reference(py_cls_instance));

        ndarray arr = ndarray(python::detail::borrowed_reference(result));
        arr.set_base(cls_instance);

        return result;
    }
};

}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_NDARRAY_ACCESSOR_RETURN_HPP_INCLUDED
