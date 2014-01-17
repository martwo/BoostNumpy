/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 * 2010-2012
 *     Jim Bosch
 *
 * @file boost/numpy/scalars.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <martin.wolf@icecube.wisc.edu>
 * @brief This file contains the implementation of the numpy.void array scalar
 *        type in boost::python.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#define BOOST_NUMPY_INTERNAL_IMPL
#include <boost/numpy/internal_impl.hpp>

#include <boost/python.hpp>

#include <boost/numpy/numpy_c_api.hpp>
#include <boost/numpy/object_manager_traits_impl.hpp>
#include <boost/numpy/scalars.hpp>

BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL(boost::numpy::void_, PyVoidArrType_Type);

namespace boost {
namespace numpy {

//______________________________________________________________________________
void_::
void_(Py_ssize_t size)
  : object(python::detail::new_reference(PyObject_CallFunction((PyObject*)&PyVoidArrType_Type, const_cast<char*>("i"), size)))
{}

//______________________________________________________________________________
void_
void_::
view(dtype const & dt) const
{
    return void_(python::detail::new_reference
        (PyObject_CallMethod(this->ptr(), const_cast<char*>("view"), const_cast<char*>("O"), dt.ptr())));
}

//______________________________________________________________________________
void_
void_::
copy() const
{
    return void_(python::detail::new_reference
        (PyObject_CallMethod(this->ptr(), const_cast<char*>("copy"), NULL)));
}

}/*numpy*/
}/*boost*/
