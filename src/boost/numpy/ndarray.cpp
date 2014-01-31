/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 * 2010-2012
 *     Jim Bosch
 *
 * @file    boost/numpy/ndarray.cpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>,
 *          Jim Bosch
 *
 * @brief This file implements the boost::numpy::ndarray object manager.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#define BOOST_NUMPY_INTERNAL_IMPL
#include <boost/numpy/internal_impl.hpp>

#include <stdint.h>

#include <vector>

#include <boost/python/refcount.hpp>
#include <boost/python/object_protocol.hpp>

#include <boost/numpy/numpy_c_api.hpp>
#include <boost/numpy/object_manager_traits_impl.hpp>
#include <boost/numpy/ndarray.hpp>

BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL(boost::numpy::ndarray, PyArray_Type);

namespace boost {
namespace numpy {

namespace detail {

//______________________________________________________________________________
inline
ndarray::flags
npy_array_flags_to_bn_ndarray_flags(int flags)
{
    return ndarray::flags(flags);
}

//______________________________________________________________________________
inline
int
bn_ndarray_flags_to_npy_array_flags(ndarray::flags flags)
{
    return int(flags);
}

//______________________________________________________________________________
bool
is_c_contiguous(
    std::vector<Py_intptr_t> const & shape,
    std::vector<Py_intptr_t> const & strides,
    int                              itemsize)
{
    std::vector<Py_intptr_t>::const_reverse_iterator i = shape.rbegin();
    std::vector<Py_intptr_t>::const_reverse_iterator j = strides.rbegin();
    int total = itemsize;
    for(; i != shape.rend(); ++i, ++j)
    {
        if(total != *j) {
            return false;
        }
        total *= (*i);
    }
    return true;
}

//______________________________________________________________________________
bool
is_f_contiguous(
    std::vector<Py_intptr_t> const & shape,
    std::vector<Py_intptr_t> const & strides,
    int                              itemsize)
{
    std::vector<Py_intptr_t>::const_iterator i = shape.begin();
    std::vector<Py_intptr_t>::const_iterator j = strides.begin();
    int total = itemsize;
    for(; i != shape.end(); ++i, ++j)
    {
        if(total != *j) {
            return false;
        }
        total *= (*i);
    }
    return true;
}

//______________________________________________________________________________
bool
is_aligned(
    std::vector<Py_intptr_t> const & strides,
    int                              itemsize)
{
    std::vector<Py_intptr_t>::const_iterator i = strides.begin();
    for(; i != strides.end(); ++i)
    {
        if(*i % itemsize) {
            return false;
        }
    }
    return true;
}

//______________________________________________________________________________
inline
PyArray_Descr *
incref_dtype(dtype const & dt)
{
    Py_INCREF(dt.ptr());
    return reinterpret_cast<PyArray_Descr*>(dt.ptr());
}

//______________________________________________________________________________
ndarray
from_data_impl(
    void *                           data,
    dtype const &                    dt,
    std::vector<Py_intptr_t> const & shape,
    std::vector<Py_intptr_t> const & strides,
    python::object const &           owner,
    bool                             writeable)
{
    if(shape.size() != strides.size())
    {
        PyErr_SetString(PyExc_ValueError,
            "Length of shape and strides arrays do not match.");
        python::throw_error_already_set();
    }
    int itemsize = dt.get_itemsize();

    // Calculate the array flags.
    ndarray::flags flags = ndarray::NONE;
    if(writeable)
        flags = flags | ndarray::WRITEABLE;
    if(is_c_contiguous(shape, strides, itemsize))
        flags = flags | ndarray::C_CONTIGUOUS;
    if(is_f_contiguous(shape, strides, itemsize))
        flags = flags | ndarray::F_CONTIGUOUS;
    if(is_aligned(strides, itemsize))
        flags = flags | ndarray::ALIGNED;
    if(owner == python::object())
        flags = flags | ndarray::OWNDATA;

    ndarray arr(python::detail::new_reference(
        PyArray_NewFromDescr(
            &PyArray_Type,
            incref_dtype(dt),
            int(shape.size()),
            const_cast<Py_intptr_t*>(&shape.front()),
            const_cast<Py_intptr_t*>(&strides.front()),
            data,
            bn_ndarray_flags_to_npy_array_flags(flags),
            NULL)));
    arr.set_base(owner);
    return arr;
}

//______________________________________________________________________________
ndarray
from_data_impl(
    void *                 data,
    dtype const &          dt,
    python::object const & shape,
    python::object const & strides,
    python::object const & owner,
    bool                   writeable)
{
    std::vector<Py_intptr_t> shape_(python::len(shape));
    std::vector<Py_intptr_t> strides_(python::len(strides));
    if(shape_.size() != strides_.size())
    {
        PyErr_SetString(PyExc_ValueError,
            "Length of shape and strides arrays do not match.");
        python::throw_error_already_set();
    }
    for(std::size_t i=0; i<shape_.size(); ++i)
    {
        shape_[i]   = python::extract<Py_intptr_t>(shape[i]);
        strides_[i] = python::extract<Py_intptr_t>(strides[i]);
    }
    return from_data_impl(data, dt, shape_, strides_, owner, writeable);
}

}/*namespace detail*/

//______________________________________________________________________________
ndarray
ndarray::
view(dtype const & dt) const
{
    return ndarray(python::detail::new_reference(
        PyObject_CallMethod(this->ptr(), const_cast<char*>("view"), const_cast<char*>("O"), dt.ptr())));
}

//______________________________________________________________________________
ndarray
ndarray::
copy(std::string order) const
{
    return ndarray(python::detail::new_reference(
        PyObject_CallMethod(this->ptr(), const_cast<char*>("copy"), const_cast<char*>(order.c_str()))));
}

//______________________________________________________________________________
dtype
ndarray::
get_dtype() const
{
    return dtype(python::detail::borrowed_reference(PyArray_DESCR((PyArrayObject*)this->ptr())));
}

//______________________________________________________________________________
python::object
ndarray::
get_base() const
{
    PyObject* base = PyArray_BASE((PyArrayObject*)this->ptr());
    if(base == NULL)
        return python::object();
    return python::object(python::detail::borrowed_reference(base));
}

//______________________________________________________________________________
void
ndarray::
set_base(python::object const & base)
{
    boost::python::xincref(base.ptr());

#if NPY_FEATURE_VERSION >= 0x00000007
    if(PyArray_SetBaseObject((PyArrayObject*)this->ptr(), base.ptr()))
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not set base of PyArrayObject!");
        boost::python::throw_error_already_set();
    }
#else
    if(base != python::object())
    {
        PyArray_BASE((PyArrayObject*)this->ptr()) = base.ptr();
    }
    else
    {
        PyArray_BASE((PyArrayObject*)this->ptr()) = NULL;
    }
#endif
}

//______________________________________________________________________________
ndarray::flags const
ndarray::
get_flags() const
{
    return detail::npy_array_flags_to_bn_ndarray_flags(PyArray_FLAGS((PyArrayObject*)this->ptr()));
}

//______________________________________________________________________________
ndarray
ndarray::
transpose() const
{
    return ndarray(python::detail::new_reference(
        PyArray_Transpose(reinterpret_cast<PyArrayObject*>(this->ptr()), NULL)));
}

//______________________________________________________________________________
ndarray
ndarray::
squeeze() const
{
    return ndarray(python::detail::new_reference(
        PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(this->ptr()))));
}

//______________________________________________________________________________
ndarray
ndarray::
reshape(python::tuple const & shape) const
{
    return ndarray(python::detail::new_reference(
        PyArray_Reshape(reinterpret_cast<PyArrayObject*>(this->ptr()), shape.ptr())));
}

//______________________________________________________________________________
ndarray
ndarray::
reshape(python::list const & shape) const
{
    return ndarray(python::detail::new_reference(
        PyArray_Reshape(reinterpret_cast<PyArrayObject*>(this->ptr()), shape.ptr())));
}

//______________________________________________________________________________
python::object
ndarray::
scalarize() const
{
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(ptr());

    // Check if we got a 1x1 ndarray.
    if(this->get_nd() == 1 && this->shape(0) == 1)
    {
        return python::object(python::detail::new_reference(
            PyArray_ToScalar(PyArray_DATA(arr), arr)));
    }

    // This ndarray is either a 0-dimensional array or something else than 1x1.
    // The reference count is decremented by PyArray_Return so we need to
    // increment it first.
    Py_INCREF(ptr());
    return python::object(python::detail::new_reference(
        PyArray_Return(arr)));
}

//______________________________________________________________________________
bool
ndarray::
has_shape(std::vector<intptr_t> const & shape) const
{
    int const nd = this->get_nd();
    if(nd != int(shape.size())) {
        return false;
    }
    intptr_t const * this_shape = this->get_shape();
    for(int i=0; i<nd; ++i)
    {
        if(this_shape[i] != shape[i]) {
            return false;
        }
    }
    return true;
}

//______________________________________________________________________________
ndarray &
ndarray::
operator=(ndarray::object_cref rhs)
{
    python::object::operator=(rhs);
    return *this;
}

//______________________________________________________________________________
ndarray
ndarray::
operator[](ndarray::object_cref obj) const
{
    python::object item = python::api::getitem((python::object)*this, obj);
    if(! PyArray_Check(item.ptr()))
    {
        PyErr_SetString(PyExc_TypeError,
            "ndarray::operator[]: The item object is not a sub-type of "
            "PyArray_Type!");
        python::throw_error_already_set();
    }
    ndarray arr = from_object(item);
    return arr;
}

//______________________________________________________________________________
ndarray
zeros(
    int              nd,
    intptr_t const * shape,
    dtype const &    dt)
{
    return ndarray(python::detail::new_reference(
        PyArray_Zeros(nd, const_cast<intptr_t*>(shape), detail::incref_dtype(dt), 0)));
}

//______________________________________________________________________________
ndarray
zeros(
    python::tuple const & shape,
    dtype const &         dt)
{
    int nd = python::len(shape);
    intptr_t dims[nd];
    for(int n=0; n<nd; ++n) {
        dims[n] = python::extract<intptr_t>(shape[n]);
    }
    return zeros(nd, dims, dt);
}

//______________________________________________________________________________
ndarray
zeros(std::vector<intptr_t> const & shape, dtype const & dt)
{
    return zeros(int(shape.size()), &(shape.front()), dt);
}

//______________________________________________________________________________
ndarray
empty(
    int              nd,
    intptr_t const * shape,
    dtype const &    dt)
{
    return ndarray(python::detail::new_reference(
        PyArray_Empty(nd, const_cast<intptr_t*>(shape), detail::incref_dtype(dt), 0)));
}

//______________________________________________________________________________
ndarray
empty(
    python::tuple const & shape,
    dtype const &         dt)
{
    int nd = python::len(shape);
    Py_intptr_t dims[nd];
    for(int n=0; n<nd; ++n) {
        dims[n] = python::extract<intptr_t>(shape[n]);
    }
    return empty(nd, dims, dt);
}

//______________________________________________________________________________
ndarray
array(python::object const & obj)
{
    // We need to set the ndarray::ENSUREARRAY flag here, because the array
    // function is supposed to construct a numpy.ndarray object and not
    // something else.
    return ndarray(python::detail::new_reference(
        PyArray_FromAny(obj.ptr(), NULL, 0, 0, ndarray::ENSUREARRAY, NULL)));
}

//______________________________________________________________________________
ndarray
array(python::object const & obj, dtype const & dt)
{
    // We need to set the ndarray::ENSUREARRAY flag here, because the array
    // function is supposed to construct a numpy.ndarray object and not
    // something else.
    return ndarray(python::detail::new_reference(
        PyArray_FromAny(obj.ptr(), detail::incref_dtype(dt), 0, 0, ndarray::ENSUREARRAY, NULL)));
}

//______________________________________________________________________________
ndarray
from_object(
    python::object const & obj,
    dtype const &          dt,
    int                    nd_min,
    int                    nd_max,
    ndarray::flags         flags)
{
    int requirements = detail::bn_ndarray_flags_to_npy_array_flags(flags);
    return ndarray(python::detail::new_reference(
        PyArray_FromAny(
            obj.ptr(),
            detail::incref_dtype(dt),
            nd_min,
            nd_max,
            requirements,
            NULL)));
}

//______________________________________________________________________________
ndarray
from_object(
    python::object const & obj,
    int                    nd_min,
    int                    nd_max,
    ndarray::flags         flags)
{
    int requirements = detail::bn_ndarray_flags_to_npy_array_flags(flags);
    return ndarray(python::detail::new_reference(
        PyArray_FromAny(
            obj.ptr(),
            NULL,
            nd_min,
            nd_max,
            requirements,
            NULL)));
}

}/*namespace numpy*/
}/*namespace boost*/
