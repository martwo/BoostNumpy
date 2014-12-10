/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/flat_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::flat_iterator template providing
 *        a C++ style iterator over a flatted ndarray.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_FLAT_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_FLAT_ITERATOR_HPP_INCLUDED 1

#include <boost/numpy/detail/iter_iterator_base.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>

namespace boost {
namespace numpy {

namespace detail {

inline
boost::shared_ptr<iter>
construct_flat_iter(ndarray & arr, iter_flags_t iter_flags)
{
    intptr_t itershape[1];
    itershape[0] = -1;
    iter_operand_flags_t arr_op_flags = iter_operand::flags::READONLY::value;
    int arr_op_bcr[1];
    arr_op_bcr[0] = 0;
    detail::iter_operand arr_op(arr, arr_op_flags, arr_op_bcr);
    boost::shared_ptr<iter> iter_ptr = boost::shared_ptr<iter>(new iter(
        iter_flags
      , KEEPORDER
      , NO_CASTING
      , 1           // n_iter_axes
      , itershape
      , 0           // buffersize
      , arr_op
    ));
    iter_ptr->init_full_iteration();

    return iter_ptr;
}

}//namespace detail

// The ValueType template parameter must be the C++ corresponding type of the
// values stored in the ndarray.
template <typename ValueType>
class flat_iterator
  : public detail::iter_iterator_base<flat_iterator<ValueType>, ValueType, ValueType &>
{
  public:
    typedef detail::iter_iterator_base<flat_iterator<ValueType>, ValueType, ValueType &>
            base_t;

    static
    boost::shared_ptr<detail::iter>
    construct_iter(ndarray & arr)
    {
        return detail::construct_flat_iter(arr, detail::iter::flags::DONT_NEGATE_STRIDES::value);
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    flat_iterator()
      : base_t()
    {}

    explicit flat_iterator(ndarray & arr)
      : base_t(arr, &flat_iterator<ValueType>::construct_iter)
    {}

    ValueType &
    dereference() const
    {
        assert(base_t::iter_ptr_.get());
        return *reinterpret_cast<ValueType*>(base_t::iter_ptr_->get_data(0));
    }

  private:
    friend class boost::iterator_core_access;
};

// Specialization for boost::python::object. In this case the dereferencing
// returns an object (i.e. bp::object) and not a reference.
template <>
class flat_iterator<boost::python::object>
  : public detail::iter_iterator_base<flat_iterator<boost::python::object>, boost::python::object, boost::python::object>
{
  public:
    typedef detail::iter_iterator_base<flat_iterator<boost::python::object>, boost::python::object, boost::python::object>
            base_t;

    static
    boost::shared_ptr<detail::iter>
    construct_iter(ndarray & arr)
    {
        return detail::construct_flat_iter(arr,   detail::iter::flags::DONT_NEGATE_STRIDES::value
                                                | detail::iter::flags::REFS_OK::value);
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    flat_iterator()
      : base_t()
    {}

    explicit flat_iterator(ndarray & arr)
      : base_t(arr, &flat_iterator<boost::python::object>::construct_iter)
    {}

    boost::python::object
    dereference() const
    {
        assert(iter_ptr_.get());
        uintptr_t * data = reinterpret_cast<uintptr_t*>(iter_ptr_->get_data(0));
        boost::python::object obj(boost::python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*data)));
        return obj;
    }

    uintptr_t*
    get_object_ptr_ptr() const
    {
        return reinterpret_cast<uintptr_t*>(iter_ptr_->get_data(0));
    }

  private:
    friend class boost::iterator_core_access;
};

}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_FLAT_ITERATOR_HPP_INCLUDED

