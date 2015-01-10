/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/iterators/value_type_traits.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines pre-defined value type traits for the iterators.
 *        A pre-defined value type traits class is the
 *        boost::numpy::iterators::single_value<T> class for
 *        single value type arrays, i.e. no structured arrays.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_ITERATORS_VALUE_TYPE_TRAITS_HPP_INCLUDED
#define BOOST_NUMPY_ITERATORS_VALUE_TYPE_TRAITS_HPP_INCLUDED 1

namespace boost {
namespace numpy {
namespace iterators {

struct value_type_traits {};

template <typename ValueType>
struct single_value
  : value_type_traits
{
    typedef ValueType
            value_type;
    typedef value_type &
            value_ref_type;
    typedef value_type *
            value_ptr_type;

    single_value()
    {}

    single_value(ndarray const &)
    {}

    static
    value_ref_type
    dereference(
        value_type_traits &
      , char * data_ptr
    )
    {
        value_ref_type value = *reinterpret_cast<value_ptr_type>(data_ptr);
        return value;
    }
};

template <>
struct single_value<python::object>
  : value_type_traits
{
    typedef python::object
            value_type;
    typedef value_type &
            value_ref_type;
    typedef value_type *
            value_ptr_type;

    single_value()
    {}

    single_value(ndarray const &)
    {}

    // We need a temporay bp::object for holding the value.
    python::object value_obj_;
    uintptr_t * value_obj_ptr_;

    static
    value_ref_type
    dereference(
        value_type_traits & vtt_base
      , char * data_ptr
    )
    {
        single_value<python::object> & vtt = *static_cast<single_value<python::object> *>(&vtt_base);

        vtt.value_obj_ptr_ = reinterpret_cast<uintptr_t*>(data_ptr);
        vtt.value_obj_ = python::object(python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*vtt.value_obj_ptr_)));
        return vtt.value_obj_;
    }
};

}// namespace iterators
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_ITERATORS_VALUE_TYPE_TRAITS_HPP_INCLUDED
