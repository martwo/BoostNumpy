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
    typedef value_type *
            value_ptr_type;

    single_value(ndarray const &)
    {}

    static
    void
    dereference(
        value_type_traits &
      , value_ptr_type & value_ptr_ref
      , char * data_ptr
    )
    {
        value_ptr_ref = reinterpret_cast<value_ptr_type>(data_ptr);
    }
};

template <>
struct single_value<python::object>
  : value_type_traits
{
    typedef python::object
            value_type;
    typedef value_type *
            value_ptr_type;

    single_value(ndarray const &)
    {}

    // We need a temporay bp::object for holding the value.
    python::object value_obj_;

    static
    void
    dereference(
        value_type_traits & vtt_base
      , value_ptr_type & value_ptr_ref
      , char * data_ptr
    )
    {
        single_value<python::object> & vtt = *static_cast<single_value<python::object> *>(&vtt_base);
        uintptr_t * ptr = reinterpret_cast<uintptr_t*>(data_ptr);
        python::object obj(python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*ptr)));
        vtt.value_obj_ = obj;
        value_ptr_ref = &vtt.value_obj_;
    }
};

}// namespace iterators
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_ITERATORS_VALUE_TYPE_TRAITS_HPP_INCLUDED
