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

#include <boost/numpy/detail/iter_iterator.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>

namespace boost {
namespace numpy {

namespace detail {

inline
boost::shared_ptr<iter>
construct_flat_iter(
    ndarray & arr
  , iter_flags_t iter_flags
  , iter_operand_flags_t arr_op_flags
)
{
    intptr_t itershape[1];
    itershape[0] = -1;
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

template <class Derived, typename ValueType, typename ValueRefType>
class flat_iterator_base
  : public iter_iterator<Derived, ValueType, boost::random_access_traversal_tag, ValueRefType>
{
  public:
    typedef detail::iter_iterator<Derived, ValueType, boost::random_access_traversal_tag, ValueRefType>
            base_t;
    typedef typename base_t::difference_type
            difference_type;

    // The existence of the default constructor is needed by the STL
    // requirements.
    flat_iterator_base()
      : base_t()
    {}

    explicit flat_iterator_base(
        ndarray & arr
      , iter_operand_flags_t arr_access_flags
      , typename base_t::iter_construct_fct_ptr_t iter_construct_fct
    )
      : base_t(arr, arr_access_flags, iter_construct_fct)
    {}

    explicit flat_iterator_base(
        ndarray const & arr
      , iter_operand_flags_t arr_access_flags
      , typename base_t::iter_construct_fct_ptr_t iter_construct_fct
    )
      : base_t(arr, arr_access_flags, iter_construct_fct)
    {}

    void
    advance(difference_type n)
    {
        intptr_t const iteridx = base_t::iter_ptr_->get_iter_index() + n;
        base_t::iter_ptr_->jump_to_iter_index(iteridx);
    }

    difference_type
    distance_to(flat_iterator_base<Derived, ValueType, ValueRefType> const & z)
    {
        return z.iter_ptr_->get_iter_index() - base_t::iter_ptr_->get_iter_index();
    }
};

}//namespace detail

// The ValueType template parameter must be the C++ corresponding type of the
// values stored in the ndarray.
template <typename ValueType>
class flat_iterator
  : public detail::flat_iterator_base<flat_iterator<ValueType>, ValueType, ValueType &>
{
  public:
    typedef detail::flat_iterator_base<flat_iterator<ValueType>, ValueType, ValueType &>
            base_t;

    static
    boost::shared_ptr<detail::iter>
    construct_iter(detail::iter_iterator_type & iter_base, ndarray & arr)
    {
        flat_iterator<ValueType> & iter = *static_cast<flat_iterator<ValueType> *>(&iter_base);
        detail::iter_flags_t iter_flags = detail::iter::flags::C_INDEX::value
                                        | detail::iter::flags::DONT_NEGATE_STRIDES::value;
        return detail::construct_flat_iter(arr, iter_flags, iter.arr_access_flags_);
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    flat_iterator()
      : base_t()
    {}

    explicit flat_iterator(
        ndarray & arr
      , detail::iter_operand_flags_t arr_access_flags = detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags, &flat_iterator<ValueType>::construct_iter)
    {}

    // In case a constant array is given, the READONLY flag for the array will
    // be set automatically through the base iterator.
    explicit flat_iterator(
        ndarray const & arr
      , detail::iter_operand_flags_t arr_access_flags = detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags, &flat_iterator<ValueType>::construct_iter)
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
  : public detail::flat_iterator_base<flat_iterator<boost::python::object>, boost::python::object, boost::python::object>
{
  public:
    typedef detail::flat_iterator_base<flat_iterator<boost::python::object>, boost::python::object, boost::python::object>
            base_t;

    static
    boost::shared_ptr<detail::iter>
    construct_iter(detail::iter_iterator_type & iter_base, ndarray & arr)
    {
        flat_iterator<boost::python::object> & iter = *static_cast<flat_iterator<boost::python::object> *>(&iter_base);
        detail::iter_flags_t iter_flags = detail::iter::flags::C_INDEX::value
                                        | detail::iter::flags::DONT_NEGATE_STRIDES::value
                                        | detail::iter::flags::REFS_OK::value;
        return detail::construct_flat_iter(arr, iter_flags, iter.arr_access_flags_);
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    flat_iterator()
      : base_t()
    {}

    explicit flat_iterator(
        ndarray & arr
      , detail::iter_operand_flags_t arr_access_flags = detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags, &flat_iterator<boost::python::object>::construct_iter)
    {}

    explicit flat_iterator(
        ndarray const & arr
      , detail::iter_operand_flags_t arr_access_flags = detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags, &flat_iterator<boost::python::object>::construct_iter)
    {}

    boost::python::object
    dereference() const
    {
        assert(iter_ptr_.get());
        uintptr_t * data = reinterpret_cast<uintptr_t*>(iter_ptr_->get_data(0));
        boost::python::object obj(boost::python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*data)));
        return obj;
    }

    uintptr_t *
    get_object_ptr_ptr() const
    {
        return reinterpret_cast<uintptr_t *>(iter_ptr_->get_data(0));
    }

  private:
    friend class boost::iterator_core_access;
};

}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_FLAT_ITERATOR_HPP_INCLUDED

