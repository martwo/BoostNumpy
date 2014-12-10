/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/indexed_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::indexed_iterator template
 *        providing a C++ style iterator over a ndarray keeping track of
 *        indices. It provides the ``jump_to`` method to jump to a specified
 *        set of indices.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_INDEXED_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_INDEXED_ITERATOR_HPP_INCLUDED 1

#include <boost/numpy/detail/iter_iterator_base.hpp>

namespace boost {
namespace numpy {

// The ValueType template parameter must be the C++ corresponding type of the
// values stored in the ndarray.
template <typename ValueType>
class indexed_iterator
  : public detail::iter_iterator_base<indexed_iterator<ValueType>, ValueType, ValueType &>
{
  public:
    typedef detail::iter_iterator_base<indexed_iterator<ValueType>, ValueType, ValueType &>
            base_t;

    static
    boost::shared_ptr<detail::iter>
    construct_iter(detail::iterator_base & iter_base, ndarray & arr)
    {
        indexed_iterator<ValueType> & cppiter = *static_cast<indexed_iterator<ValueType> *>(&iter_base);

        int const nd = arr.get_nd();
        intptr_t itershape[nd];
        int arr_op_bcr[nd];
        for(size_t i=0; i<nd; ++i)
        {
            itershape[i] = -1;
            arr_op_bcr[i] = i;
        }
        detail::iter_flags_t iter_flags =
            detail::iter::flags::MULTI_INDEX::value
          | detail::iter::flags::DONT_NEGATE_STRIDES::value;

        detail::iter_operand_flags_t arr_op_flags = detail::iter_operand::flags::NONE::value;
        arr_op_flags |= cppiter.arr_access_flags_ & detail::iter_operand::flags::READONLY::value;
        arr_op_flags |= cppiter.arr_access_flags_ & detail::iter_operand::flags::WRITEONLY::value;
        arr_op_flags |= cppiter.arr_access_flags_ & detail::iter_operand::flags::READWRITE::value;

        detail::iter_operand arr_op(arr, arr_op_flags, arr_op_bcr);
        boost::shared_ptr<detail::iter> iter(new detail::iter(
            iter_flags
          , KEEPORDER
          , NO_CASTING
          , nd           // n_iter_axes
          , itershape
          , 0            // buffersize
          , arr_op
        ));
        iter->init_full_iteration();
        return iter;
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    indexed_iterator()
      : base_t()
    {}

    explicit indexed_iterator(
        ndarray & arr
      , detail::iter_operand_flags_t arr_access_flags = detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags, &indexed_iterator<ValueType>::construct_iter)
    {}

    ValueType &
    dereference() const
    {
        assert(base_t::iter_ptr_.get());
        return *reinterpret_cast<ValueType*>(base_t::iter_ptr_->get_data(0));
    }

    void
    jump_to(std::vector<intptr_t> const & indices)
    {
        base_t::iter_ptr_->jump_to(indices);
    }

  private:
    friend class boost::iterator_core_access;
};

// Specialization for object arrays. Here, the dereferencing works differently.
template <>
class indexed_iterator<python::object>
  : public detail::iter_iterator_base<indexed_iterator<python::object>, python::object, python::object>
{
  public:
    typedef detail::iter_iterator_base<indexed_iterator<python::object>, python::object, python::object>
            base_t;

    static
    boost::shared_ptr<detail::iter>
    construct_iter(detail::iterator_base & iter_base, ndarray & arr)
    {
        indexed_iterator<python::object> & cppiter = *static_cast<indexed_iterator<python::object> *>(&iter_base);

        int const nd = arr.get_nd();
        intptr_t itershape[nd];
        int arr_op_bcr[nd];
        for(size_t i=0; i<nd; ++i)
        {
            itershape[i] = -1;
            arr_op_bcr[i] = i;
        }
        detail::iter_flags_t iter_flags =
            detail::iter::flags::MULTI_INDEX::value
          | detail::iter::flags::REFS_OK::value
          | detail::iter::flags::DONT_NEGATE_STRIDES::value;

        detail::iter_operand_flags_t arr_op_flags = detail::iter_operand::flags::NONE::value;
        arr_op_flags |= cppiter.arr_access_flags_ & detail::iter_operand::flags::READONLY::value;
        arr_op_flags |= cppiter.arr_access_flags_ & detail::iter_operand::flags::WRITEONLY::value;
        arr_op_flags |= cppiter.arr_access_flags_ & detail::iter_operand::flags::READWRITE::value;

        detail::iter_operand arr_op(arr, arr_op_flags, arr_op_bcr);
        boost::shared_ptr<detail::iter> iter(new detail::iter(
            iter_flags
          , KEEPORDER
          , NO_CASTING
          , nd           // n_iter_axes
          , itershape
          , 0            // buffersize
          , arr_op
        ));
        iter->init_full_iteration();
        return iter;
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    indexed_iterator()
      : base_t()
    {}

    explicit indexed_iterator(
        ndarray & arr
      , detail::iter_operand_flags_t arr_access_flags = detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags, &indexed_iterator<python::object>::construct_iter)
    {}

    python::object
    dereference() const
    {
        assert(base_t::iter_ptr_.get());
        uintptr_t * ptr = reinterpret_cast<uintptr_t*>(base_t::iter_ptr_->get_data(0));
        if(*ptr == 0)
        {
            PyErr_SetString(PyExc_RuntimeError, "Dereferencing non-existing object from an object array!");
            python::throw_error_already_set();
        }
        python::object obj(python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*ptr)));
        return obj;
    }

    void
    jump_to(std::vector<intptr_t> const & indices)
    {
        base_t::iter_ptr_->jump_to(indices);
    }

  private:
    friend class boost::iterator_core_access;
};

}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_INDEXED_ITERATOR_HPP_INCLUDED
