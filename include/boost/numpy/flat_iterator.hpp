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

#include <boost/iterator/iterator_facade.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>

namespace boost {
namespace numpy {

namespace detail {

inline
boost::shared_ptr<iter>
construct_flat_iter(ndarray & arr)
{
    iter_flags_t iter_flags = iter::flags::REFS_OK::value
                            | iter::flags::DONT_NEGATE_STRIDES::value;
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

template <typename ValueType>
ValueType &
dereference(boost::shared_ptr<detail::iter> const & iter_ptr)
{
    return *reinterpret_cast<ValueType*>(iter_ptr->get_data(0));
}

inline
template <>
boost::python::object &
dereference(boost::shared_ptr<detail::iter> const & iter_ptr)
{
    uintptr_t * data = reinterpret_cast<uintptr_t*>(iter_ptr->get_data(0));
    boost::python::object obj(boost::python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*data)));
    return obj;
}

}//namespace detail

// The ValueType template parameter must be the C++ corresponding type of the
// values stored in the ndarray.
template <typename ValueType>
class flat_iterator
  : public boost::iterator_facade<
        flat_iterator<ValueType>
      , ValueType
      , boost::forward_traversal_tag
    >
{
  public:
    // The existence of the default constructor is needed by the STL
    // requirements.
    flat_iterator()
      : iter_ptr_(boost::shared_ptr<detail::iter>())
    {}

    explicit flat_iterator(ndarray & arr)
    {
        // Construct a iterator on the heap which keeps the state of the current
        // iteration.
        this->iter_ptr_ = detail::construct_flat_iter(arr);
    }

    void
    increment()
    {
        if(! iter_ptr_->next())
        {
            // We reached the end of the iteration. So we need to put this
            // iterator into the END state, wich is (by definition) indicated
            // that iter_ptr_ is not hold a pointer.
            // Deallocate the iter object by forgetting about it.
            iter_ptr_ = boost::shared_ptr<detail::iter>();
        }
    }

    bool
    equal(flat_iterator<ValueType> const & other) const
    {
        detail::iter * const p1 = iter_ptr_.get();
        detail::iter * const p2 = other.iter_ptr_.get();
        if(p1 == NULL && p2 == NULL)
        {
            return true;
        }
        // Check if one of the two iterators is the END state.
        if(p1 == NULL || p2 == NULL)
        {
            return false;
        }
        // If the data pointers point to the same address, we are equal.
        return (iter_ptr_->get_data(0) == other.iter_ptr_->get_data(0));
    }

    ValueType &
    dereference() const
    {
        assert(this->iter_ptr_.get());
        return detail::dereference<ValueType>(this->iter_ptr_);
    }

  private:
    friend class boost::iterator_core_access;

    boost::shared_ptr<detail::iter> iter_ptr_;
};

}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_FLAT_ITERATOR_HPP_INCLUDED

