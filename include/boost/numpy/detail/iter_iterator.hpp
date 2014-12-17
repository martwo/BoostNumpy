/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/detail/iter_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::detail::iter_iterator
 *        template providing the base for all BoostNumpy C++ style iterators
 *        using the boost::numpy::detail::iter class.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_ITER_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_ITER_ITERATOR_HPP_INCLUDED 1

#include <boost/function.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>

namespace boost {
namespace numpy {
namespace detail {

struct iter_iterator_type
{};

template <class Derived, typename ValueType, class CategoryOrTraversal, typename ValueRefType>
class iter_iterator_base
  : public boost::iterator_facade<
        Derived
      , ValueType
      , CategoryOrTraversal
      , ValueRefType
    >
    , public iter_iterator_type
{
  public:
    typedef typename boost::iterator_facade<Derived, ValueType, CategoryOrTraversal, ValueRefType>::difference_type
            difference_type;

    typedef boost::function< boost::shared_ptr<iter> (iter_iterator_type &, ndarray &) >
            iter_construct_fct_ptr_t;

    iter_iterator_base()
      : is_end_point_(true)
      , arr_access_flags_(iter_operand::flags::READONLY::value)
    {}

    explicit iter_iterator_base(
        ndarray & arr
      , iter_operand_flags_t arr_access_flags
      , iter_construct_fct_ptr_t iter_construct_fct
    )
      : is_end_point_(false)
      , arr_access_flags_(arr_access_flags)
    {
        iter_ptr_ = iter_construct_fct(*this, arr);
    }

    // In case a const array is given, the READONLY flag for the array set
    // automatically.
    explicit iter_iterator_base(
        ndarray const & arr
      , iter_operand_flags_t arr_access_flags
      , iter_construct_fct_ptr_t iter_construct_fct
    )
      : is_end_point_(false)
      , arr_access_flags_(arr_access_flags | iter_operand::flags::READONLY::value)
    {
        iter_ptr_ = iter_construct_fct(*this, const_cast<ndarray&>(arr));
    }

    void
    increment()
    {
        if(is_end_point_)
        {
            reset();
        }
        else if(! iter_ptr_->next())
        {
            // We reached the end of the iteration. So we need to put this
            // iterator into the END state, wich is (by definition) indicated
            // through the is_end_point_ member variable set to ``true``.
            // Note: We still keep the iterator object, in case the user wants
            //       to reset the iterator and start iterating from the
            //       beginning.
            is_end_point_ = true;
        }
    }

    bool
    equal(iter_iterator_base<Derived, ValueType, CategoryOrTraversal, ValueRefType> const & other) const
    {
        if(is_end_point_ && other.is_end_point_)
        {
            return true;
        }
        // Check if one of the two iterators is the END state.
        if(is_end_point_ || other.is_end_point_)
        {
            return false;
        }
        // If the data pointers point to the same address, we are equal.
        return (iter_ptr_->get_data(0) == other.iter_ptr_->get_data(0));
    }

    bool
    reset(bool throws=true)
    {
        is_end_point_ = false;
        return iter_ptr_->reset(throws);
    }

  protected:
    boost::shared_ptr<detail::iter> iter_ptr_;
    bool is_end_point_;
    // Stores if the array is readonly, writeonly or readwrite'able.
    iter_operand_flags_t arr_access_flags_;
};

template <class Derived, typename ValueType, class CategoryOrTraversal, typename ValueRefType>
class iter_iterator
  : public iter_iterator_base<Derived, ValueType, CategoryOrTraversal, ValueRefType>
{
  public:
    typedef iter_iterator_base<Derived, ValueType, CategoryOrTraversal, ValueRefType>
            base_t;

    typedef typename base_t::difference_type
            difference_type;

    typedef typename base_t::iter_construct_fct_ptr_t
            iter_construct_fct_ptr_t;

    iter_iterator()
      : base_t()
    {}

    explicit iter_iterator(
        ndarray & arr
      , iter_operand_flags_t arr_access_flags
      , iter_construct_fct_ptr_t iter_construct_fct
    )
      : base_t(arr, arr_access_flags, iter_construct_fct)
    {}

    // In case a const array is given, the READONLY flag for the array set
    // automatically.
    explicit iter_iterator(
        ndarray const & arr
      , iter_operand_flags_t arr_access_flags
      , iter_construct_fct_ptr_t iter_construct_fct
    )
      : base_t(arr, arr_access_flags, iter_construct_fct)
    {}

    // The iterator object representing the end of the iteration process.
    // In order to have this member here, the base class is required and
    // introduced.
    base_t end;
};

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DETAIL_ITER_ITERATOR_HPP_INCLUDED
