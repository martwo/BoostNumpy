/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/detail/iter_iterator_base.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::detail::iter_iterator_base
 *        template providing the base for all BoostNumpy C++ style iterators.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_ITER_ITERATOR_BASE_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_ITER_ITERATOR_BASE_HPP_INCLUDED 1

#include <boost/function.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>

namespace boost {
namespace numpy {
namespace detail {

template <class Derived, typename ValueType, typename ValueRefType>
class iter_iterator_base
  : public boost::iterator_facade<
        Derived                      // Derived
      , ValueType                    // Value
      , boost::forward_traversal_tag // CategoryOrTraversal
      , ValueRefType                 // Reference
    >
{
  public:
    typedef boost::function<boost::shared_ptr<iter> (ndarray &)>
            iter_construct_fct_ptr_t;

    iter_iterator_base()
      : is_end_point_(true)
    {}

    explicit iter_iterator_base(ndarray & arr, iter_construct_fct_ptr_t iter_construct_fct)
    {
        iter_ptr_ = iter_construct_fct(arr);
        is_end_point_ = false;
    }

    void
    increment()
    {
        if(! iter_ptr_->next())
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
    equal(iter_iterator_base<Derived, ValueType, ValueRefType> const & other) const
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
};

}//namespace detail
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DETAIL_ITER_ITERATOR_BASE_HPP_INCLUDED
