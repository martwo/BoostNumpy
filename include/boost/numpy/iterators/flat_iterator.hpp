/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/iterators/flat_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the
 *        boost::numpy::iterators::flat_iterator template providing a
 *        C++ style iterator over a single ndarray.
 *        It is a random access iterator.
 *        The value type of the operand is specified via a value type traits
 *        class, which also provides the appropriate dereferencing procedure.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_ITERATORS_FLAT_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_ITERATORS_FLAT_ITERATOR_HPP_INCLUDED 1

#include <boost/numpy/iterators/detail/iter_iterator.hpp>

namespace boost {
namespace numpy {
namespace iterators {

template <typename ValueTypeTraits>
class flat_iterator
  : public detail::iter_iterator<flat_iterator<ValueTypeTraits>, boost::random_access_traversal_tag, ValueTypeTraits>
{
  public:
    typedef flat_iterator<ValueTypeTraits>
            type_t;
    typedef detail::iter_iterator<flat_iterator<ValueTypeTraits>, boost::random_access_traversal_tag, ValueTypeTraits>
            base_t;
    typedef typename base_t::difference_type
            difference_type;

    static
    boost::shared_ptr<boost::numpy::detail::iter>
    construct_iter(detail::iter_iterator_type & iter_base, ndarray & arr)
    {
        return base_t::construct_iter(
            iter_base
          , boost::numpy::detail::iter::flags::C_INDEX::value
          , arr
        );
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    flat_iterator()
        : base_t()
    {}

    // Explicit constructor.
    explicit flat_iterator(
        ndarray & arr
      , boost::numpy::detail::iter_operand_flags_t arr_access_flags = boost::numpy::detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags)
    {}

    // In case a const array is given, the READONLY flag for the array set
    // automatically.
    explicit flat_iterator(
        ndarray const & arr
      , boost::numpy::detail::iter_operand_flags_t arr_access_flags = boost::numpy::detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags)
    {}

    // Copy constructor.
    flat_iterator(type_t const & other)
      : base_t(other)
    {}

    intptr_t
    get_iter_index() const
    {
        if(base_t::is_end())
        {
            return base_t::iter_ptr_->get_iter_size();
        }

        return base_t::iter_ptr_->get_iter_index();
    }

    void
    advance(difference_type n)
    {
        intptr_t const iteridx = get_iter_index() + n;
        if(iteridx < 0)
        {
            base_t::reset();
            return;
        }
        else if(iteridx >= base_t::iter_ptr_->get_iter_size())
        {
            base_t::is_end_point_ = true;
            return;
        }

        base_t::iter_ptr_->jump_to_iter_index(iteridx);
    }

    difference_type
    distance_to(type_t const & z) const
    {
        return z.get_iter_index() - get_iter_index();
    }

    void
    jump_to_iter_index(intptr_t iteridx)
    {
        base_t::iter_ptr_->jump_to_iter_index(iteridx);
    }

  private:
    friend class boost::iterator_core_access;
};

}// namespace iterators
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_ITERATORS_FLAT_ITERATOR_HPP_INCLUDED
