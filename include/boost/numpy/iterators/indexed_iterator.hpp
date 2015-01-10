/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/iterators/indexed_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the
 *        boost::numpy::iterators::indexed_iterator template
 *        providing a C++ style iterator over a single ndarray keeping track
 *        of indices. It provides the ``jump_to`` method to jump to a specified
 *        set of indices.
 *        The value type of the operand is specified via a value type traits
 *        class, which also provides the appropriate dereferencing procedure.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_ITERATORS_INDEXED_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_ITERATORS_INDEXED_ITERATOR_HPP_INCLUDED 1

#include <vector>
#include <boost/iterator/iterator_categories.hpp>
#include <boost/shared_ptr.hpp>

#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/iterators/detail/iter_iterator.hpp>

namespace boost {
namespace numpy {
namespace iterators {

template <typename ValueTypeTraits>
class indexed_iterator
  : public detail::iter_iterator< indexed_iterator<ValueTypeTraits>, boost::forward_traversal_tag, ValueTypeTraits>
{
  public:
    typedef indexed_iterator<ValueTypeTraits>
            type_t;
    typedef detail::iter_iterator< indexed_iterator<ValueTypeTraits>, boost::forward_traversal_tag, ValueTypeTraits>
            base_t;

    typedef typename ValueTypeTraits::value_ref_type
            value_ref_type;

    static
    boost::shared_ptr<boost::numpy::detail::iter>
    construct_iter(detail::iter_iterator_type & iter_base, ndarray & arr)
    {
        return base_t::construct_iter(
            iter_base
          , boost::numpy::detail::iter::flags::MULTI_INDEX::value
          , arr
        );
    }

    // The existence of the default constructor is needed by the STL
    // requirements.
    indexed_iterator()
      : base_t()
    {}

    // Explicit constructor.
    explicit indexed_iterator(
        ndarray & arr
      , boost::numpy::detail::iter_operand_flags_t arr_access_flags = boost::numpy::detail::iter_operand::flags::READONLY::value
    )
      : base_t(arr, arr_access_flags)
    {}

    // Copy constructor.
    indexed_iterator(type_t const & other)
      : base_t(other)
    {}

    void
    jump_to(std::vector<intptr_t> const & indices)
    {
        base_t::iter_ptr_->jump_to(indices);
    }

    void
    get_indices(std::vector<intptr_t> & indices)
    {
        base_t::iter_ptr_->get_multi_index_vector(indices);
    }

  private:
    friend class boost::iterator_core_access;
};

}// namespace iterators
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_ITERATORS_INDEXED_ITERATOR_HPP_INCLUDED
