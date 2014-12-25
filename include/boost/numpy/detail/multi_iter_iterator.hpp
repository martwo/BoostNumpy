/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/detail/multi_iter_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::detail::multi_iter_iterator
 *        template providing the base for all BoostNumpy C++ style iterators
 *        using the boost::numpy::detail::iter class for multiple ndarrays.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_MULTI_ITER_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_MULTI_ITER_ITERATOR_HPP_INCLUDED 1

#include <boost/preprocessor/repetition/enum.hpp>

#include <boost/function.hpp>
#include <boost/iterator/iterator_facade.hpp>

#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>

namespace boost {
namespace numpy {
namespace detail {

struct multi_iter_iterator_type
{};

template <int nd>
struct multi_iter_iterator
{};

#define ND 3

template <>
struct multi_iter_iterator<ND>
{
    template <class Derived, class CategoryOrTraversal>
    class iter
      : public boost::iterator_facade<
          Derived
        , bool // ValueType
        , CategoryOrTraversal
        , bool // ValueRefType
        >
      , public multi_iter_iterator_type
    {
      public:
        typedef multi_iter_iterator<ND>::iter<Derived, CategoryOrTraversal>
                type_t;
        typedef typename boost::iterator_facade<Derived, bool, CategoryOrTraversal, bool>::difference_type
                difference_type;

        #define BOOST_NUMPY_DEF(z, n, data) data
        typedef boost::function< boost::shared_ptr<boost::numpy::detail::iter> (multi_iter_iterator_type &, BOOST_PP_ENUM(ND, BOOST_NUMPY_DEF, ndarray &)) >
                iter_construct_fct_ptr_t;
        #undef BOOST_NUMPY_DEF

        // Default constructor.
        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(arr_access_flags_,n)(iter_operand::flags::READONLY::value)
        iter()
          : is_end_point_(true)
          , BOOST_PP_ENUM(ND, BOOST_NUMPY_DEF, ~)
        {}
        #undef BOOST_NUMPY_DEF

        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(arr_access_flags_,n)(BOOST_PP_CAT(arr_access_flags,n))
        explicit iter(
            BOOST_PP_ENUM_PARAMS(ND, ndarray & arr)
          , BOOST_PP_ENUM_PARAMS(ND, iter_operand_flags_t arr_access_flags)
          , iter_construct_fct_ptr_t iter_construct_fct
        )
          : is_end_point_(false)
          , BOOST_PP_ENUM(ND, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
        {
            iter_ptr_ = iter_construct_fct(*this, BOOST_PP_ENUM_PARAMS(ND, arr));
        }

        // Copy constructor.
        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(arr_access_flags_,n)(other.BOOST_PP_CAT(arr_access_flags_,n))
        iter(type_t const & other)
          : is_end_point_(other.is_end_point_)
          , BOOST_PP_ENUM(ND, BOOST_NUMPY_DEF, ~)
        {
            if(other.iter_ptr_.get()) {
                iter_ptr_ = boost::shared_ptr<detail::iter>(new detail::iter(*other.iter_ptr_));
            }
        }

        // Creates an interator that points to the first element.
        Derived begin() const
        {
            Derived it(*static_cast<Derived*>(const_cast<type_t*>(this)));
            it.reset();
            return it;
        }

        // Creates an iterator that points to the element after the last element.
        Derived end() const
        {
            Derived it(*static_cast<Derived*>(const_cast<type_t*>(this)));
            it.is_end_point_ = true;
            return it;
        }

        void
        increment()
        {
            if(is_end())
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
        equal(type_t const & other) const
        {
            //std::cout << "iter_iterator: equal" << std::endl;
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
            //std::cout << "iter_iterator: reset" << std::endl;
            is_end_point_ = false;
            return iter_ptr_->reset(throws);
        }

        bool
        is_end() const
        {
            return is_end_point_;
        }

      protected:
        boost::shared_ptr<detail::iter> iter_ptr_;
        bool is_end_point_;
        // Stores if the array is readonly, writeonly or readwrite'able.
        #define BOOST_NUMPY_DEF(z, n, data) \
            iter_operand_flags_t BOOST_PP_CAT(arr_access_flags_,n);
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
    };
};

#undef ND

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DETAIL_MULTI_ITER_ITERATOR_HPP_INCLUDED
