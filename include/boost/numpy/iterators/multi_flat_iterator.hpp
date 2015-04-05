/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/iterators/multi_flat_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the
 *        boost::numpy::iterators::multi_flat_iterator<ARITY>::impl
 *        template providing a C++ style iterator over multiple ndarrays.
 *        It is a random access iterator.
 *        Due to the multiple operands, the dereference method
 *        always just returns a boolean value, but the individual values can
 *        be accessed through the value_ptr0, value_ptr1, ... member variables
 *        being pointers to the individual data values.
 *        The value type of each operand is specified via a value type traits
 *        class, which also provides the appropriate dereferencing procedure.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_ITERATORS_MULTI_FLAT_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_ITERATORS_MULTI_FLAT_ITERATOR_HPP_INCLUDED 1

#include <boost/preprocessor/iterate.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/iterators/detail/multi_iter_iterator.hpp>

namespace boost {
namespace numpy {
namespace iterators {

template <int n>
struct multi_flat_iterator
{};

#define BOOST_PP_ITERATION_PARAMS_1 \
    (4, (2, BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY, <boost/numpy/iterators/multi_flat_iterator.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace iterators
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_ITERATORS_MULTI_FLAT_ITERATOR_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define N BOOST_PP_ITERATION()

template <>
struct multi_flat_iterator<N>
{
    template <BOOST_PP_ENUM_PARAMS(N, typename ValueTypeTraits)>
    class impl
      : public detail::multi_iter_iterator<N>::impl< multi_flat_iterator<N>::impl<BOOST_PP_ENUM_PARAMS(N, ValueTypeTraits)>, boost::random_access_traversal_tag, BOOST_PP_ENUM_PARAMS(N, ValueTypeTraits)>
    {
      public:
        typedef multi_flat_iterator<N>::impl<BOOST_PP_ENUM_PARAMS(N, ValueTypeTraits)>
                type_t;
        typedef detail::multi_iter_iterator<N>::impl< multi_flat_iterator<N>::impl<BOOST_PP_ENUM_PARAMS(N, ValueTypeTraits)>, boost::random_access_traversal_tag, BOOST_PP_ENUM_PARAMS(N, ValueTypeTraits)>
                base_t;
        typedef typename base_t::difference_type
                difference_type;

        // Define typedefs value_type_# for each operand.
        #define BOOST_NUMPY_DEF(z, n, data) \
            typedef typename BOOST_PP_CAT(ValueTypeTraits,n)::value_type \
                    BOOST_PP_CAT(value_type_,n);
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF

        // Define typedefs value_ref_type_# for each operand.
        #define BOOST_NUMPY_DEF(z, n, data) \
            typedef typename BOOST_PP_CAT(ValueTypeTraits,n)::value_ref_type \
                    BOOST_PP_CAT(value_ref_type_,n);
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF

        static
        boost::shared_ptr<boost::numpy::detail::iter>
        construct_iter(detail::multi_iter_iterator_type & iter_base, BOOST_PP_ENUM_PARAMS(N, ndarray & arr))
        {
            return base_t::construct_iter(
                iter_base
              , boost::numpy::detail::iter::flags::C_INDEX::value
              , BOOST_PP_ENUM_PARAMS(N, arr)
            );
        }

        // The existence of the default constructor is needed by the STL
        // requirements.
        impl()
          : base_t()
        {}

        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_COMMA_IF(n) boost::numpy::detail::iter_operand_flags_t BOOST_PP_CAT(arr_access_flags,n) = boost::numpy::detail::iter_operand::flags::READONLY::value
        explicit impl(
            BOOST_PP_ENUM_PARAMS(N, ndarray & arr)
          , BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
        )
        #undef BOOST_NUMPY_DEF
          : base_t(BOOST_PP_ENUM_PARAMS(N, arr), BOOST_PP_ENUM_PARAMS(N, arr_access_flags))
        {}

        // Copy constructor.
        impl(type_t const & other)
          : base_t(other)
        {}

        // Special methods for the flat random access iterator.
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
};

#undef N

#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
