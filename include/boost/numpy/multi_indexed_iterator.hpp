/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/multi_indexed_iterator.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::multi_indexed_iterator template
 *        providing a C++ style iterator over a multiple ndarrays keeping track
 *        of indices. It provides the ``jump_to`` method to jump to a specified
 *        set of indices. Due to the multiple operands, the dereference method
 *        always just returns a boolean value, but the individual values can
 *        be accessed through the value_ptr0, value_ptr1, ... member variables
 *        being pointers to the individual data values.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_MULTI_INDEXED_ITERATOR_HPP_INCLUDED
#define BOOST_NUMPY_MULTI_INDEXED_ITERATOR_HPP_INCLUDED 1

#include <boost/python.hpp>

#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/numpy/detail/multi_iter_iterator.hpp>

namespace boost {
namespace numpy {
namespace iterators {

template <typename ValueType>
struct value_type_traits
{
    typedef ValueType *
            ptr_type;

    ptr_type
    reinterpret_ptr(char* data_ptr)
    {
        return reinterpret_cast<ptr_type>(data_ptr);
    }
};

template <>
struct value_type_traits<python::object>
{
    typedef python::object
            ptr_type;

    ptr_type
    reinterpret_ptr(char* data_ptr)
    {
        uintptr_t * ptr = reinterpret_cast<uintptr_t*>(data_ptr);
        python::object obj(python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*ptr)));
        return obj;
    }
};

template <int nd>
struct multi_indexed_iterator
{};

#define ND 3

template <>
struct multi_indexed_iterator<ND>
{
    template <BOOST_PP_ENUM_PARAMS(ND, typename ValueType)>
    class iter
      : public detail::multi_iter_iterator<ND>::iter< multi_indexed_iterator<ND>::iter<BOOST_PP_ENUM_PARAMS(ND, ValueType)>, boost::forward_traversal_tag>
    {
      public:
        typedef multi_indexed_iterator<ND>::iter<BOOST_PP_ENUM_PARAMS(ND, ValueType)>
                type_t;
        typedef detail::multi_iter_iterator<ND>::iter< multi_indexed_iterator<ND>::iter<BOOST_PP_ENUM_PARAMS(ND, ValueType)>, boost::forward_traversal_tag>
                base_t;

        static
        boost::shared_ptr<detail::iter>
        construct_iter(detail::multi_iter_iterator_type & iter_base, BOOST_PP_ENUM_PARAMS(ND, ndarray & arr))
        {
            type_t & cppiter = *static_cast<type_t *>(&iter_base);
            int const nd = arr0.get_nd();

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

            #define BOOST_NUMPY_DEF(z, n, data) \
                detail::iter_operand_flags_t BOOST_PP_CAT(arr_op_flags,n) = detail::iter_operand::flags::NONE::value;\
                BOOST_PP_CAT(arr_op_flags,n) |= cppiter.BOOST_PP_CAT(arr_access_flags_,n) & detail::iter_operand::flags::READONLY::value;\
                BOOST_PP_CAT(arr_op_flags,n) |= cppiter.BOOST_PP_CAT(arr_access_flags_,n) & detail::iter_operand::flags::WRITEONLY::value;\
                BOOST_PP_CAT(arr_op_flags,n) |= cppiter.BOOST_PP_CAT(arr_access_flags_,n) & detail::iter_operand::flags::READWRITE::value;
            BOOST_PP_REPEAT(ND, BOOST_NUMPY_DEF, ~)
            #undef BOOST_NUMPY_DEF
            #define BOOST_NUMPY_DEF(z, n, data) \
                detail::iter_operand BOOST_PP_CAT(arr_op,n)(BOOST_PP_CAT(arr,n), BOOST_PP_CAT(arr_op_flags,n), arr_op_bcr);
            BOOST_PP_REPEAT(ND, BOOST_NUMPY_DEF, ~)
            #undef BOOST_NUMPY_DEF

            boost::shared_ptr<detail::iter> it(new detail::iter(
                iter_flags
              , KEEPORDER
              , NO_CASTING
              , nd           // n_iter_axes
              , itershape
              , 0            // buffersize
              , BOOST_PP_ENUM_PARAMS(ND, arr_op)
            ));
            it->init_full_iteration();
            return it;
        }

        // The existence of the default constructor is needed by the STL
        // requirements.
        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(value_ptr_,n)(NULL)
        iter()
          : base_t()
          , BOOST_PP_ENUM(ND, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
        {}

        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_COMMA_IF(n) detail::iter_operand_flags_t BOOST_PP_CAT(arr_access_flags,n) = detail::iter_operand::flags::READONLY::value
        explicit iter(
            BOOST_PP_ENUM_PARAMS(ND, ndarray & arr)
          , BOOST_PP_REPEAT(ND, BOOST_NUMPY_DEF, ~)
        )
        #undef BOOST_NUMPY_DEF
        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(value_ptr_,n)(NULL)
          : base_t(BOOST_PP_ENUM_PARAMS(ND, arr), BOOST_PP_ENUM_PARAMS(ND, arr_access_flags), &type_t::construct_iter)
          , BOOST_PP_ENUM(ND, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
        {}

        // Copy constructor.
        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(value_ptr_,n)(other.BOOST_PP_CAT(value_ptr_,n))
        iter(type_t const & other)
          : base_t(other)
          , BOOST_PP_ENUM(ND, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
        {}

        bool
        dereference() const
        {
            #define BOOST_NUMPY_DEF(z, n, data) \
                BOOST_PP_CAT(value_ptr_,n) = value_type_traits<BOOST_PP_CAT(ValueType,n)>::reinterpret_ptr( base_t::iter_ptr_->data_ptr_array_ptr_[ n ] );
            BOOST_PP_REPEAT(ND, BOOST_NUMPY_DEF, ~)
            #undef BOOST_NUMPY_DEF
            return true;
        }

        void
        jump_to(std::vector<intptr_t> const & indices)
        {
            base_t::iter_ptr_->jump_to(indices);
        }

        // Define the value_ptr_# pointers to the array values.
        #define BOOST_NUMPY_DEF(z, n, data) \
            typename value_type_traits<BOOST_PP_CAT(ValueType,n)>::ptr_type BOOST_PP_CAT(value_ptr_,n);
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
    };
};

#undef ND

}// namespace iterators
}// namespace numpy
}// namespace boost

#endif // BOOST_NUMPY_MULTI_INDEXED_ITERATOR_HPP_INCLUDED
