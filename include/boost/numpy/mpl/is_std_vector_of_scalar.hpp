/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/mpl/is_std_vector_of_scalar.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines a MPL template for checking if a type T is a
 *        std::vector of a scalar type.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_MPL_IS_STD_VECTOR_OF_SCALAR_HPP_INCLUDED
#define BOOST_NUMPY_MPL_IS_STD_VECTOR_OF_SCALAR_HPP_INCLUDED

#include <vector>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/mpl/has_allocator_type.hpp>
#include <boost/numpy/mpl/has_value_type.hpp>

namespace boost {
namespace numpy {
namespace mpl {

namespace detail {

template <class T, bool has_allocator_and_value_type>
struct is_std_vector_of_scalar_impl
{};

template <class T>
struct is_std_vector_of_scalar_impl<T, false>
{
    typedef boost::mpl::false_
            type;
};

template <class T>
struct is_std_vector_of_scalar_impl<T, true>
{
    typedef typename boost::mpl::if_<
                  typename is_scalar<typename remove_reference<typename T::value_type>::type>::type
                , typename boost::mpl::if_<
                        typename is_same< T, std::vector<typename T::value_type, typename T::allocator_type> >::type
                      , boost::mpl::true_
                      , boost::mpl::false_
                  >::type
                , boost::mpl::false_
            >::type
            type;
};

}// namespace detail

template <class T>
struct is_std_vector_of_scalar
{
    typedef typename detail::is_std_vector_of_scalar_impl
            < T
            , boost::mpl::and_< numpy::mpl::has_allocator_type<T>, numpy::mpl::has_value_type<T> >::type::value
            >::type
            type;
};

}// namespace mpl
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_MPL_IS_STD_VECTOR_OF_SCALAR_HPP_INCLUDED
