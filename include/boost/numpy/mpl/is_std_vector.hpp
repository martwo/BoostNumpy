/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/mpl/is_std_vector.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines a MPL template for checking if a type T is a
 *        std::vector type.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_MPL_IS_STD_VECTOR_HPP_INCLUDED
#define BOOST_NUMPY_MPL_IS_STD_VECTOR_HPP_INCLUDED

#include <vector>

#include <boost/mpl/bool.hpp>

namespace boost {
namespace numpy {
namespace mpl {

template <typename T>
struct is_std_vector
  : boost::mpl::false_
{};

// Specialize the is_std_vector template for any std::vector (with no
// additional template arguments).
template <typename V, typename A>
struct is_std_vector< std::vector<V, A> >
  : boost::mpl::true_
{};

}// namespace mpl
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_MPL_IS_STD_VECTOR_HPP_INCLUDED
