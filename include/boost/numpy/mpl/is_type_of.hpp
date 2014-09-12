/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/mpl/is_type_of.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines a MPL template for checking if a type T is the same
 *        as type U using an unary metafunction for checking.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_MPL_IS_TYPE_OF_HPP_INCLUDED
#define BOOST_NUMPY_MPL_IS_TYPE_OF_HPP_INCLUDED

#include <boost/type_traits/is_same.hpp>

namespace boost {
namespace numpy {
namespace mpl {

template <class U>
struct is_type_of
{
    template <class T>
    struct apply
    {
        typedef typename is_same<T, U>::type
                type;
    };
};

}// namespace mpl
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_MPL_IS_TYPE_OF_HPP_INCLUDED
