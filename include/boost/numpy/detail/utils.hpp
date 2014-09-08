/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/detail/utils.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines some internal utility functions for boost::numpy.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_UTILS_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_UTILS_HPP_INCLUDED

#include <string>
#include <sstream>
#include <vector>

namespace boost {
namespace numpy {
namespace detail {

template <typename T>
std::string
pprint_shape(std::vector<T> const & v) // This function name is DEPRECATED.
{
    typename std::vector<T>::const_iterator it;
    typename std::vector<T>::const_iterator const v_begin = v.begin();
    typename std::vector<T>::const_iterator const v_end = v.end();

    std::stringstream os;
    os << "(";
    for(it=v_begin; it!=v_end; ++it)
    {
        if(it != v_begin) os << ", ";
        os << *it;
    }
    os << ")";

    return os.str();
}

template <typename T>
std::string
shape_vector_to_string(std::vector<T> const & v)
{
    return pprint_shape<T>(v);
}

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DETAIL_UTILS_HPP_INCLUDED
