/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/array_definition.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the
 *        boost::numpy::dstream::array_definition template that
 *        provides a definition of an array. An array definition consists of
 *        an array core shape and a data type for its element type.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_ARRAY_DEFINITION_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_ARRAY_DEFINITION_HPP_INCLUDED

namespace boost {
namespace numpy {
namespace dstream {

template <
      class CoreShape
    , class T
>
struct array_definition
{
    typedef CoreShape
            core_shape_type;

    typedef T
            value_type;
};

}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_ARRAY_DEFINITION_HPP_INCLUDED
