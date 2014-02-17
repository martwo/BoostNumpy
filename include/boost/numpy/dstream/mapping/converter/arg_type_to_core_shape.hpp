/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/converter/arg_type_to_core_shape.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the arg_type_to_core_shape template for
 *        converting a function's argument type to a core shape type.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_ARG_TYPE_TO_CORE_SHAPE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_ARG_TYPE_TO_CORE_SHAPE_HPP_INCLUDED

#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>

#include <boost/numpy/mpl/is_std_vector_of_scalar.hpp>
#include <boost/numpy/dstream/dim.hpp>
#include <boost/numpy/dstream/mapping/detail/core_shape.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace converter {

namespace detail {

struct arg_type_to_core_shape_type
{};

}// namespace detail

template <class T, class Enable=void>
struct arg_type_to_core_shape
  : detail::arg_type_to_core_shape_type
{
    // The arg_type_to_core_shape needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_arg_type_to_core_shape_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_ARGUMENT_TYPE_T, (T));
};

namespace detail {

template <class T, class Enable=void>
struct scalar_to_core_shape
  : arg_type_to_core_shape_type
{
    typedef mapping::detail::core_shape<0>::shape<>
            type;
};

template <class T, class Enable=void>
struct std_vector_of_scalar_to_core_shape
  : arg_type_to_core_shape_type
{
    typedef mapping::detail::core_shape<1>::shape< dim::I >
            type;
};

template <class T>
struct select_arg_type_to_core_shape
{
    typedef typename boost::mpl::if_<
              typename is_scalar<typename remove_reference<T>::type>::type
            , scalar_to_core_shape<T>

            , typename boost::mpl::if_<
                typename numpy::mpl::is_std_vector_of_scalar<T>::type
              , std_vector_of_scalar_to_core_shape<T>

              , typename numpy::dstream::mapping::converter::arg_type_to_core_shape<T>
              >::type
            >::type

            type;
};

template <class T>
struct arg_type_to_core_shape
  : select_arg_type_to_core_shape<T>::type
{
    typedef typename select_arg_type_to_core_shape<T>::type
            base;
    typedef typename base::type
            type;
};

}// namespace detail

}// namespace converter
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_ARG_TYPE_TO_CORE_SHAPE_HPP_INCLUDED
