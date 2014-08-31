/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/arg_type_to_array_dtype.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the arg_type_to_array_dtype converter template that
 *        should translate a C++ function argument type to a C++ data type which
 *        should be used to construct the ndarray's dtype object (via the
 *        boost::numpy::dtype::get_builtin< converter::type >() function).
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_WIRING_ARG_TYPE_TO_ARRAY_DTYPE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_ARG_TYPE_TO_ARRAY_DTYPE_HPP_INCLUDED

#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/remove_reference.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace converter {

template <class T, class Enable=void>
struct arg_type_to_array_dtype
{
    typedef arg_type_to_array_dtype<T, Enable>
            type;

    // The arg_type_to_core_shape needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_arg_type_to_array_dtype_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_ARGUMENT_TYPE_T, (T));
};

namespace detail {

template <class T>
struct std_vector_arg_type_to_array_dtype
{
    typedef typename remove_reference<T>::type
            bare_t;
    typedef typename remove_reference<typename bare_t::value_type>::type
            vector_bare_value_type;

    typedef typename boost::mpl::eval_if<
              typename numpy::mpl::is_std_vector<vector_bare_value_type>::type
            , std_vector_arg_type_to_array_dtype<vector_bare_value_type>
            , boost::mpl::identity<vector_bare_value_type>
            >::type
            type;
};

template <class T>
struct select_arg_type_to_array_dtype
{
    typedef typename remove_reference<T>::type
            bare_t;

    typedef typename boost::mpl::if_<
              typename is_scalar<bare_t>::type
            , boost::mpl::identity<bare_t>

            , typename boost::mpl::if_<
                typename is_same<bare_t, python::object>::type
              , boost::mpl::identity<python::object>

              , typename boost::mpl::if_<
                  typename numpy::mpl::is_std_vector<bare_t>::type
                , std_vector_arg_type_to_array_dtype<T>

                , ::boost::numpy::dstream::wiring::converter::arg_type_to_array_dtype<T>
                >::type
              >::type
            >::type
            apply;
};

template <class T>
struct arg_type_to_array_dtype
{
    typedef typename select_arg_type_to_array_dtype<T>::apply::type
            type;
};

}// namespace detail
}// namespace converter
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_ARG_TYPE_TO_ARRAY_DTYPE_HPP_INCLUDED
