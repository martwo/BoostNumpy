/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/converter/return_type_to_out_mapping.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the return_type_to_out_mapping template for
 *        converting a function's return type to an output mapping type.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_HPP_INCLUDED

#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/mpl/is_std_vector_of_scalar.hpp>
#include <boost/numpy/dstream/dim.hpp>
#include <boost/numpy/dstream/mapping/detail/out.hpp>
#include <boost/numpy/dstream/mapping/detail/core_shape.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace converter {

namespace detail {

struct return_type_to_out_mapping_type
{};

}// namespace detail

template <class T, class Enable=void>
struct return_type_to_out_mapping
  : detail::return_type_to_out_mapping_type
{
    // The return_type_to_out_mapping needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_return_type_to_out_mapping_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_RETURN_TYPE_T, (T));
};

namespace detail {

template <class T, class Enable=void>
struct void_to_out_mapping
  : return_type_to_out_mapping_type
{
    typedef mapping::detail::out<0>::core_shapes<>
            type;
};

template <class T, class Enable=void>
struct scalar_to_out_mapping
  : return_type_to_out_mapping_type
{
    typedef mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<0>::shape<> >
            type;
};

template <class T, class Enable=void>
struct std_vector_of_scalar_to_out_mapping
  : return_type_to_out_mapping_type
{
    typedef mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<1>::shape< dim::I > >
            type;
};

template <class T>
struct select_return_type_to_out_mapping
{
    typedef typename boost::mpl::if_<
              typename is_same<T, void>::type
            , void_to_out_mapping<T>

            , typename boost::mpl::if_<
                typename is_scalar<typename remove_reference<T>::type>::type
              , scalar_to_out_mapping<T>

              , typename boost::mpl::if_<
                  typename numpy::mpl::is_std_vector_of_scalar<T>::type
                , std_vector_of_scalar_to_out_mapping<T>

                , typename numpy::dstream::mapping::converter::return_type_to_out_mapping<T>
                >::type
              >::type
            >::type
            type;
};

template <class T>
struct return_type_to_out_mapping
  : select_return_type_to_out_mapping<T>::type
{
    typedef typename select_return_type_to_out_mapping<T>::type
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

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_HPP_INCLUDED
