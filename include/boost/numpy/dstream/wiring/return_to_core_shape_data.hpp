/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/return_to_core_shape_data.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the return_to_core_shape_data template that should
 *        put a function's return value into the output arrays defined by the
 *        out mapping type and its core shapes.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_WIRING_RETURN_TO_CORE_SHAPE_DATA_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_RETURN_TO_CORE_SHAPE_DATA_HPP_INCLUDED

#include <stdint.h>

#include <vector>

#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>

#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/dstream/mapping/detail/definition.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace converter {

template <class OutMapping, class RT, class Enable=void>
struct return_to_core_shape_data
{
    typedef return_to_core_shape_data<OutMapping, RT, Enable>
            type;

    // The return_to_core_shape_data needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_return_to_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_RETURN_TYPE_RT_AND_OUT_MAPPING_TYPE_OutMapping, (RT, OutMapping));
};

namespace detail {

template <class OutMapping, class RT>
struct scalar_return_to_core_shape_data
{
    typedef scalar_return_to_core_shape_data<OutMapping, RT>
            type;

    static
    void
    apply(
        RT result
      , numpy::detail::iter & iter
      , std::vector< std::vector<intptr_t> > const & out_core_shapes
    )
    {

    }
};

template <class OutMapping, class RT>
struct std_vector_return_to_core_shape_data
{
    // FIXME: Add some useful implementation.
    typedef typename ::boost::numpy::dstream::wiring::converter::return_to_core_shape_data<OutMapping, RT>::type
            type;
};

template <class OutMapping, class RT>
struct select_return_to_core_shape_data_converter
{
    typedef typename remove_reference<RT>::type
            bare_rt;

    typedef mapping::detail::out_mapping<OutMapping>
            out_mapping_utils;

    typedef typename boost::mpl::if_<
              typename boost::mpl::and_<
                         typename is_scalar<bare_rt>::type
                       , typename out_mapping_utils::template arity_is_equal_to<1>::type
                       , typename out_mapping_utils::template array<0>::is_scalar::type
                       >::type
            , scalar_return_to_core_shape_data<OutMapping, RT>

            , typename boost::mpl::if_<
                typename numpy::mpl::is_std_vector<bare_rt>::type
              , std_vector_return_to_core_shape_data<OutMapping, RT>

              , ::boost::numpy::dstream::wiring::converter::return_to_core_shape_data<OutMapping, RT>
              >::type
            >::type
            apply;
};

template <class OutMapping, class RT>
struct return_to_core_shape_data_converter
{
    typedef typename select_return_to_core_shape_data_converter<OutMapping, RT>::apply::type
            type;
};

}// namespace detail
}// namespace converter
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_WIRING_RETURN_TO_CORE_SHAPE_DATA_HPP_INCLUDED
