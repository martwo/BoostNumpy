/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/return_type_to_array_dtype.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the return_type_to_array_dtype converter template
 *        that should translate a C++ function return type to a C++ data type
 *        which should be used to construct the ndarray's dtype object (via the
 *        boost::numpy::dtype::get_builtin< converter::type >() function) for
 *        the idx'th output array under consideration of the out mapping type.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_WIRING_RETURN_TYPE_TO_ARRAY_DTYPE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_RETURN_TYPE_TO_ARRAY_DTYPE_HPP_INCLUDED

#include <boost/mpl/assert.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numpy/mpl/is_std_vector.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace converter {

template <class OutMapping, class RT, unsigned idx, class Enable=void>
struct return_type_to_array_dtype
{
    typedef return_type_to_array_dtype<RT, OutMapping, idx, Enable>
            type;

    // The return_type_to_core_shape needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_return_type_to_array_dtype_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_RETURN_TYPE_RT_AND_OUTMAPPING_TYPE_OutMapping_AND_OUTPUT_ARRAY_INDEX_idx, (RT, OutMapping));
};

namespace detail {

template <class OutMapping, class RT, unsigned idx>
struct std_vector_return_type_to_array_dtype
{
    typedef typename remove_reference<RT>::type
            vector_t;
    typedef typename vector_t::value_type
            vector_value_t;
    typedef typename remove_reference<vector_value_t>::type
            vector_bare_value_t;

    typedef typename boost::mpl::if_<
              typename is_scalar<vector_bare_value_t>::type
            , vector_bare_value_t

            , typename boost::mpl::if_<
                typename is_same<vector_bare_value_t, python::object>::type
              , python::object

              , typename boost::mpl::eval_if<
                  typename numpy::mpl::is_std_vector<vector_bare_value_t>::type
                , std_vector_return_type_to_array_dtype<OutMapping, vector_bare_value_t, idx>

                , numpy::mpl::unspecified
                >::type
              >::type
            >::type
            type;
};

template <class OutMapping, class RT, unsigned idx>
struct select_return_type_to_array_dtype
{
    typedef typename remove_reference<RT>::type
            bare_rt;

    typedef typename boost::mpl::if_<
              typename is_scalar<bare_rt>::type
            , boost::mpl::identity<bare_rt>

            , typename boost::mpl::if_<
                typename is_same<bare_rt, python::object>::type
              , boost::mpl::identity<python::object>

              , typename boost::mpl::if_<
                  typename numpy::mpl::is_std_vector<bare_rt>::type
                , std_vector_return_type_to_array_dtype<OutMapping, RT, idx>

                , numpy::mpl::unspecified
                >::type
              >::type
            >::type
            type;
};

template <class OutMapping, class RT, unsigned idx>
struct return_type_to_array_dtype
{
    typedef typename select_return_type_to_array_dtype<OutMapping, RT, idx>::type
            builtin_converter_selector;
    typedef typename boost::mpl::eval_if<
              is_same<typename builtin_converter_selector::type, numpy::mpl::unspecified>
            , ::boost::numpy::dstream::wiring::converter::return_type_to_array_dtype<OutMapping, RT, idx>
            , builtin_converter_selector
            >::type
            type;
};

}// namespace detail
}// namespace converter
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_RETURN_TYPE_TO_ARRAY_DTYPE_HPP_INCLUDED
