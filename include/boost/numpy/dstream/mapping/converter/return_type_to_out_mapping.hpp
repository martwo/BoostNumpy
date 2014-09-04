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
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>

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

template <class T, class Enable=void>
struct return_type_to_out_mapping
{
    // The return_type_to_out_mapping needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_return_type_to_out_mapping_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_RETURN_TYPE_T, (T));
};

namespace detail {

template <class T>
struct void_to_out_mapping
{
    typedef mapping::detail::out<0>::core_shapes<>
            type;
};

template <class T>
struct scalar_return_type_to_out_mapping
{
    typedef mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<0>::shape<> >
            type;
};

template <class T>
struct std_vector_of_scalar_return_type_to_out_mapping
{
    typedef mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<1>::shape< dim::I > >
            type;
};

template <class T, unsigned nd>
struct std_vector_return_type_to_out_mapping;

// Define ND specializations for dimensions J to Z, i.e. up to 18 dimensions.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, 18, <boost/numpy/dstream/mapping/converter/return_type_to_out_mapping.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class T>
struct select_return_type_to_out_mapping
{
    typedef typename remove_reference<T>::type
            bare_t;

    typedef typename boost::mpl::if_<
              typename is_same<bare_t, void>::type
            , void_to_out_mapping<T>

            , typename boost::mpl::if_<
                typename is_scalar<bare_t>::type
              , scalar_return_type_to_out_mapping<T>

              , typename boost::mpl::if_<
                  typename is_same<bare_t, python::object>::type
                , scalar_return_type_to_out_mapping<T>

                , typename boost::mpl::if_<
                    typename numpy::mpl::is_std_vector<T>::type
                  , std_vector_return_type_to_out_mapping<T, 1>

                  , numpy::mpl::unspecified
                  >::type
                >::type
              >::type
            >::type
            type;
};

template <class T>
struct return_type_to_out_mapping
{
    typedef typename select_return_type_to_out_mapping<T>::type
            builtin_converter_selector;

    typedef typename boost::mpl::eval_if<
              typename is_same<typename builtin_converter_selector::type, numpy::mpl::unspecified>::type
            , ::boost::numpy::dstream::mapping::converter::return_type_to_out_mapping<T>
            , builtin_converter_selector
            >::type
            type;
};

}// namespace detail
}// namespace converter
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define ND BOOST_PP_ITERATION()

template <class T>
struct std_vector_return_type_to_out_mapping<T, ND>
{
    typedef typename remove_reference<T>::type
            vector_t;
    typedef typename vector_t::value_type
            vector_value_t;
    typedef typename remove_reference<vector_value_t>::type
            vector_bare_value_t;

    #define BOOST_NUMPY_DSTREAM_DEF(z, n, data) \
        BOOST_PP_COMMA_IF(n) dim::I - n
    typedef typename boost::mpl::if_<
              typename is_scalar<vector_bare_value_t>::type
            , mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<ND>::shape< BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_DEF, ~) > >

            , typename boost::mpl::if_<
                typename is_same<vector_bare_value_t, python::object>::type
              , mapping::detail::out<1>::core_shapes< mapping::detail::core_shape<ND>::shape< BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_DEF, ~) > >

              , typename boost::mpl::eval_if<
                  typename numpy::mpl::is_std_vector<vector_bare_value_t>::type
                , std_vector_return_type_to_out_mapping<vector_bare_value_t, ND+1>

                , numpy::mpl::unspecified
                >::type
              >::type
            >::type
            type;
    #undef BOOST_NUMPY_DSTREAM_DEF
};

#undef ND

#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
