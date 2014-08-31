/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/mpl/types_from_fctptr_signature.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the types_from_fctptr_signature meta-function to
 *        get the output type and the input types of a C++ (member) function
 *        pointer type from its signature MPL vector.
 *        The template defines sub-types for the output type and the input
 *        types. The output sub-type is named return_type and the input sub-types are
 *        named arg_type#, where # is a number from 0 to arity-1, where arity is
 *        static unsigned integer constant specifying the input arity of the
 *        function (without the hidden "this" argument in case of a member
 *        function).
 *        If the function pointer type is a member function pointer type, the
 *        class_t sub-type will be set accordingly, otherwise it is set to
 *        boost::numpy::mpl::unspecified.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_MPL_OUTIN_TYPES_FROM_FCTPTR_SIGNATURE_HPP_INCLUDED
#define BOOST_NUMPY_MPL_OUTIN_TYPES_FROM_FCTPTR_SIGNATURE_HPP_INCLUDED

#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/mpl/at.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/long.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>

#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_scalar.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/unspecified.hpp>
#include <boost/numpy/mpl/is_std_vector_of_scalar.hpp>

namespace boost {
namespace numpy {
namespace mpl {

namespace detail {

template <
      unsigned in_arity
    , class ClassT
    , class Signature
    , int sig_arg_offset
>
struct types_from_fctptr_signature_impl;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/mpl/types_from_fctptr_signature.hpp>, 1))
#include BOOST_PP_ITERATE()

template <unsigned in_arity, class FTypes>
struct all_fct_args_are_scalars_arity;

// Specialization for arity = 1 because boost::mpl::and_ needs at least 2
// arguments.
template <class FTypes>
struct all_fct_args_are_scalars_arity<1, FTypes>
{
    typedef boost::is_scalar<typename FTypes::arg_type0>
            type;
};

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (2, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/mpl/types_from_fctptr_signature.hpp>, 2))
#include BOOST_PP_ITERATE()

template <class F, class Signature>
struct types_from_fctptr_signature_impl_select
{
    typedef typename boost::mpl::if_<
                  is_member_function_pointer<F>
                , typename boost::mpl::at<Signature, boost::mpl::long_<1> >::type
                , unspecified
            >::type
            class_t;

    typedef typename boost::mpl::if_<
                  is_same<class_t, unspecified>
                , boost::mpl::long_<1>
                , boost::mpl::long_<2>
            >::type
            sig_arg_offset;

    typedef types_from_fctptr_signature_impl<
                  boost::mpl::size<Signature>::value - sig_arg_offset::value
                , class_t
                , Signature
                , sig_arg_offset::value
            >
            type;
};

}// namespace detail

template <class F, class Signature>
struct types_from_fctptr_signature
  : detail::types_from_fctptr_signature_impl_select<F, Signature>::type
{
    typedef typename detail::types_from_fctptr_signature_impl_select<F, Signature>::type
            type;
};

template <class FTypes>
struct all_fct_args_are_scalars
{
    typedef typename detail::all_fct_args_are_scalars_arity<FTypes::arity, FTypes>::type
            type;
};

template <class FTypes>
struct fct_return_is_scalar
{
    typedef boost::is_scalar<typename FTypes::return_type>
            type;
};

template <class FTypes>
struct fct_return_is_std_vector_of_scalar
{
    typedef typename numpy::mpl::is_std_vector_of_scalar<typename FTypes::return_type>::type
            type;
};

template <class FTypes, unsigned Idx>
struct fct_arg_type
{
    typedef typename boost::mpl::at< typename FTypes::in_type_vector, boost::mpl::long_<Idx> >::type
            type;
};

}// namespace mpl
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_MPL_OUTIN_TYPES_FROM_FCTPTR_SIGNATURE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

template <
      class ClassT
    , class Signature
    , int sig_arg_offset
>
struct types_from_fctptr_signature_impl<
      N
    , ClassT
    , Signature
    , sig_arg_offset
>
{
    BOOST_STATIC_CONSTANT(unsigned, arity = N);

    typedef Signature
            signature_t;

    typedef typename boost::mpl::begin<Signature>::type::type
            return_type;

    typedef typename is_same<return_type, void>::type
            has_void_return_t;
    BOOST_STATIC_CONSTANT(bool, has_void_return = has_void_return_t::value);

    typedef ClassT class_type;

    typedef typename boost::mpl::not_< is_same<class_type, numpy::mpl::unspecified> >::type
            is_mfp_t;
    BOOST_STATIC_CONSTANT(bool, is_mfp = is_mfp_t::value);

    // Define arg_type# sub-types.
    #define BOOST_NUMPY_DEF(z, n, data) \
        typedef typename boost::mpl::at<Signature, boost::mpl::long_<sig_arg_offset + n> >::type BOOST_PP_CAT(arg_type,n);
    BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
    #undef BOOST_NUMPY_DEF

    typedef boost::mpl::vector< BOOST_PP_ENUM_PARAMS_Z(1, N, arg_type) >
            in_type_vector;
};

#else
#if BOOST_PP_ITERATION_FLAGS() == 2

template <class FTypes>
struct all_fct_args_are_scalars_arity<N, FTypes>
{
    // By default, boost::mpl::and_ has only a maximal arity of 5, so we have
    // to construct a sequence of boost::mpl::and_<.,.> with always, two
    // arguments.
    #define BOOST_NUMPY_DEF_is_scalar(n) \
        boost::is_scalar<typename FTypes:: BOOST_PP_CAT(arg_type,n) >
    #define BOOST_NUMPY_DEF_pre_and(z, n, data) \
        typename boost::mpl::and_<
    #define BOOST_NUMPY_DEF_post_and(z, n, data) \
        BOOST_PP_COMMA() BOOST_NUMPY_DEF_is_scalar(BOOST_PP_ADD(n,1)) >::type

    typedef BOOST_PP_REPEAT(BOOST_PP_SUB(N,1), BOOST_NUMPY_DEF_pre_and, ~)
            BOOST_NUMPY_DEF_is_scalar(0)
            BOOST_PP_REPEAT(BOOST_PP_SUB(N,1), BOOST_NUMPY_DEF_post_and, ~)
            type;

    #undef BOOST_NUMPY_DEF_post_and
    #undef BOOST_NUMPY_DEF_pre_and
    #undef BOOST_NUMPY_DEF_is_scalar
};

#endif // BOOST_PP_ITERATION_FLAGS == 2
#endif // BOOST_PP_ITERATION_FLAGS == 1

#undef N

#endif // BOOST_PP_IS_ITERATING
