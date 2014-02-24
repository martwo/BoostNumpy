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

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/mpl/at.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/long.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>

#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_scalar.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/unspecified.hpp>

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

template <unsigned in_arity, class FTypes>
struct all_fct_args_are_scalars_incl_bool_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/mpl/types_from_fctptr_signature.hpp>))
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
struct all_fct_args_are_scalars_incl_bool
{
    typedef typename detail::all_fct_args_are_scalars_incl_bool_arity<FTypes::arity, FTypes>::type
            type;
};

template <class FTypes>
struct fct_return_is_scalar_or_bool
{
    typedef typename boost::mpl::or_<
                typename boost::is_scalar<typename FTypes::return_type>::type
              , typename boost::is_same<typename FTypes::return_type, bool>::type
            >::type
            type;
};

}// namespace mpl
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_MPL_OUTIN_TYPES_FROM_FCTPTR_SIGNATURE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

template <
      class Class
    , class Signature
    , int sig_arg_offset
>
struct types_from_fctptr_signature_impl<
      N
    , Class
    , Signature
    , sig_arg_offset
>
{
    BOOST_STATIC_CONSTANT(unsigned, arity = N);

    typedef typename boost::mpl::begin<Signature>::type::type
            return_type;

    typedef typename is_same<return_type, void>::type
            has_void_return_t;
    BOOST_STATIC_CONSTANT(bool, has_void_return = has_void_return_t::value);

    typedef Class class_type;

    typedef typename boost::mpl::not_< is_same<class_type, numpy::mpl::unspecified> >::type
            is_mfp_t;
    BOOST_STATIC_CONSTANT(bool, is_mfp = is_mfp_t::value);

    // Define arg_type# sub-types.
    #define BOOST_PP_LOCAL_MACRO(n) \
        typedef typename boost::mpl::at<Signature, boost::mpl::long_<sig_arg_offset + n> >::type BOOST_PP_CAT(arg_type,n);
    #define BOOST_PP_LOCAL_LIMITS (0, N-1)
    #include BOOST_PP_LOCAL_ITERATE()

    typedef boost::mpl::vector< BOOST_PP_ENUM_PARAMS_Z(1, N, arg_type) >
            in_type_vector;
};

template <class FTypes>
struct all_fct_args_are_scalars_incl_bool_arity<N, FTypes>
{
    typedef typename boost::mpl::and_<
                #define BOOST_PP_LOCAL_MACRO(n) \
                    BOOST_PP_COMMA_IF(n) typename boost::mpl::or_< \
                          typename boost::is_scalar<typename FTypes:: BOOST_PP_CAT(arg_type,n) >::type \
                        , typename boost::is_same<typename FTypes:: BOOST_PP_CAT(arg_type,n), bool>::type \
                        >::type
                #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(N,1))
                #include BOOST_PP_LOCAL_ITERATE()
            >::type
            type;
};

#undef N

#endif // BOOST_PP_IS_ITERATING
