/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/detail/callable_caller.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines the callable_caller template to implement
 *        functionalities to call a callable (i.e. function or class member
 *        function).
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_CALLABLE_CALLER_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_CALLABLE_CALLER_HPP_INCLUDED

#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/unspecified.hpp>
#include <boost/numpy/detail/callable_ptr.hpp>

namespace boost {
namespace numpy {
namespace detail {

//==============================================================================
/**
 * \brief The master template is used to handle class member functions. The
 *     specialized template with ClassT = numpy::mpl::unspecified is used for
 *     handling standalone functions.
 */
template <
    int InArity
  , class ClassT
  , typename OutT
  , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
struct callable_caller
{
    typedef callable_ptr<
        InArity
      , ClassT
      , OutT
      , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
      > callable_ptr_call_t;

    template <class F>
    callable_caller(F f)
      : call(callable_ptr_call_t(typename callable_ptr_call_t::bfunc_t(f)))
    {}

    callable_ptr_call_t const call;
};

//______________________________________________________________________________
// Template specialization for ClassT = numpy::mpl::unspecified for calling a
// standalone function.
template <
    int InArity
  , typename OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_)
>
struct callable_caller<
    InArity
  , numpy::mpl::unspecified
  , OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
>
{
    typedef callable_ptr<
        InArity
      , numpy::mpl::unspecified
      , OutT
      , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
      > callable_ptr_call_t;

    template <class F>
    callable_caller(F f)
      : call(callable_ptr_call_t(typename callable_ptr_call_t::bfunc_t(f)))
    {}

    callable_ptr_call_t const call;
};

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DETAIL_CALLABLE_CALLER_HPP_INCLUDED
