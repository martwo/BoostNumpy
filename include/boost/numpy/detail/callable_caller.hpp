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

#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/types.hpp>
#include <boost/numpy/detail/config.hpp>
#include <boost/numpy/detail/callable_ptr.hpp>

namespace boost {
namespace numpy {
namespace detail {

//==============================================================================
/**
 * \brief The master template is used to handle class member functions. The
 *     specialized template with Class = BOOST_NUMPY_PP_MPL_VOID is used for
 *     handling standalone functions.
 */
template <
    int InArity
  , class Class
  , typename OutT
  , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_, = BOOST_NUMPY_PP_MPL_VOID BOOST_PP_INTERCEPT)
>
struct callable_caller
{
    typedef callable_ptr<
        InArity
      , Class
      , OutT
      , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
      > callable_ptr_t;

    //__________________________________________________________________________
    callable_caller(setting_t const & setting)
      : flags(setting.flags)
    {
        if(bool(flags & IS_MEMBER_FUNCTION_POINTER))
        {
            bfunc = setting.template get_data<typename callable_ptr_t::member_ptr_t>();
        }
        else if(bool(flags & IS_POINTER))
        {
            bfunc = setting.template get_data<typename callable_ptr_t::function_ptr_t>();
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError,
                "Wrong configuration: The setting flags of the function "
                "pointer do not indicate whether it is a function pointer "
                "or a member function pointer!");
            python::throw_error_already_set();
        }
    }

    //__________________________________________________________________________
    int flags;
    typename callable_ptr_t::bfunc_t bfunc;
};

//______________________________________________________________________________
// Template specialization for Class = mpl::unspecified for calling a
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
      > callable_ptr_t;

    //__________________________________________________________________________
    callable_caller(setting_t const & setting)
      : flags(setting.flags)
    {
        if(bool(flags & IS_POINTER))
        {
            bfunc = setting.template get_data<typename callable_ptr_t::function_ptr_t>();
        }
        else
        {
            PyErr_SetString(PyExc_RuntimeError,
                "Wrong configuration: The setting flags of the function "
                "pointer do not indicate if it is a function pointer!");
            python::throw_error_already_set();
        }
    }

    //__________________________________________________________________________
    int flags;
    typename callable_ptr_t::bfunc_t bfunc;
};

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DETAIL_CALLABLE_CALLER_HPP_INCLUDED
