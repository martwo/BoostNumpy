/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/detail/callable_ptr.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines a template for describing a function pointer or a
 *        class member function pointer.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DETAIL_CALLABLE_PTR_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_CALLABLE_PTR_HPP_INCLUDED

#include <boost/function.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/unspecified.hpp>

namespace boost {
namespace numpy {
namespace detail {

/**
 * \brief The callable_ptr class template that does not specialize the ClassT
 *     template parameter, handles a class member function pointer. Whereas the
 *     callable_ptr template that specializes the ClassT template parameter with
 *     numpy::mpl::unspecified, handles a function pointer.
 *     The master template is left blank, because the callable_ptr template
 *     needs an explicit specialization for each input arity.
 */
template <
    int InArity
  , class ClassT
  , typename OutT
  , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
struct callable_ptr;

//______________________________________________________________________________
// Partial specialization for in_arity = N.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/detail/callable_ptr.hpp>))
#include BOOST_PP_ITERATE()

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DETAIL_CALLABLE_PTR_HPP_INCLUDED
// EOF
//==============================================================================
#else

#define IN_ARITY BOOST_PP_ITERATION()

//______________________________________________________________________________
// Specialization for input arity IN_ARITY of a class member function pointer.
template <
    class ClassT
  , typename OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_)
>
struct callable_ptr<
    IN_ARITY
  , ClassT
  , OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
>
{
    typedef boost::function<OutT (ClassT* self, BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_))>
            bfunc_t;

    bfunc_t bfunc_;
    callable_ptr(bfunc_t bfunc)
      : bfunc_(bfunc)
    {}

    OutT
    operator()(ClassT* self, BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_)) const
    {
        return bfunc_(self, BOOST_PP_ENUM_PARAMS(IN_ARITY, in_));
    }
};

//______________________________________________________________________________
// Specialization for input arity IN_ARITY of a function pointer.
template <
    typename OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_)
>
struct callable_ptr<
    IN_ARITY
  , mpl::unspecified
  , OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
>
{
    typedef boost::function<OutT (BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_))>
            bfunc_t;

    bfunc_t bfunc_;
    callable_ptr(bfunc_t bfunc)
      : bfunc_(bfunc)
    {}

    OutT
    operator()(numpy::mpl::unspecified*, BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_)) const
    {
        return bfunc_(BOOST_PP_ENUM_PARAMS(IN_ARITY, in_));
    }
};

#undef IN_ARITY

#endif // !BOOST_PP_IS_ITERATING
