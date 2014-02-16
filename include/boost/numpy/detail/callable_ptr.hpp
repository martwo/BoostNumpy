/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/detail/callable_ptr.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
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

#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/unspecified.hpp>

namespace boost {
namespace numpy {
namespace detail {

/**
 * \brief The callable_ptr class template that does not specialize the Class
 *     template parameter, handles a class member function pointer. Whereas the
 *     callable_ptr template that specializes the Class template parameter with
 *     mpl::unspecified, handles a function pointer.
 *     The master template is left blank, because the callable_ptr template
 *     needs an explicit specialization for each input arity.
 */
template <
    int InArity
  , class Class
  , typename OutT
  , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_, = mpl::unspecified BOOST_PP_INTERCEPT)
>
struct callable_ptr;

//______________________________________________________________________________
// Partial specialization for in_arity = N.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/detail/callable_ptr.hpp>))
#include BOOST_PP_ITERATE()

}/*namespace detail*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DETAIL_CALLABLE_PTR_HPP_INCLUDED
// EOF
//==============================================================================
#else

#define IN_ARITY BOOST_PP_ITERATION()

//______________________________________________________________________________
// Specialization for input arity N of a class member function pointer.
template <
    class Class
  , typename OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_)
>
struct callable_ptr<
    IN_ARITY
  , Class
  , OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
>
{
    typedef boost::function<OutT (Class* self, BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_))> bfunc_t;
    typedef OutT (Class::*member_ptr_t)(BOOST_PP_ENUM_PARAMS(IN_ARITY, InT_));
    typedef OutT (*function_ptr_t)(Class*, BOOST_PP_ENUM_PARAMS(IN_ARITY, InT_));

    inline static
    OutT
    call(bfunc_t bfunc, Class* self, BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_))
    {
        return bfunc(self, BOOST_PP_ENUM_PARAMS(IN_ARITY, in_));
    }
};

//______________________________________________________________________________
// Specialization for input arity N of a function pointer.
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
    typedef boost::function<OutT (BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_))> bfunc_t;
    typedef OutT (*function_ptr_t)(BOOST_PP_ENUM_PARAMS(IN_ARITY, InT_));

    inline static
    OutT
    call(bfunc_t bfunc, mpl::unspecified*, BOOST_PP_ENUM_BINARY_PARAMS(IN_ARITY, InT_, in_))
    {
        return bfunc(BOOST_PP_ENUM_PARAMS(IN_ARITY, in_));
    }
};

#undef IN_ARITY

#endif // !BOOST_PP_IS_ITERATING
