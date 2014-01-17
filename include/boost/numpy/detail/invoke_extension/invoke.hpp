/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/detail/invoke_extension/invoke.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines special versions of the
 *        boost::python::detail::invoke function template that will be used
 *        for invoking detailed boost::numpy functions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/punctuation/paren.hpp>

#include <boost/mpl/at.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/mpl/if.hpp>
#include <boost/pointer_cast.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/function_types/function_arity.hpp>
#include <boost/function_types/is_function_pointer.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/remove_pointer.hpp>

#include <boost/python/detail/preprocessor.hpp>

#include <boost/numpy/dstream/detail/callable_base.hpp>

namespace boost {
namespace python {
namespace detail {

/*
namespace numpy {

template <class Args, long Idx>
struct arg
{
    // The 0th argument is the class type if Args is the argument type vector
    // for a class member function.
    typedef typename boost::mpl::at_c<Args, Idx>::type type;
};


template <class T>
struct is_callable_type
{
    typedef typename boost::is_same<T, boost::numpy::detail::callable_nonvoid_rtype>::type type;
};

template <class F>
struct is_callable
{
    typedef boost::function_types::function_arity<F> arity_t;
    typedef typename boost::mpl::equal< arity_t, boost::mpl::int_<2> >::type has_correct_arity_t;
    typedef boost::function_types::parameter_types<F> args_t;
    typedef boost::function_types::is_function_pointer<F> is_fp_t;

    typedef typename boost::mpl::if_
            < is_fp_t
              , typename boost::mpl::if_
                < has_correct_arity_t
                  , typename is_callable_type< typename arg<args_t,0>::type >::type
                  , boost::mpl::false_
                >::type
              , boost::mpl::false_>::type type;
};

template <class F>
struct callable_ptr_type
{
    typedef typename boost::mpl::if_
            < typename is_callable<F>::type
              , F
              , boost::numpy::detail::unspecified_ptr
            >::type type;
};

}//namespace numpy
*/

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (0, BOOST_PYTHON_MAX_ARITY, </home/mwolf/i3projects/i3ndarray/public/boost/numpy/detail/invoke_extension/invoke.hpp>))
#include BOOST_PP_ITERATE()

}/*namespace detail*/
}/*namespace python*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_HPP_INCLUDED
#else // !defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

template <class RC, class F BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
inline PyObject* invoke(boost::numpy::dstream::detail::callable_mf_base<false>, RC const& rc, F& f BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac) )
{
    return rc( f->call(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, ac, () BOOST_PP_INTERCEPT)) );
}

#define DEF(z, n, data) \
    BOOST_PP_COMMA() BOOST_PP_CAT(ac, n) BOOST_PP_LPAREN() BOOST_PP_RPAREN()

template <class RC, class F, class TC BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
inline PyObject* invoke(boost::numpy::dstream::detail::callable_mf_base<true>, RC const& rc, F& f, TC& tc BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac) )
{
    //FIXME: Use BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z
    return rc( f->call(tc() BOOST_PP_REPEAT(N, DEF, ~) ) );
}

#undef DEF

#undef N

#endif // BOOST_PP_IS_ITERATING
