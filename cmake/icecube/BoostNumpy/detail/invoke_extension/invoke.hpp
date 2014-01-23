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
 *        NOTE: This file just a copy of its original version located at the
 *              original include tree position as stated above. Paths to
 *              <boost/numpy/> have been changed to <BoostNumpy/>.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>

#include <boost/python/detail/preprocessor.hpp>

#include <BoostNumpy/dstream/detail/callable_base.hpp>

namespace boost {
namespace python {
namespace detail {

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (0, BOOST_PYTHON_MAX_ARITY, <BoostNumpy/detail/invoke_extension/invoke.hpp>))
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

template <class RC, class F, class TC BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
inline PyObject* invoke(boost::numpy::dstream::detail::callable_mf_base<true>, RC const& rc, F& f, TC& tc BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac) )
{
    return rc( f->call(tc() BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, ac, () BOOST_PP_INTERCEPT)) );
}

#undef N

#endif // BOOST_PP_IS_ITERATING
