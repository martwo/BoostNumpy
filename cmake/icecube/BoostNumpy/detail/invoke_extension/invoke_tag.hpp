/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/detail/invoke_extension/invoke_tag.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines a special invoke_tag template for invoking detailed
 *        boost::numpy functions.
 *
 *        NOTE: This file just a copy of its original version located at the
 *              original include tree position as stated above. Paths to
 *              <boost/numpy/> have been changed to <BoostNumpy/>.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_TAG_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_TAG_HPP_INCLUDED

#include <boost/python/object_fwd.hpp>
#include <BoostNumpy/dstream/detail/callable_base.hpp>

namespace boost {
namespace python {
namespace detail {

template <class R, class F>
struct invoke_tag;

// Partially specialize the invoke_tag class template to tag boost::numpy
// functions, in order to be able to treat the pointer values of these function
// pointers as pointers to callable_t instances.
template <class F>
struct invoke_tag<
      boost::numpy::dstream::detail::callable_mf_base<false>
    , F
>
  : boost::numpy::dstream::detail::callable_mf_base<false>
{};
template <class F>
struct invoke_tag<
      boost::numpy::dstream::detail::callable_mf_base<true>
    , F
>
  : boost::numpy::dstream::detail::callable_mf_base<true>
{};


template <class Policies, class Result>
struct select_result_converter;

// Partially specialize the select_result_converter template in order to always
// select the correct result converter (for boost::python::object).
template <class Policies>
struct select_result_converter<
      Policies
    , boost::numpy::dstream::detail::callable_mf_base<false>
>
  : select_result_converter<Policies, boost::python::object>
{};
template <class Policies>
struct select_result_converter<
      Policies
    , boost::numpy::dstream::detail::callable_mf_base<true>
>
  : select_result_converter<Policies, boost::python::object>
{};

}/*namespace detail*/
}/*namespace python*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DETAIL_INVOKE_EXTENSION_INVOKE_TAG_HPP_INCLUDED
