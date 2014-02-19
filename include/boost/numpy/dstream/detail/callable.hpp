/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/dstream/detail/callable.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the callable template that is the actual
 *        implementation of the iteration procedure.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_HPP_INCLUDED

#include <boost/mpl/if.hpp>

#include <boost/numpy/mpl/outin_types_from_fctptr_signature.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <unsigned InArity>
struct callable_inarity;

template <
      class F
    , class FSignature
    , class MappingDefinition
    , template <
            class _MappingDefinition
          , class _Class
          , class _OutInTypes
      >
      class WiringModel
    , class ThreadAbility
>
struct callable_base_select
{
    typedef typename numpy::mpl::outin_types_from_fctptr_signature<F, FSignature>::type
            outin_types_t;

    typedef typename outin_types_t::class_t
            class_t;

    typedef WiringModel<MappingDefinition, class_t, outin_types_t>
            wiring_model_t;

    // FIXME Construct callable_base type
    //typedef type;
};

template <
      class F
    , class FSignature
    , class MappingDefinition
    , template <
            class _MappingDefinition
          , class _Class
          , class _OutInTypes
      >
      class WiringModel
    , class ThreadAbility
>
struct callable
  : callable_base_select<F, MappingDefinition, WiringModel, ThreadAbility>::type
{
    typedef callable_base_select<F, FSignature, MappingDefinition, WiringModel, ThreadAbility>::type
            base_t;
};


}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_HPP_INCLUDED
#else

#endif // BOOST_PP_IS_ITERATING
