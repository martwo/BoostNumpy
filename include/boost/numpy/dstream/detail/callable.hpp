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

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/mpl/if.hpp>

#include <boost/python/object.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/types_from_fctptr_signature.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <unsigned OutArity, unsigned InArity>
struct callable_outin_arity;

// Do a 2D file iteration with output and input arity as loop variables.
#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, BOOST_NUMPY_LIMIT_OUTPUT_ARITY, <boost/numpy/dstream/detail/callable.hpp>))
#include BOOST_PP_ITERATE()

template <
      class F
    , class FSignature
    , class MappingDefinition
    , template <
         class _MappingDefinition
      >
      class WiringModel
    , class ThreadAbility
>
struct callable_base_select
{
    typedef typename numpy::mpl::types_from_fctptr_signature<F, FSignature>::type
            f_types_t;

    typedef WiringModel<MappingDefinition>
            wiring_model_t;

    typedef typename callable_outin_arity<MappingDefinition::out::arity, MappingDefinition::in::arity>::template impl<
                  f_types_t::is_mfp
                , MappingDefinition::maps_to_void
                , ThreadAbility::threads_allowed_t::value
                , F
                , f_types_t
                , MappingDefinition
                , wiring_model_t
                , ThreadAbility
            >
            type;
};

template <
      class F
    , class FSignature
    , class MappingDefinition
    , template <
          class _MappingDefinition
      >
      class WiringModel
    , class ThreadAbility
>
struct callable
  : callable_base_select<F, FSignature, MappingDefinition, WiringModel, ThreadAbility>::type
{
    typedef typename callable_base_select<F, FSignature, MappingDefinition, WiringModel, ThreadAbility>::type
            base_t;

    callable(F f)
      : base_t(f)
    {}
};


}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_DEPTH() == 1

// Loop over the InArity.
#define BOOST_PP_ITERATION_PARAMS_2 (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/detail/callable.hpp>))
#include BOOST_PP_ITERATE()

#elif BOOST_PP_ITERATION_DEPTH() == 2

#define O BOOST_PP_RELATIVE_ITERATION(1)
#define I BOOST_PP_ITERATION()

template <>
struct callable_outin_arity<O,I>
{
    template <
          bool is_member_function
        , bool has_void_return
        , bool allows_threads
        , class F
        , class FTypes
        , class MappingDefinition
        , class WiringModel
        , class ThreadAbility
    >
    struct impl;

    //--------------------------------------------------------------------------
    // Partial specialization for member function with void-return and threads
    // allowed.
    template <class MappingDefinition, class F, class FTypes, class WiringModel, class ThreadAbility>
    struct impl<true, true, true, F, FTypes, MappingDefinition, WiringModel, ThreadAbility>
    {
        typedef boost::mpl::vector<
                  python::object
                , typename FTypes::class_t &
                , BOOST_PP_ENUM_PARAMS_Z(1, I, python::object const & BOOST_PP_INTERCEPT)
                , python::object &
                , unsigned
                >
                signature_t;

        typedef boost::numpy::detail::callable_caller<
                  I
                , typename FTypes::class_t
                , void
                , BOOST_PP_ENUM_PARAMS_Z(1, I, FTypes::in_t)
                >
                f_caller_t;

        impl(F f)
          : m_f(f)
        {}

        python::object
        operator()(
              typename FTypes::class_t & self
            , BOOST_PP_ENUM_PARAMS_Z(1, I, python::object const & a)
            , python::object & out
            , unsigned nthreads
        )
        {
            std::cout << I << std::endl;

            // Invoke WiringModel::iterate<F, FTypes, FCaller>(m_f, iter, core_shapes)
            return python::object();
        }

        F m_f;
    };
};

#undef I
#undef O

#endif // BOOST_PP_ITERATION_DEPTH

#endif // BOOST_PP_IS_ITERATING
