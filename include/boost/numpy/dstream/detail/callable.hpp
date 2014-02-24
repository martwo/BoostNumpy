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
#include <boost/numpy/dstream/detail/callable_call.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <
      unsigned InArity
    , class FTypes
    , class MappingDefinition
    , class WiringModel
    , class ThreadAbility
>
struct callable_in_arity;

#define BOOST_PP_ITERATION_PARAMS_1 \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/detail/callable.hpp>))
#include BOOST_PP_ITERATE()

template <
      class F
    , class FSignature
    , class MappingDefinition
    , template <
           class _MappingDefinition
         , class _FTypes
      >
      class WiringModelTemplate
    , class ThreadAbility
>
struct callable_base_select
{
    typedef typename numpy::mpl::types_from_fctptr_signature<F, FSignature>::type
            f_types_t;

    typedef WiringModelTemplate<MappingDefinition, f_types_t>
            wiring_model_t;

    typedef typename callable_in_arity<
                  MappingDefinition::in::arity
                , f_types_t
                , MappingDefinition
                , wiring_model_t
                , ThreadAbility
            >::template impl<
                  f_types_t::is_mfp
                , MappingDefinition::maps_to_void
                , ThreadAbility::threads_allowed_t::value
                , F
            >
            type;
};

template <
      class F
    , class FSignature
    , class MappingDefinition
    , template <
            class _MappingDefinition
          , class _FTypes
      >
      class WiringModelTemplate
    , class ThreadAbility
>
struct callable
  : callable_base_select<F, FSignature, MappingDefinition, WiringModelTemplate, ThreadAbility>::type
{
    typedef typename callable_base_select<F, FSignature, MappingDefinition, WiringModelTemplate, ThreadAbility>::type
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

#define IN_ARITY BOOST_PP_ITERATION()

template <class FTypes, class MappingDefinition, class WiringModel, class ThreadAbility>
struct callable_in_arity<IN_ARITY, FTypes, MappingDefinition, WiringModel, ThreadAbility>
{
    typedef numpy::detail::callable_caller<
              IN_ARITY
            , typename FTypes::class_type
            , typename FTypes::return_type
            , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, typename FTypes::arg_type)
            >
            f_caller_t;

    typedef typename callable_call<
                  FTypes
                , f_caller_t
                , MappingDefinition
                , WiringModel
                , ThreadAbility
                >::type
            callable_call_t;

    template <
          bool is_member_function
        , bool has_void_return
        , bool allows_threads
        , class F
    >
    struct impl;

    //--------------------------------------------------------------------------
    // Partial specialization for member function with void-return and threads
    // allowed.
    template <class F>
    struct impl<true, true, true, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , typename FTypes::class_type &
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                , unsigned
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              typename FTypes::class_type & self
            , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
            , unsigned nthreads
        )
        {
            f_caller_t const f_caller(m_f);

            python::object out_obj;

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for member function with void-return and threads
    // forbidden.
    template <class F>
    struct impl<true, true, false, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , typename FTypes::class_type &
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              typename FTypes::class_type & self
            , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
        )
        {
            f_caller_t const f_caller(m_f);

            python::object out_obj;
            unsigned const nthreads = 1;

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for member function with non-void-return and
    // threads allowed.
    template <class F>
    struct impl<true, false, true, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , typename FTypes::class_type &
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                , python::object &
                , unsigned
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              typename FTypes::class_type & self
            , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
            , python::object & out_obj
            , unsigned nthreads
        )
        {
            f_caller_t const f_caller(m_f);

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for member function with non-void-return and
    // threads forbidden.
    template <class F>
    struct impl<true, false, false, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , typename FTypes::class_type &
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                , python::object &
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              typename FTypes::class_type & self
            , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
            , python::object & out_obj
        )
        {
            f_caller_t const f_caller(m_f);
            unsigned const nthreads = 1;

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for stand-alone function with non-void-return and
    // threads forbidden.
    template <class F>
    struct impl<false, false, false, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                , python::object &
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
            , python::object & out_obj
        )
        {
            f_caller_t const f_caller(m_f);
            typename FTypes::class_type self();
            unsigned const nthreads = 1;

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for stand-alone function with void-return and
    // threads forbidden.
    template <class F>
    struct impl<false, true, false, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
        )
        {
            f_caller_t const f_caller(m_f);
            typename FTypes::class_type self();
            python::object out_obj;
            unsigned const nthreads = 1;

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for stand-alone function with void-return and
    // threads allowed.
    template <class F>
    struct impl<false, true, true, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                , unsigned
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
            , unsigned nthreads
        )
        {
            f_caller_t const f_caller(m_f);
            typename FTypes::class_type self();
            python::object out_obj;

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for stand-alone function with non-void-return and
    // threads allowed.
    template <class F>
    struct impl<false, false, true, F>
    {
        typedef boost::mpl::vector<
                  python::object
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & BOOST_PP_INTERCEPT)
                , python::object &
                , unsigned
                >
                signature_t;

        F m_f;
        impl(F f) : m_f(f) {}

        python::object
        operator()(
              BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, python::object const & in_obj)
            , python::object & out_obj
            , unsigned nthreads
        )
        {
            f_caller_t const f_caller(m_f);
            typename FTypes::class_type self();

            return callable_call_t::call(
                  f_caller
                , self
                , BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_obj)
                , out_obj
                , nthreads);
        }
    };
};

#undef IN_ARITY

#endif // BOOST_PP_IS_ITERATING
