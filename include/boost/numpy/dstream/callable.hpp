/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/callable.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines the data stream callable class template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_CALLABLE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_CALLABLE_HPP_INCLUDED

#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/function.hpp>
#include <boost/ref.hpp>
#include <boost/thread.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/numpy/detail/prefix.hpp>

#include <boost/python/make_function.hpp>

#include <boost/numpy/mpl/types.hpp>
#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/detail/pygil.hpp>
#include <boost/numpy/detail/max.hpp>

#include <boost/numpy/dstream/defaults.hpp>
#include <boost/numpy/dstream/dshape.hpp>

#include <boost/numpy/dstream/detail/callable_base.hpp>
#include <boost/numpy/dstream/detail/callable_call.hpp>

namespace boost {
namespace numpy {
namespace dstream {

namespace detail {
namespace error {

template <int nkeywords, int arity>
struct less_or_more_keywords_than_function_arguments
{
    typedef less_or_more_keywords_than_function_arguments<nkeywords, arity> type;
    typedef char too_few_or_many_keywords[nkeywords != arity ? -1 : 1];
};

}/*namespace error*/

template <
      unsigned InArity
    , class Class
    , class MappingModel
    , class WiringModel
    , class OutArrTransform
    , class ThreadAbility
>
struct callable_impl_base
  : callable_base
        < (boost::is_same<Class,numpy::mpl::unspecified>::value == false)
        , MappingModel::maps_to_void
        >
{
    typedef callable_impl_base
            <InArity, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
            callable_impl_base_t;

    BOOST_STATIC_CONSTANT(unsigned, in_arity = InArity);

    typedef Class
            class_t;

    typedef MappingModel
            mapping_model_t;

    typedef WiringModel
            wiring_model_t;

    typedef typename wiring_model_t::wiring_model_config_t
            wiring_model_config_t;

    typedef OutArrTransform
            out_arr_transform_t;

    typedef ThreadAbility
            thread_ability_t;

    wiring_model_t const wiring_model_;

    callable_impl_base(wiring_model_config_t const & wmc)
      : wiring_model_(wiring_model_t(wmc))
    {}
};

template <unsigned InArity>
struct callable_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/callable.hpp>))
#include BOOST_PP_ITERATE()

}/*namespace detail*/

//==============================================================================
/**
 * \brief The boost::numpy::dstream::callable template provides a helper
 *     class to create a function or class member function for a Python object,
 *     that supports several numpy arrays as input arguments and one numpy array
 *     as output. Data will be iterated over the first axis (i.e. the data
 *     stream) of all the arrays.
 */
template<
      class Class
    //, class IOTypes
    , class MappingModel
    , template <
            class _MappingModel
          , class _Class
          //, class _IOTypes
      >
      class WiringModel
    , template <
          class _MappingModel
      >
      class OutArrTransform
    , class ThreadAbility
>
struct callable
  : detail::callable_arity<MappingModel::in_arity>::template callable_impl
        < Class
        , MappingModel::maps_to_void
        , ThreadAbility::threads_allowed_t::value
        , MappingModel
        , typename WiringModel<MappingModel, Class/*, F*/>::type
        , typename OutArrTransform<MappingModel>::type
        , ThreadAbility
        >
{
    typedef callable<Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
            callable_t;

    typedef typename detail::callable_arity<MappingModel::in_arity>::template callable_impl
            < Class
            , MappingModel::maps_to_void
            , ThreadAbility::threads_allowed_t::value
            , MappingModel
            , typename WiringModel<MappingModel, Class/*, F*/>::type
            , typename OutArrTransform<MappingModel>::type
            , ThreadAbility
            >
            callable_impl_t;

    //__________________________________________________________________________
    callable(typename WiringModel<MappingModel, Class>::type::wiring_model_config_t const & wmc)
      : callable_impl_t(wmc)
    {}
};

}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_CALLABLE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref(z, n, data)   \
    BOOST_PP_COMMA_IF(n) boost::python::object const &

template <>
struct callable_arity<N>
{
    BOOST_STATIC_CONSTANT(unsigned, in_arity = N);

    template <
          class Class
        , bool has_void_return
        , bool allows_threads
        , class MappingModel
        , class WiringModel
        , class OutArrTransform
        , class ThreadAbility
    >
    struct callable_impl;

    //--------------------------------------------------------------------------
    // Partial specialization for standalone functions with non-void-return and
    // threads allowed.
    template <class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<numpy::mpl::unspecified, false, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <numpy::mpl::unspecified, false, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector<
              // This should be the return type of the call function but since
              // it is always boost::python::object we use it for tagging the
              // base type of the callable object.
              typename callable_impl_base_t::callable_base_t::callable_mf_base_t
            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
            , python::object &
            , int
            > bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
              BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , python::object & out_obj
            , int nthreads
        ) const
        {
            // Create a dummy self instance.
            mpl::unspecified self;

            typedef detail::callable_call_arity<N>::template callable_call
                    <false, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , out_obj
                , nthreads);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( kwargs
                  , python::arg("out")=python::object()
                  , python::arg("nthreads")=1
                  )
                , bp_call_signature_t()
            );
        }
    };

    //--------------------------------------------------------------------------
    // Specialization for standalone functions with non-void-return and threads
    // forbidden.
    template <class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<numpy::mpl::unspecified, false, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <numpy::mpl::unspecified, false, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector<
              // This should be the return type of the call function but since
              // it is always boost::python::object we use it for tagging the
              // base type of the callable object.
              typename callable_impl_base_t::callable_base_t::callable_mf_base_t
            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
            , python::object &
            > bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
              BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , python::object & out_obj
        ) const
        {
            // Create a dummy self instance.
            mpl::unspecified self;

            typedef detail::callable_call_arity<N>::template callable_call
                    <false, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , out_obj
                , 1);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( kwargs
                  , python::arg("out")=python::object()
                  )
                , bp_call_signature_t()
            );
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for standalone functions with void-return and
    // threads allowed.
    template <class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<numpy::mpl::unspecified, true, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <numpy::mpl::unspecified, true, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector<
              // This should be the return type of the call function but since
              // it is always boost::python::object we use it for tagging the
              // base type of the callable object.
              typename callable_impl_base_t::callable_base_t::callable_mf_base_t
            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
            , int
            > bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
              BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , int nthreads
        ) const
        {
            // Create a dummy self instance.
            mpl::unspecified self;

            typedef detail::callable_call_arity<N>::template callable_call
                    <true, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , nthreads);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( kwargs
                  , python::arg("nthreads")=1
                  )
                , bp_call_signature_t()
            );
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for standalone functions with void-return and
    // threads forbidden.
    template <class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<numpy::mpl::unspecified, true, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <numpy::mpl::unspecified, true, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector<
              // This should be the return type of the call function but since
              // it is always boost::python::object we use it for tagging the
              // base type of the callable object.
              typename callable_impl_base_t::callable_base_t::callable_mf_base_t
            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
            > bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
            BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
        ) const
        {
            // Create a dummy self instance.
            mpl::unspecified self;

            typedef detail::callable_call_arity<N>::template callable_call
                    <true, numpy::mpl::unspecified, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , 1);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( kwargs
                  )
                , bp_call_signature_t()
            );
        }
    };

    //--------------------------------------------------------------------------
    // Specialization for member functions with non-void-return and threads
    // allowed.
    template <class Class, class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<Class, false, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <Class, false, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector
                  // This should be the return type of the call function but
                  // since it is always boost::python::object we use it for
                  // tagging the base type of the callable object.
                < typename callable_impl_base_t::callable_base_t::callable_mf_base_t
                , Class &
                , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
                , python::object &
                , int
                >
                bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
              Class & self
            , BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , python::object & out_obj
            , int nthreads
        ) const
        {
            typedef detail::callable_call_arity<N>::template callable_call
                    <false, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , out_obj
                , nthreads);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( python::arg("self")
                  , kwargs
                  , python::arg("out")=python::object()
                  , python::arg("nthreads")=1
                  )
                , bp_call_signature_t()
            );
        }
    };

    //--------------------------------------------------------------------------
    // Specialization for member functions with void-return and threads allowed.
    template <class Class, class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<Class, true, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <Class, true, true, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector
                  // This should be the return type of the call function but
                  // since it is always boost::python::object we use it for
                  // tagging the base type of the callable object.
                < typename callable_impl_base_t::callable_base_t::callable_mf_base_t
                , Class &
                , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
                , int
                >
                bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
              Class & self
            , BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , int nthreads
        ) const
        {
            typedef detail::callable_call_arity<N>::template callable_call
                    <true, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , nthreads);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( python::arg("self")
                  , kwargs
                  , python::arg("nthreads")=1
                  )
                , bp_call_signature_t()
            );
        }
    };

    //--------------------------------------------------------------------------
    // Specialization for member functions with non-void-return and threads
    // forbidden.
    template <class Class, class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<Class, false, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <Class, false, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector
                  // This should be the return type of the call function but
                  // since it is always boost::python::object we use it for
                  // tagging the base type of the callable object.
                < typename callable_impl_base_t::callable_base_t::callable_mf_base_t
                , Class &
                , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
                , python::object &
                >
                bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
              Class & self
            , BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
            , python::object & out_obj
        ) const
        {
            typedef detail::callable_call_arity<N>::template callable_call
                    <false, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , out_obj
                , 1);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( python::arg("self")
                  , kwargs
                  , python::arg("out")=python::object()
                  )
                , bp_call_signature_t()
            );
        }
    };

    //--------------------------------------------------------------------------
    // Specialization for member functions with void-return and threads
    // forbidden.
    template <class Class, class MappingModel, class WiringModel, class OutArrTransform, class ThreadAbility>
    struct callable_impl<Class, true, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
      : callable_impl_base<N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
    {
        typedef callable_impl
                <Class, true, false, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_t;

        typedef callable_impl_base
                <N, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                callable_impl_base_t;

        typedef boost::mpl::vector
                  // This should be the return type of the call function but
                  // since it is always boost::python::object we use it for
                  // tagging the base type of the callable object.
                < typename callable_impl_base_t::callable_base_t::callable_mf_base_t
                , Class &
                , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref, ~)
                >
                bp_call_signature_t;

        callable_impl(typename WiringModel::wiring_model_config_t const & wmc)
          : callable_impl_base_t(wmc)
        {}

        python::object
        call(
              Class & self
            , BOOST_PP_ENUM_PARAMS(N, python::object const & in_obj_)
        ) const
        {
            typedef detail::callable_call_arity<N>::template callable_call
                    <true, Class, MappingModel, WiringModel, OutArrTransform, ThreadAbility>
                    callable_call_t;

            return callable_call_t::call(
                  this->wiring_model_
                , self
                , BOOST_PP_ENUM_PARAMS(N, in_obj_)
                , 1);
        }

        template <class Keywords>
        python::object
        make_function(
            Keywords const & kwargs
        ) const
        {
            // Trigger a compiler error when the number of specified keyword
            // arguments does not match the number of function arguments.
            typedef typename detail::error::less_or_more_keywords_than_function_arguments<
                  Keywords::size
                , N
                >::too_few_or_many_keywords assertion;

            return python::make_function(
                  (callable_impl_t*)this
                , python::default_call_policies()
                , ( python::arg("self")
                  , kwargs
                  )
                , bp_call_signature_t()
            );
        }
    };
};

#undef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE__bp_object_const_ref

#undef N

#endif // BOOST_PP_IS_ITERATING
