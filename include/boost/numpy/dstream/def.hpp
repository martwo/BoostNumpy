/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * @file    boost/numpy/dstream/def.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * @brief This file defines the boost::numpy::dstream::def function to expose a
 *        C++ function to python with numpy data stream support.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_DEF_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DEF_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/size.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/detail/prefix.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/types.hpp>
#include <boost/numpy/dstream/callable.hpp>
#include <boost/numpy/dstream/defaults.hpp>
#include <boost/numpy/dstream/detail/def_helper.hpp>
#include <boost/numpy/dstream/mapping/models/NxS_to_S.hpp>
#include <boost/numpy/dstream/wiring/models/scalar_callable.hpp>
#include <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis_if_single_input_and_scalarize.hpp>

#include <boost/python/signature.hpp>
#include <boost/python/object/add_to_namespace.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

//==============================================================================
/** The mapping_model_selector template selects the appropriate mapping model
 *  based on the given output and input argument types of the to-be-exposed C++
 *  function/method.
 *  By default the _NxS_to_S mapping model is used. In order to use other
 *  mapping models, one has to specialize the mapping_model_selector template.
 */
template <
      unsigned InArity
    , class OutT
    , BOOST_PP_ENUM_BINARY_PARAMS_Z(1, BOOST_NUMPY_LIMIT_INPUT_ARITY, class InT_, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
struct mapping_model_selector;

template <
      int in_arity
    , class Signature
    , class OutT
    , int sig_arg_offset
>
struct mapping_model_selector_from_signature_impl;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/def.hpp>, 1))
#include BOOST_PP_ITERATE()

//==============================================================================
/** The wiring_model_selector template selects the appropriate wiring model
 *  based on the given mapping model, the class, and the output type and input
 *  argument types of the to-be-exposed C++ function/method.
 */
template <
      class MappingModel
    , class Class
>
struct wiring_model_selector
  : wiring::wiring_model_selector_type
{
    template <
          class _MappingModel
        , class _Class
    >
    struct wiring_model
    {
        typedef wiring::model::scalar_callable
                < _MappingModel, _Class >
                type;
    };
};

//==============================================================================
template <
      int InArity
    , class MappingModel
>
struct out_arr_transform_selector
  : out_arr_transforms::out_arr_transform_selector_type
{
    template <
          int _InArity
        , class _MappingModel
    >
    struct out_arr_transform
    {
        typedef out_arr_transforms::squeeze_first_axis_if_single_input_and_scalarize
                < _InArity, _MappingModel >
                type;
    };
};

//==============================================================================
template <bool is_mfp, class Signature>
struct mapping_model_selector_from_signature
{
    typedef typename boost::mpl::begin<Signature>::type::type
            out_t;

    typedef mapping_model_selector_from_signature_impl
            < boost::mpl::size<Signature>::value - 2
            , Signature
            , out_t
            , 2
            >
            mapping_model_selector_from_signature_impl_t;

    typedef typename mapping_model_selector_from_signature_impl_t::mapping_model_selector_t
            mapping_model_selector_t;
};

// Specialization for standalone functions.
template <class Signature>
struct mapping_model_selector_from_signature<false, Signature>
{
    typedef typename boost::mpl::begin<Signature>::type::type
            out_t;

    typedef mapping_model_selector_from_signature_impl
            < boost::mpl::size<Signature>::value - 1
            , Signature
            , out_t
            , 1
            >
            mapping_model_selector_from_signature_impl_t;

    typedef typename mapping_model_selector_from_signature_impl_t::mapping_model_selector_t
            mapping_model_selector_t;
};

template <
      class Class
    , class Signature
>
struct default_selectors
{
    // Choose the mapping model selector class by applying the function
    // signature.
    typedef typename mapping_model_selector_from_signature
                < boost::mpl::not_< boost::is_same<Class, numpy::mpl::unspecified> >::value
                , Signature
                >::mapping_model_selector_t
            mapping_model_selector_t;
    typedef typename mapping_model_selector_t::type
            mapping_model_t;

    typedef wiring_model_selector<mapping_model_t, Class>
            wiring_model_selector_t;

    typedef out_arr_transform_selector<mapping_model_t::in_arity, mapping_model_t>
            out_arr_transform_selector_t;

    typedef default_thread_ability
            thread_ability_selector_t;
};

//==============================================================================
template <
      class F
    , class KW
    , class Class
    , class MappingModelSelector
    , class WiringModelSelector
    , class OutArrTransformSelector
    , class ThreadAbilitySelector
>
void create_and_add_callable_object(
      python::scope const& scope
    , char const* name
    , F f
    , KW const& kwargs
    , char const* doc
    , Class const*
    , MappingModelSelector const &
    , WiringModelSelector const &
    , OutArrTransformSelector const &
    , ThreadAbilitySelector const &
)
{
    typedef callable<
                  Class
                , typename MappingModelSelector::type
                , WiringModelSelector::template wiring_model
                , OutArrTransformSelector::template out_arr_transform
                , typename ThreadAbilitySelector::type
                >
            callable_t;

    // Create a wiring model configuration object with the
    // function/member function pointer as setting.
    typedef typename callable_t::wiring_model_config_t
            wiring_model_config_t;
    wiring_model_config_t wmc((boost::numpy::detail::cfg()=f));

    // Create a callable_t object on the heap.
    callable_t* callable = new callable_t(wmc);

    // Finally, create a python function object via boost::python within the
    // given scope, where scope could be a python module or a python class.
    python::objects::add_to_namespace(scope, name, callable->make_function(kwargs), doc);
}

//==============================================================================
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, 5, <boost/numpy/dstream/def.hpp>, 2))
#include BOOST_PP_ITERATE()

}/*namespace detail*/

// The def(...)/classdef(...) functions need at least 3 arguments:
//   - the name of the python function,
//   - the pointer to the to-be-exposed C++ function, and
//   - the names of the keyword arguments.
// Optionally, a
//   - boost::python::scope, a
//   - docstring, a
//   - mapping model selector, a
//   - wiring model selector, an
//   - out array transform selector, and a
//   - thread ability selector
// can be specified in any order.
//
// When exposing a class method, either the scope, i.e. the class_
// object needs to be supplied, or a boost::python::scope object of the class_
// object needs to in existence when calling the classdef(...) function.
//______________________________________________________________________________
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, 5, <boost/numpy/dstream/def.hpp>, 3))
#include BOOST_PP_ITERATE()

}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_DEF_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

template <
      class OutT
    , BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_LIMIT_INPUT_ARITY, class InT_)
>
struct mapping_model_selector<N, OutT, BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)>
  : mapping::mapping_model_selector_type
{
    typedef mapping::model::NxS_to_S<N, OutT BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, InT_)>
            type;
};

#define BOOST_NUMPY_DSTREAM_DEF__InT(z, n, data) \
    typedef typename boost::mpl::at<Signature, boost::mpl::long_<sig_arg_offset + n> >::type BOOST_PP_CAT(InT_,n);

template <
      class Signature
    , class OutT
    , int sig_arg_offset
>
struct mapping_model_selector_from_signature_impl<
      N
    , Signature
    , OutT
    , sig_arg_offset
>
{
    BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DEF__InT, ~)

    typedef mapping_model_selector<N, OutT BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, InT_)>
            mapping_model_selector_t;
};

#undef BOOST_NUMPY_DSTREAM_DEF__InT

#elif BOOST_PP_ITERATION_FLAGS() == 2

template <
      class Class
    , class F
    , class KW
    , class Signature
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void make_def_with_signature(
      python::scope const& sc
    , char const* name
    , Class*
    , F f
    , KW const& kwargs
    , Signature const &
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    typedef default_selectors<Class, Signature>
            default_selectors_t;

    typedef def_helper
                < typename default_selectors_t::mapping_model_selector_t
                , typename default_selectors_t::wiring_model_selector_t
                , typename default_selectors_t::out_arr_transform_selector_t
                , typename default_selectors_t::thread_ability_selector_t
                BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, A)
                >
            def_helper_t;
    def_helper_t helper(BOOST_PP_ENUM_PARAMS_Z(1, N, a));

    create_and_add_callable_object(
          sc, name, f, kwargs
        , helper.get_doc()
        , (Class*)(NULL)
        , helper.get_mapping_model_selector()
        , helper.get_wiring_model_selector()
        , helper.get_out_arr_transform_selector()
        , helper.get_thread_ability_selector()
    );
}

template <
      class F
    , class KW
    , class Signature
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void def_with_signature(
      python::scope const& sc
    , char const* name
    , F f
    , KW const& kwargs
    , Signature const & sig
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    typedef numpy::mpl::unspecified
            class_t;

    make_def_with_signature(sc, name, (class_t*)(NULL), f, kwargs, sig BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a));
}

template <
      class F
    , class KW
    , class Signature
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void classdef_with_signature(
      python::scope const& sc
    , char const* name
    , F f
    , KW const& kwargs
    , Signature const & sig
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    // Get the class type either from the first argument type of the provided
    // static function or from the member function pointer type.
    // In both cases it's the second type of the signature MPL type vector.
    typedef typename boost::remove_reference< typename boost::mpl::at<Signature, boost::mpl::long_<1> >::type >::type
            class_t;

    make_def_with_signature(sc, name, (class_t*)(NULL), f, kwargs, sig BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a));
}

#elif BOOST_PP_ITERATION_FLAGS() == 3

template <
      class F
    , class KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void
def(
      python::scope const& sc
    , char const * name
    , F f
    , KW const & kwargs
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    detail::def_with_signature(sc, name, f, kwargs, python::detail::get_signature(f) BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a));
}

template <
      class F
    , class KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void
def(
      char const * name
    , F f
    , KW const & kwargs
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    // Get the current scope by creating a python::scope object.
    python::scope const sc;

    def(sc, name, f, kwargs BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a));
}

template <
      class F
    , class KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void
classdef(
      python::scope const& sc
    , char const * name
    , F f
    , KW const & kwargs
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    detail::classdef_with_signature(sc, name, f, kwargs, python::detail::get_signature(f) BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a));
}

template <
      class F
    , class KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void
classdef(
      char const * name
    , F f
    , KW const & kwargs
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    // Get the current scope by creating a python::scope object and assuming
    // that it is indeed a class_ object.
    python::scope const sc;

    classdef(sc, name, f, kwargs BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a));
}

#endif // BOOST_PP_ITERATION_FLAGS

#undef N

#endif // BOOST_PP_IS_ITERATING
