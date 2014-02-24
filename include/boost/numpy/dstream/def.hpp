/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/dstream/def.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::numpy::dstream::def and
 *        boost::numpy::dstream::classdef functions to expose a
 *        C++ function or member function to python with numpy data stream
 *        support.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_DEF_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DEF_HPP_INCLUDED

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/iteration/local.hpp>
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
#include <boost/numpy/mpl/types_from_fctptr_signature.hpp>
#include <boost/numpy/mpl/unspecified.hpp>
#include <boost/numpy/dstream/callable.hpp>
#include <boost/numpy/dstream/defaults.hpp>
#include <boost/numpy/dstream/detail/def_helper.hpp>
#include <boost/numpy/dstream/mapping/converter/arg_type_to_core_shape.hpp>
#include <boost/numpy/dstream/mapping/converter/return_type_to_out_mapping.hpp>
#include <boost/numpy/dstream/wiring/default_wiring_model_selector_fwd.hpp>
#include <boost/numpy/dstream/wiring/models/scalar_callable.hpp>

#include <boost/python/signature.hpp>
#include <boost/python/object/add_to_namespace.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

//==============================================================================
/** The default_mapping_definition_selector template provides a select template
 *  to construct the appropriate mapping definition based on the to-be-exposed
 *  function's output and input types. It uses the mapping converters for
 *  converting the output and input types to output and input mapping types.
 */
template <unsigned InArity>
struct default_mapping_definition_selector;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/def.hpp>, 1))
#include BOOST_PP_ITERATE()

//==============================================================================
template <
      class Class
    , class IOTypes
    , class UserMappingModelSelector
>
struct default_selectors
{
    // Choose the mapping model selector class: Either the user defined one, or
    // the default one based on the IO types of the function.
    typedef typename boost::mpl::if_
                < typename is_same<UserMappingModelSelector, numpy::mpl::unspecified>::type
                , typename default_mapping_model_selector_from_io_types
                      < IOTypes::in_arity
                      , IOTypes
                      >::default_mapping_model_selector_t
                , UserMappingModelSelector
                >::type
            mapping_model_selector_t;

    typedef typename mapping_model_selector_t::template select<IOTypes>::type
            mapping_model_t;

    typedef typename default_wiring_model_selector<mapping_model_t, Class>::type
            wiring_model_selector_t;

    typedef typename default_out_arr_transform_selector<mapping_model_t>::type
            out_arr_transform_selector_t;

    typedef default_thread_ability
            thread_ability_selector_t;
};

//==============================================================================
template <
      class F
    , class KW
    , class Class
    , class IOTypes
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
    , IOTypes const &
    , MappingModelSelector const &
    , WiringModelSelector const &
    , OutArrTransformSelector const &
    , ThreadAbilitySelector const &
)
{
    typedef callable<
                  Class
                //, IOTypes
                , typename MappingModelSelector::template select<IOTypes>::type
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

}// namespace detail

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
// object needs to be in existence when calling the classdef(...) function.
//______________________________________________________________________________
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, 5, <boost/numpy/dstream/def.hpp>, 3))
#include BOOST_PP_ITERATE()

namespace detail {

template <
      unsigned Arity
    , class F
    , class KW
    , BOOST_PP_ENUM_BINARY_PARAMS_Z(1, 5, class A, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
class method_visitor;

template <
      unsigned Arity
    , class F
    , class KW
    , BOOST_PP_ENUM_BINARY_PARAMS_Z(1, 5, class A, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
class staticmethod_visitor;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, 5, <boost/numpy/dstream/def.hpp>, 4))
#include BOOST_PP_ITERATE()

}// namespace detail

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, 5, <boost/numpy/dstream/def.hpp>, 5))
#include BOOST_PP_ITERATE()

}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_DEF_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

#define IN_ARITY BOOST_PP_ITERATION()

template <>
struct default_mapping_definition_selector<IN_ARITY>
  : mapping::mapping_definition_selector_type
{
    template <class FTypes>
    struct select
    {
        // Construct a boost::numpy::dstream::mapping::detail::out type based on
        // the FTypes::out_t type.
        typedef typename mapping::converter::detail::return_type_to_out_mapping<typename FTypes::out_t>::type
                out_mapping_t;

        #define BOOST_PP_LOCAL_MACRO(n) \
            typedef typename mapping::converter::detail::arg_type_to_core_shape<typename FTypes:: BOOST_PP_CAT(in_t,n) >::type BOOST_PP_CAT(in_core_shape_t,n);
        #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(IN_ARITY, 1))
        #include BOOST_PP_LOCAL_ITERATE()
        typedef mapping::detail::in<IN_ARITY>::core_shapes< BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_core_shape_t) >
                in_mapping_t;

        typedef mapping::detail::definition<out_mapping_t, in_mapping_t>
                type;
    };
};

#undef IN_ARITY

#elif BOOST_PP_ITERATION_FLAGS() == 2

template <
      class Class
    , class F
    , class KW
    , class IOTypes
    , class MappingModelSelector
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void make_def_with_iotypes_and_mapping_model_selector(
      python::scope const& sc
    , char const* name
    , Class*
    , F f
    , KW const& kwargs
    , IOTypes const & io_types
    , MappingModelSelector const &
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    // At this point we have a mapping model selector selected (either a default
    // one or the user specified one). Now we select the default wiring model
    // selector based on the fixed mapping model selector and its mapping model.
    typedef default_selectors<Class, IOTypes, MappingModelSelector>
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
        , io_types
        , helper.get_mapping_model_selector()
        , helper.get_wiring_model_selector()
        , helper.get_out_arr_transform_selector()
        , helper.get_thread_ability_selector()
    );
}

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
    , Signature const & sig
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    typedef io_types_from_signature<Class, Signature>
            io_types_t;

    typedef default_selectors<Class, io_types_t, numpy::mpl::unspecified>
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

    make_def_with_iotypes_and_mapping_model_selector(
          sc
        , name
        , (Class*)NULL
        , f
        , kwargs
        , io_types_t()
        , helper.get_mapping_model_selector()
        BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a)
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

#elif BOOST_PP_ITERATION_FLAGS() == 4

template <
      class F
    , class KW
    , BOOST_PP_ENUM_PARAMS_Z(1, 5, class A)
>
class method_visitor<
      N
    , F
    , KW
    , BOOST_PP_ENUM_PARAMS_Z(1, 5, A)
>
  : public python::def_visitor< method_visitor<N, F, KW, BOOST_PP_ENUM_PARAMS_Z(1, 5, A)> >
{
  public:
    method_visitor(
          char const * name
        , F f
        , KW const & kwargs
        BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
    )
      : m_name(name)
      , m_f(f)
      , m_kwargs(kwargs)
      #define BOOST_NUMPY_DSTREAM_DEF_m_a(z, n, data) \
              , BOOST_PP_CAT(m_a,n) ( BOOST_PP_CAT(a,n) )
      BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DEF_m_a, ~)
      #undef BOOST_NUMPY_DSTREAM_DEF_m_a
    {}

  private:
    friend class python::def_visitor_access;

    template <class Class>
    void visit(Class& cls) const
    {
        dstream::classdef(
              cls
            , m_name
            , m_f
            , m_kwargs
            BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, m_a)
        );
    }

    char const * m_name;
    F m_f;
    KW const & m_kwargs;
    #define BOOST_NUMPY_DSTREAM_DEF_m_a(z, n, data) \
            BOOST_PP_CAT(A,n) const & BOOST_PP_CAT(m_a,n) ;
    BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DEF_m_a, ~)
    #undef BOOST_NUMPY_DSTREAM_DEF_m_a
};

template <
      class F
    , class KW
    , BOOST_PP_ENUM_PARAMS_Z(1, 5, class A)
>
class staticmethod_visitor<
      N
    , F
    , KW
    , BOOST_PP_ENUM_PARAMS_Z(1, 5, A)
>
  : public python::def_visitor< staticmethod_visitor<N, F, KW, BOOST_PP_ENUM_PARAMS_Z(1, 5, A)> >
{
  public:
    staticmethod_visitor(
          char const * name
        , F f
        , KW const & kwargs
        BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
    )
      : m_name(name)
      , m_f(f)
      , m_kwargs(kwargs)
      #define BOOST_NUMPY_DSTREAM_DEF_m_a(z, n, data) \
              , BOOST_PP_CAT(m_a,n) ( BOOST_PP_CAT(a,n) )
      BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DEF_m_a, ~)
      #undef BOOST_NUMPY_DSTREAM_DEF_m_a
    {}

  private:
    friend class python::def_visitor_access;

    template <class Class>
    void visit(Class& cls) const
    {
        dstream::def(
              cls
            , m_name
            , m_f
            , m_kwargs
            BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, m_a)
        );
        cls.staticmethod(m_name);
    }

    char const * m_name;
    F m_f;
    KW const & m_kwargs;
    #define BOOST_NUMPY_DSTREAM_DEF_m_a(z, n, data) \
            BOOST_PP_CAT(A,n) const & BOOST_PP_CAT(m_a,n) ;
    BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_DEF_m_a, ~)
    #undef BOOST_NUMPY_DSTREAM_DEF_m_a
};

#elif BOOST_PP_ITERATION_FLAGS() == 5

template <
      class F
    , class KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
detail::method_visitor<
      N
    , F
    , KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, A)
>
method(
      char const * name
    , F f
    , KW const & kwargs
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    typedef detail::method_visitor<
                  N
                , F
                , KW
                BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, A)
            >
            method_visitor_t;

    method_visitor_t visitor(
          name
        , f
        , kwargs
        BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a)
    );

    return visitor;
}

template <
      class F
    , class KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
detail::staticmethod_visitor<
      N
    , F
    , KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, A)
>
staticmethod(
      char const * name
    , F f
    , KW const & kwargs
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    typedef detail::staticmethod_visitor<
                  N
                , F
                , KW
                BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, A)
            >
            staticmethod_visitor_t;

    staticmethod_visitor_t visitor(
          name
        , f
        , kwargs
        BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a)
    );

    return visitor;
}

#endif // BOOST_PP_ITERATION_FLAGS

#undef N

#endif // BOOST_PP_IS_ITERATING
