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

#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/size.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/python.hpp>
#include <boost/python/signature.hpp>
#include <boost/python/object/add_to_namespace.hpp>
#include <boost/python/object/py_function.hpp>
#include <boost/python/object/function_object.hpp>
#include <boost/python/refcount.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/types_from_fctptr_signature.hpp>
#include <boost/numpy/mpl/unspecified.hpp>
#include <boost/numpy/dstream/threading.hpp>
#include <boost/numpy/dstream/detail/callable.hpp>
#include <boost/numpy/dstream/detail/caller.hpp>
#include <boost/numpy/dstream/detail/def_helper.hpp>
#include <boost/numpy/dstream/mapping.hpp>
#include <boost/numpy/dstream/mapping/converter/arg_type_to_core_shape.hpp>
#include <boost/numpy/dstream/mapping/converter/return_type_to_out_mapping.hpp>
#include <boost/numpy/dstream/wiring.hpp>

// Include all built-in wiring models.
#include <boost/numpy/dstream/wiring/models/scalars_to_scalar_callable.hpp>
#include <boost/numpy/dstream/wiring/models/scalars_to_vector_of_scalar_callable.hpp>

#define BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS 4

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
template <unsigned InArity, class FTypes>
struct default_mapping_definition_selector;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/def.hpp>, 1))
#include BOOST_PP_ITERATE()

//==============================================================================
template <
      class FTypes
    , class MappingDefinition
>
struct default_selectors
{
    typedef MappingDefinition
            mapping_definition_t;

    typedef typename wiring::default_wiring_model_selector<MappingDefinition, FTypes>::type
            wiring_model_selector_t;

    typedef default_thread_ability
            thread_ability_selector_t;
};

//==============================================================================
template <
      class F
    , class FTypes
    , class KW
    , class MappingDefinition
    , class WiringModelSelector
    , class ThreadAbilitySelector
>
void create_and_add_py_function(
      python::scope const& scope
    , char const* name
    , F f
    , FTypes *
    , KW const& kwargs
    , char const* doc
    , MappingDefinition const &
    , WiringModelSelector const &
    , ThreadAbilitySelector const &
)
{
    // Construct a callable object and a caller object that takes that callable
    // object as argument, so it can call this callable.
    typedef callable<
                  F
                , FTypes
                , MappingDefinition
                , WiringModelSelector::template select
                , typename ThreadAbilitySelector::type
            >
            callable_t;

    typedef caller<callable_t>
            caller_t;

    callable_t callable(f);
    caller_t caller(callable);

    // Create a py_function object that takes a caller object for implementing
    // the call procedure.
    python::objects::py_function pyfunc(caller, typename callable_t::signature_t());

    // Create a python::object holding a Python function object.
    python::object pyfunct_obj = python::objects::function_object(
          pyfunc
        , callable_t::template make_kwargs(kwargs).range()
    );

    // Finally, add the Python function object to the Python namespace scope,
    // where scope could be a python module or a python class.
    python::objects::add_to_namespace(scope, name, pyfunct_obj, doc);
}

//==============================================================================
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, <boost/numpy/dstream/def.hpp>, 2))
#include BOOST_PP_ITERATE()

}// namespace detail

// The def(...) function needs at least 3 arguments:
//   - the name of the python function,
//   - the pointer to the to-be-exposed C++ function, and
//   - the names of the keyword arguments.
// Optionally, a
//   - boost::python::scope
// can be specified as first argument and
// optionally, a
//   - docstring, a
//   - mapping definition, a
//   - wiring model selector, and a
//   - thread ability selector
// can be specified in any order as last arguments.
//
// When exposing a class method, either the scope, i.e. the class_
// object needs to be supplied, or a boost::python::scope object of the class_
// object needs to be in existence when calling the def(...) function.
//______________________________________________________________________________
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, <boost/numpy/dstream/def.hpp>, 3))
#include BOOST_PP_ITERATE()

namespace detail {

template <
      unsigned Arity
    , class F
    , class KW
    , BOOST_PP_ENUM_BINARY_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, class A, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
class method_visitor;

template <
      unsigned Arity
    , class F
    , class KW
    , BOOST_PP_ENUM_BINARY_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, class A, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
class staticmethod_visitor;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, <boost/numpy/dstream/def.hpp>, 4))
#include BOOST_PP_ITERATE()

}// namespace detail

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (0, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, <boost/numpy/dstream/def.hpp>, 5))
#include BOOST_PP_ITERATE()

}// namespace dstream
}// namespace numpy
}// namespace boost

#undef BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS

#endif // !BOOST_NUMPY_DSTREAM_DEF_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

#define IN_ARITY BOOST_PP_ITERATION()

template <class FTypes>
struct default_mapping_definition_selector<IN_ARITY, FTypes>
{
    struct select
    {
        // Construct a boost::numpy::dstream::mapping::detail::out type based on
        // the FTypes::return_type type.
        typedef typename mapping::converter::detail::return_type_to_out_mapping<typename FTypes::return_type>::type
                out_mapping_t;

        #define BOOST_NUMPY_DEF(z, n, data) \
            typedef typename mapping::converter::detail::arg_type_to_core_shape<typename FTypes:: BOOST_PP_CAT(arg_type,n) >::type \
                    BOOST_PP_CAT(in_core_shape_t,n);
        BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF

        typedef mapping::detail::in<IN_ARITY>::core_shapes< BOOST_PP_ENUM_PARAMS_Z(1, IN_ARITY, in_core_shape_t) >
                in_mapping_t;

        typedef mapping::detail::definition<out_mapping_t, in_mapping_t>
                type;
    };
};

#undef IN_ARITY

#else
#if BOOST_PP_ITERATION_FLAGS() == 2

template <
      class F
    , class FTypes
    , class KW
    , class MappingDefinition
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void def_with_ftypes_and_mapping_definition(
      python::scope const& sc
    , char const* name
    , F f
    , FTypes *
    , KW const & kwargs
    , MappingDefinition const &
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    // At this point we have a mapping definition selected (either a default
    // one or the user specified one). Now we select the default wiring model
    // selector based on the fixed mapping definition and the function's types.
    typedef default_selectors<FTypes, MappingDefinition>
            default_selectors_t;

    typedef def_helper<
                  typename default_selectors_t::mapping_definition_t
                , typename default_selectors_t::wiring_model_selector_t
                , typename default_selectors_t::thread_ability_selector_t
                BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, A)
            >
            def_helper_t;
    def_helper_t const helper = def_helper_t(BOOST_PP_ENUM_PARAMS_Z(1, N, a));

    create_and_add_py_function(
          sc
        , name
        , f
        , (FTypes*)NULL
        , kwargs
        , helper.get_doc()
        , helper.get_mapping_definition()
        , helper.get_wiring_model_selector()
        , helper.get_thread_ability_selector()
    );
}

template <
      class F
    , class FTypes
    , class KW
    , class MappingDefinition
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void def_with_ftypes_selecting_mapping_definition(
      python::scope const& sc
    , char const* name
    , F f
    , FTypes *
    , KW const & kwargs
    , MappingDefinition const &
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    // Choose the mapping definition: Either the user defined
    // one, or the default one based on the argument and return types of the
    // to-be-exposed function.
    typedef typename boost::mpl::eval_if<
                  typename boost::is_same<MappingDefinition, mapping::detail::null_definition>::type
                , typename default_mapping_definition_selector<FTypes::arity, FTypes>::select
                , MappingDefinition
                >::type
            mapping_definition_t;

    def_with_ftypes_and_mapping_definition(
          sc
        , name
        , f
        , (FTypes*)NULL
        , kwargs
        , mapping_definition_t()
        BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a)
    );
}

template <
      class F
    , class FTypes
    , class KW
    BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class A)
>
void def_with_ftypes(
      python::scope const& sc
    , char const* name
    , F f
    , FTypes*
    , KW const& kwargs
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    typedef dstream::detail::def_helper<
                  mapping::detail::null_definition
                , wiring::detail::null_wiring_model_selector
                , threading::detail::null_thread_ability_selector
                BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, A)
            >
            def_helper_t;
    def_helper_t const helper = def_helper_t(BOOST_PP_ENUM_PARAMS_Z(1, N, a));

    def_with_ftypes_selecting_mapping_definition(
          sc
        , name
        , f
        , (FTypes*)NULL
        , kwargs
        , helper.get_mapping_definition()
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
    , Signature const &
    BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, A, const & a)
)
{
    typedef typename numpy::mpl::types_from_fctptr_signature<F, Signature>::type
            f_types_t;

    def_with_ftypes(sc, name, f, (f_types_t*)NULL, kwargs BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, a));
}

#else
#if BOOST_PP_ITERATION_FLAGS() == 3

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

#else
#if BOOST_PP_ITERATION_FLAGS() == 4

template <
      class F
    , class KW
    , BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, class A)
>
class method_visitor<
      N
    , F
    , KW
    , BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, A)
>
  : public python::def_visitor< method_visitor<N, F, KW, BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, A)> >
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
      #if N
          #define BOOST_NUMPY_DEF(z, n, data) \
              , BOOST_PP_CAT(m_a,n) ( BOOST_PP_CAT(a,n) )
          BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
          #undef BOOST_NUMPY_DEF
      #endif
    {}

  private:
    friend class python::def_visitor_access;

    template <class ClassT>
    void visit(ClassT & cls) const
    {
        dstream::def(
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
    #if N
        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(A,n) const & BOOST_PP_CAT(m_a,n) ;
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
    #endif
};

template <
      class F
    , class KW
    , BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, class A)
>
class staticmethod_visitor<
      N
    , F
    , KW
    , BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, A)
>
  : public python::def_visitor< staticmethod_visitor<N, F, KW, BOOST_PP_ENUM_PARAMS_Z(1, BOOST_NUMPY_DSTREAM_DEF_MAX_OPTIONAL_ARGS, A)> >
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
      #if N
          #define BOOST_NUMPY_DEF(z, n, data) \
              , BOOST_PP_CAT(m_a,n) ( BOOST_PP_CAT(a,n) )
          BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
          #undef BOOST_NUMPY_DEF
      #endif
    {}

  private:
    friend class python::def_visitor_access;

    template <class ClassT>
    void visit(ClassT & cls) const
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
    #if N
        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_CAT(A,n) const & BOOST_PP_CAT(m_a,n) ;
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
    #endif
};

#else
#if BOOST_PP_ITERATION_FLAGS() == 5

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

#endif // BOOST_PP_ITERATION_FLAGS() == 5
#endif // BOOST_PP_ITERATION_FLAGS() == 4
#endif // BOOST_PP_ITERATION_FLAGS() == 3
#endif // BOOST_PP_ITERATION_FLAGS() == 2
#endif // BOOST_PP_ITERATION_FLAGS() == 1

#undef N

#endif // BOOST_PP_IS_ITERATING
