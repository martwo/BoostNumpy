/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 * This file has been adopted from <boost/python/detail/caller.hpp>.
 *
 * @file    boost/numpy/dstream/detail/caller.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the caller template that is used as the caller
 *        implementation for boost::python of boost::numpy generalized universal
 *        functions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_DETAIL_CALLER_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_CALLER_HPP_INCLUDED

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/dec.hpp>
#include <boost/preprocessor/if.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/mpl/int.hpp>
#include <boost/mpl/next.hpp>
#include <boost/mpl/size.hpp>

// For this dedicated caller, we use function templates from the boost::python
// caller.
#include <boost/python/detail/caller.hpp>

#include <boost/numpy/limits.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <unsigned> struct invoke_arity;
template <unsigned> struct caller_arity;

// The +1 is for the class instance and the +2 is for the possible automatically
// added extra arguments ("out" and "nthreads").
// The minimum input arity is forced to be 1 instead of 0 because a vectorized
// function with no input is just non-sense.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY + 1 + 2, <boost/numpy/dstream/detail/caller.hpp>))
#include BOOST_PP_ITERATE()

template <class Callable>
struct caller_base_select
{
    // Note: arity includes the class type if Callable::f_t is a member function
    //       pointer.
    BOOST_STATIC_CONSTANT(unsigned, arity = boost::mpl::size<typename Callable::signature_t>::value - 1);

    typedef typename caller_arity<arity>::template impl<Callable>
            type;
};

template <class Callable>
struct caller
  : caller_base_select<Callable>::type
{
    typedef typename caller_base_select<Callable>::type
            base;

    caller(Callable ca)
      : base(ca)
    {}
};

}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_DETAIL_CALLER_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

template <>
struct invoke_arity<N>
{
    template <class RC, class Callable BOOST_PP_ENUM_TRAILING_PARAMS_Z(1, N, class AC)>
    inline
    static
    PyObject*
    invoke(RC const& rc, Callable const & ca BOOST_PP_ENUM_TRAILING_BINARY_PARAMS_Z(1, N, AC, & ac))
    {
        return rc( ca(BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, ac, () BOOST_PP_INTERCEPT)) );
    }
};

template <>
struct caller_arity<N>
{
    template <class Callable>
    struct impl
    {
        // Since we will always take Python objects as arguments and return
        // Python objects, we will always use the default_call_policies as call
        // policies.
        typedef python::default_call_policies
                call_policies_t;

        impl(Callable ca)
          : m_callable(ca)
        {}

        PyObject* operator()(PyObject* args_, PyObject*) // eliminate
                                                         // this
                                                         // trailing
                                                         // keyword dict
        {
            call_policies_t call_policies;

            typedef typename boost::mpl::begin<typename Callable::signature_t>::type
                    first;
            //typedef typename first::type result_t;
            typedef typename python::detail::select_result_converter<call_policies_t, python::object>::type
                    result_converter_t;

            typedef typename call_policies_t::argument_package
                    argument_package_t;

            // Create argument from_python converter objects c#.
            argument_package_t inner_args(args_);

            #define BOOST_NUMPY_NEXT(init, name, n)                            \
                typedef BOOST_PP_IF(n,typename boost::mpl::next< BOOST_PP_CAT(name,BOOST_PP_DEC(n)) >::type, init) name##n;

            #define BOOST_NUMPY_ARG_CONVERTER(n)                               \
                BOOST_NUMPY_NEXT(typename boost::mpl::next<first>::type, arg_iter,n) \
                typedef python::arg_from_python<BOOST_DEDUCED_TYPENAME BOOST_PP_CAT(arg_iter,n)::type> BOOST_PP_CAT(c_t,n); \
                BOOST_PP_CAT(c_t,n) BOOST_PP_CAT(c,n)(python::detail::get(boost::mpl::int_<n>(), inner_args)); \
                if(!BOOST_PP_CAT(c,n).convertible()) {                         \
                    return 0;                                                  \
                }

            #define BOOST_NUMPY_DEF(z, n, data) \
                BOOST_NUMPY_ARG_CONVERTER(n)
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
            #undef BOOST_NUMPY_DEF

            #undef BOOST_NUMPY_ARG_CONVERTER
            #undef BOOST_NUMPY_NEXT

            // All converters have been checked. Now we can do the
            // precall part of the policy.
            // Note: The precall of default_call_policies does nothing at all as
            //       of boost::python <=1.55 but we keep this call for forward
            //       compatibility and to follow the interface definition.
            if(!call_policies.precall(inner_args))
                return 0;

            PyObject* result = invoke_arity<N>::invoke(
                python::detail::create_result_converter(args_, (result_converter_t*)0, (result_converter_t*)0)
              , m_callable
                BOOST_PP_ENUM_TRAILING_PARAMS(N, c)
            );

            // Note: the postcall of default_call_policies does nothing at all
            //       as of boost::python <=1.55 but we keep this call for
            //       forward compatibility and to follow the interface
            //       definition.
            return call_policies.postcall(inner_args, result);
        }

        static unsigned min_arity() { return N; }

      private:
        Callable m_callable;
    };
};

#undef N

#endif // BOOST_PP_IS_ITERATING
