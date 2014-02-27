/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/models/scalar_callable.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the scalar_callable template for
 *        wiring one scalar function or scalar class member function that takes
 *        single scalar input values and returns one scalar output value or
 *        void.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE_HPP_INCLUDED

#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/facilities/identity.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/bitor.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/or.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/types_from_fctptr_signature.hpp>
#include <boost/numpy/detail/callable_caller.hpp>
#include <boost/numpy/dstream/mapping/detail/definition.hpp>
#include <boost/numpy/dstream/wiring.hpp>
#include <boost/numpy/dstream/wiring/default_wiring_model_selector_fwd.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace model {

namespace detail {

template <class MappingDefinition, class FTypes>
struct scalar_callable_api
{
    template <unsigned Idx>
    struct out_arr_value_type
    {
        typedef typename FTypes::return_type
                type;
    };

    template <unsigned Idx>
    struct in_arr_value_type
    {
        typedef typename boost::mpl::at< typename FTypes::in_type_vector, boost::mpl::long_<Idx> >::type
                type;
    };

    template <unsigned Idx>
    struct out_arr_iter_operand_flags
    {
        typedef boost::mpl::bitor_<
                    typename numpy::detail::iter_operand::flags::WRITEONLY
                    , typename numpy::detail::iter_operand::flags::NBO
                    , typename numpy::detail::iter_operand::flags::ALIGNED
                >
                type;
    };

    template <unsigned Idx>
    struct in_arr_iter_operand_flags
    {
        typedef typename numpy::detail::iter_operand::flags::READONLY
                type;
    };

    struct iter_flags
    {
        typedef typename numpy::detail::iter::flags::DONT_NEGATE_STRIDES
                type;
    };

    BOOST_STATIC_CONSTANT(order_t, order = numpy::KEEPORDER);

    BOOST_STATIC_CONSTANT(casting_t, casting = numpy::SAME_KIND_CASTING);

    BOOST_STATIC_CONSTANT(intptr_t, buffersize = 0);
};

template <unsigned in_arity>
struct scalar_callable_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/scalar_callable.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class MappingDefinition, class FTypes>
struct scalar_callable_impl_select
{
    typedef typename scalar_callable_arity<MappingDefinition::in::arity>::template impl<
                  FTypes::has_void_return
                , MappingDefinition
                , FTypes
                >::type
            type;
};

template <class MappingDefinition, class FTypes>
struct scalar_callable
  : scalar_callable_impl_select<MappingDefinition, FTypes>::type
{
    typedef scalar_callable<MappingDefinition, FTypes>
            type;
};

}// namespace detail

struct scalar_callable
  : wiring_model_selector_type
{
    template <
         class MappingDefinition
       , class FTypes
    >
    struct select
    {
        typedef detail::scalar_callable<MappingDefinition, FTypes>
                type;
    };
};

}// namespace model


// The scalar_callable wiring model should be selected if all of the following
// conditions on the mapping definition and function types are true:
//     - All input arrays of the mapping definition are scalars.
//     - All function argument types must be a scalar or bool type.
//     - The output mapping consists of zero or one output array.
//       - If the output mapping has zero output arrays, the function's return
//         type must be void.
//       - If the output mapping has one output array, the output array must be
//         a scalar array and the function's return type must be a scalar or
//         bool type.
template <class MappingDefinition, class FTypes>
struct default_wiring_model_selector<
      MappingDefinition
    , FTypes
    , typename enable_if<
          typename boost::mpl::and_<
              typename dstream::mapping::detail::in_mapping<typename MappingDefinition::in>::all_arrays_are_scalars::type
            , typename numpy::mpl::all_fct_args_are_scalars_incl_bool<FTypes>::type
            , typename boost::mpl::or_<
                  typename boost::mpl::and_<
                      typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template arity_is<0>::type
                    , typename boost::mpl::bool_<FTypes::has_void_return>::type
                  >::type
                , typename boost::mpl::and_<
                      typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template arity_is<1>::type
                    , typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template array<0>::is_scalar::type
                    , typename numpy::mpl::fct_return_is_scalar_or_bool<FTypes>::type
                  >::type
              >::type
          >::type
      >::type
>
{
    typedef model::scalar_callable
            type;
};

}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE_HPP_INCLUDED
// EOF
//==============================================================================
#elif BOOST_PP_ITERATION_FLAGS() == 1

#define IN_ARITY BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__in_arr_value(z, n, data) \
    BOOST_PP_COMMA_IF(n) typename FTypes:: BOOST_PP_CAT(arg_type,n)(BOOST_PP_CAT(in_arr_value,n))

template <>
struct scalar_callable_arity<IN_ARITY>
{
    BOOST_STATIC_CONSTANT(unsigned, in_arity = IN_ARITY);

    template <
          bool fct_has_void_return
        , class MappingDefinition
        , class FTypes
    >
    struct impl;

    //--------------------------------------------------------------------------
    // Partial specialization for functions returning void.
    // It implements the wiring for mapping Nx() -> None
    template <class MappingDefinition, class FTypes>
    struct impl<true, MappingDefinition, FTypes>
      : wiring_model_base<MappingDefinition, FTypes>
    {
        typedef impl<true, MappingDefinition, FTypes>
                type;

        typedef scalar_callable_api<MappingDefinition, FTypes>
                api;

        // Define the input array value types.
        #define BOOST_PP_LOCAL_MACRO(n) \
            typedef typename api::template in_arr_value_type<n>::type BOOST_PP_CAT(in_arr_value_t,n);
        #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(IN_ARITY,1))
        #include BOOST_PP_LOCAL_ITERATE()

        /** The iterate method of the wiring model does the iteration and the
         *  actual wiring.
         */
        template <class ClassT, class FCaller>
        static
        void
        iterate(
              ClassT & self
            , FCaller const & f_caller
            , numpy::detail::iter & iter
            , std::vector< std::vector<intptr_t> > const & core_shapes
            , bool & error_flag
        )
        {
            // Do the iteration loop over the array.
            // Note: The iterator flags is set with EXTERNAL_LOOP in order
            //       to allow for multi-threading. So each iteration is a
            //       chunk of size iter.get_inner_loop_size() of data, that
            //       needs to be iterated manually. The inner loop size is
            //       strongly related to the buffer size specified at
            //       iterator construction.
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Construct references to the input array values.
                    #define BOOST_PP_LOCAL_MACRO(n) \
                        BOOST_PP_CAT(in_arr_value_t,n) & BOOST_PP_CAT(in_arr_value,n) =\
                            *reinterpret_cast<BOOST_PP_CAT(in_arr_value_t,n) *>(iter.get_data(MappingDefinition::out::arity + n));
                    #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(IN_ARITY,1))
                    #include BOOST_PP_LOCAL_ITERATE()

                    // Call the scalar function (i.e. the to-be-exposed
                    // function) and implicitly convert between types, in case
                    // function types differ from array value types.
                    f_caller.call(
                          self
                        , BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__in_arr_value, ~)
                    );

                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for functions returning non-void.
    // It implements the wiring for mapping Nx() -> ()
    template <class MappingDefinition, class FTypes>
    struct impl<false, MappingDefinition, FTypes>
      : wiring_model_base<MappingDefinition, FTypes>
    {
        typedef impl<false, MappingDefinition, FTypes>
                type;

        typedef scalar_callable_api<MappingDefinition, FTypes>
                api;

        // Define the output and input array value types.
        typedef typename api::template out_arr_value_type<0>::type
                out_arr_value_t;
        #define BOOST_PP_LOCAL_MACRO(n) \
            typedef typename api::template in_arr_value_type<n>::type BOOST_PP_CAT(in_arr_value_t,n);
        #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(IN_ARITY,1))
        #include BOOST_PP_LOCAL_ITERATE()

        template <class ClassT, class FCaller>
        static
        void
        iterate(
              ClassT & self
            , FCaller const & f_caller
            , numpy::detail::iter & iter
            , std::vector< std::vector<intptr_t> > const & core_shapes
            , bool & error_flag
        )
        {
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Construct references to the output and input array
                    // values.
                    out_arr_value_t & out_arr_value = *reinterpret_cast<out_arr_value_t *>(iter.get_data(0));
                    #define BOOST_PP_LOCAL_MACRO(n) \
                        BOOST_PP_CAT(in_arr_value_t,n) & BOOST_PP_CAT(in_arr_value,n) =\
                            *reinterpret_cast<BOOST_PP_CAT(in_arr_value_t,n) *>(iter.get_data(MappingDefinition::out::arity + n));
                    #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(IN_ARITY,1))
                    #include BOOST_PP_LOCAL_ITERATE()

                    // Call the scalar function (i.e. the to-be-exposed
                    // function) and implicitly convert between types, in case
                    // function types differ from array value types.
                    typename FTypes::return_type ret = f_caller.call(
                          self
                        , BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__in_arr_value, ~)
                    );
                    out_arr_value = out_arr_value_t(ret);

                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    };
};

#undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__in_arr_value

#undef IN_ARITY

#endif // !BOOST_PP_IS_ITERATING
