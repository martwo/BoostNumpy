/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/models/scalars_to_vector_of_scalar_callable.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the scalars_to_vector_of_scalar_callable template
 *        for wiring a function or class member function that takes
 *        scalar input values and returns a std::vector of a scalar type to a
 *        mapping that has only scalar inputs and has either one 1D array with N
 *        elements, or N scalar arrays as output.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALARS_TO_VECTOR_OF_SCALAR_CALLABLE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALARS_TO_VECTOR_OF_SCALAR_CALLABLE_HPP_INCLUDED

#include <vector>

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/bitor.hpp>
#include <boost/mpl/long.hpp>
#include <boost/mpl/or.hpp>

#include <boost/utility/enable_if.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/types_from_fctptr_signature.hpp>
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
struct scalars_to_vector_of_scalar_callable_api
{
    template <unsigned Idx>
    struct out_arr_value_type
    {
        typedef typename FTypes::return_type::value_type
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

    template <class LoopService>
    struct iter_flags
    {
        typedef typename numpy::detail::iter::flags::DONT_NEGATE_STRIDES
                type;
    };

    BOOST_STATIC_CONSTANT(order_t, order = numpy::KEEPORDER);

    BOOST_STATIC_CONSTANT(casting_t, casting = numpy::SAME_KIND_CASTING);

    BOOST_STATIC_CONSTANT(intptr_t, buffersize = 0);
};

template <unsigned out_arity, unsigned in_arity>
struct scalars_to_vector_of_scalar_callable_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_OUTPUT_ARITY, <boost/numpy/dstream/wiring/models/scalars_to_vector_of_scalar_callable.hpp>))
#include BOOST_PP_ITERATE()

template <bool is_single_output, class MappingDefinition>
struct n_out_values;

template <class MappingDefinition>
struct n_out_values<true, MappingDefinition>
{
    typedef boost::mpl::integral_c<unsigned, MappingDefinition::out::core_shape_t0::dim0>
            type;
};

template <class MappingDefinition>
struct n_out_values<false, MappingDefinition>
{
    typedef boost::mpl::integral_c<unsigned, MappingDefinition::out::arity>
            type;
};

template <class MappingDefinition, class FTypes>
struct scalars_to_vector_of_scalar_callable_impl_select
{
    typedef typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template arity_is_equal_to<1>::type
            is_single_output_t;

    typedef typename n_out_values<is_single_output_t::value, MappingDefinition>::type
            n_out_values_t;

    typedef typename scalars_to_vector_of_scalar_callable_arity<n_out_values_t::value, MappingDefinition::in::arity>::template impl<
                  is_single_output_t::value
                , MappingDefinition
                , FTypes
                >::type
            type;
};

}// namespace detail

struct scalars_to_vector_of_scalar_callable
  : wiring_model_selector_type
{
    template <
         class MappingDefinition
       , class FTypes
    >
    struct select
    {
        typedef typename detail::scalars_to_vector_of_scalar_callable_impl_select<MappingDefinition, FTypes>::type
                type;
    };
};

}// namespace model

template <class MappingDefinition, class FTypes>
struct default_wiring_model_selector<
      MappingDefinition
    , FTypes
    , typename enable_if<
          typename boost::mpl::and_<
              typename dstream::mapping::detail::in_mapping<typename MappingDefinition::in>::all_arrays_are_scalars::type
            , typename numpy::mpl::all_fct_args_are_scalars<FTypes>::type
            , typename numpy::mpl::fct_return_is_std_vector_of_scalar<FTypes>::type
            , typename boost::mpl::or_<
                  typename boost::mpl::and_<
                      typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template arity_is_equal_to<1>::type
                    , typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template array<0>::is_1d::type
                  >::type
                , typename boost::mpl::and_<
                      typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template arity_is_greater<1>::type
                    , typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::all_arrays_are_scalars::type
                  >::type
              >::type
          >::type
      >::type
>
{
    typedef model::scalars_to_vector_of_scalar_callable
            type;
};

}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALARS_TO_VECTOR_OF_SCALAR_CALLABLE_HPP_INCLUDED
// EOF
//==============================================================================
#else

#if BOOST_PP_ITERATION_DEPTH() == 1

// Loop over the InArity.
#define BOOST_PP_ITERATION_PARAMS_2 \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/scalars_to_vector_of_scalar_callable.hpp>))
#include BOOST_PP_ITERATE()

#else
#if BOOST_PP_ITERATION_DEPTH() == 2

#define OUT_ARITY BOOST_PP_RELATIVE_ITERATION(1)
#define IN_ARITY BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_WIRING_MODEL__in_arr_value(z, n, data) \
    BOOST_PP_COMMA_IF(n) typename FTypes:: BOOST_PP_CAT(arg_type,n)(BOOST_PP_CAT(in_arr_value,n))

template <>
struct scalars_to_vector_of_scalar_callable_arity<OUT_ARITY, IN_ARITY>
{
    BOOST_STATIC_CONSTANT(unsigned, out_arity = OUT_ARITY);
    BOOST_STATIC_CONSTANT(unsigned, in_arity = IN_ARITY);

    template <
          bool is_single_output
        , class MappingDefinition
        , class FTypes
    >
    struct impl;

    //--------------------------------------------------------------------------
    // Partial specialization for single output, where the output is a 1d
    // array.
    template <
          class MappingDefinition
        , class FTypes
    >
    struct impl<true, MappingDefinition, FTypes>
    {
        typedef impl<true, MappingDefinition, FTypes>
                type;

        typedef scalars_to_vector_of_scalar_callable_api<MappingDefinition, FTypes>
                api;

        typedef typename api::template out_arr_value_type<0>::type
                out_arr_value_t;
        #define BOOST_NUMPY_DEF(z, n, data) \
            typedef typename api::template in_arr_value_type<n>::type \
                    BOOST_PP_CAT(in_arr_value_t,n);
        BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF

        template <class ClassT, class FCaller>
        static
        void
        iterate(
              ClassT & self
            , FCaller const & f_caller
            , numpy::detail::iter & iter
            , std::vector< std::vector<intptr_t> > const & out_core_shapes
            , std::vector< std::vector<intptr_t> > const & in_core_shapes
            , bool & error_flag
        )
        {
            do {
                intptr_t const out_op_value_stride = iter.get_item_size(0);
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Construct references to the output array values.
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        out_arr_value_t & BOOST_PP_CAT(out_arr_value,n) =\
                            *reinterpret_cast<out_arr_value_t *>(iter.get_data(0) + n * out_op_value_stride);
                    BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DEF, ~)
                    #undef BOOST_NUMPY_DEF

                    // Construct references to the input array values.
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        BOOST_PP_CAT(in_arr_value_t,n) & BOOST_PP_CAT(in_arr_value,n) =\
                            *reinterpret_cast<BOOST_PP_CAT(in_arr_value_t,n) *>(iter.get_data(MappingDefinition::out::arity + n));
                    BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DEF, ~)
                    #undef BOOST_NUMPY_DEF

                    // Call the to-be-exposed function and implicitly convert
                    // between types, in case function types differ from array
                    // value types.
                    typename FTypes::return_type ret = f_caller.call(
                          self
                        , BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_WIRING_MODEL__in_arr_value, ~)
                    );

                    if(ret.size() != OUT_ARITY)
                    {
                        std::cerr << "Function returned std::vector of wrong "
                                  << "size! Must be " << OUT_ARITY << ", but "
                                  << "was " << ret.size() << "!"
                                  << std::endl;
                        error_flag = true;
                        return;
                    }

                    // Set the return values to the output array.
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        BOOST_PP_CAT(out_arr_value,n) = out_arr_value_t(ret[n]);
                    BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DEF, ~)
                    #undef BOOST_NUMPY_DEF

                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    };

    // Specialization for multiple scalar output arrays.
    template <
          class MappingDefinition
        , class FTypes
    >
    struct impl<false, MappingDefinition, FTypes>
    {
        typedef impl<false, MappingDefinition, FTypes>
                type;

        typedef scalars_to_vector_of_scalar_callable_api<MappingDefinition, FTypes>
                api;

        #define BOOST_NUMPY_DEF(z, n, data) \
            typedef typename api::template out_arr_value_type<n>::type \
                    BOOST_PP_CAT(out_arr_value_t,n);
        BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF

        #define BOOST_NUMPY_DEF(z, n, data) \
            typedef typename api::template in_arr_value_type<n>::type \
                    BOOST_PP_CAT(in_arr_value_t,n);
        BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF

        template <class ClassT, class FCaller>
        static
        void
        iterate(
              ClassT & self
            , FCaller const & f_caller
            , numpy::detail::iter & iter
            , std::vector< std::vector<intptr_t> > const & out_core_shapes
            , std::vector< std::vector<intptr_t> > const & in_core_shapes
            , bool & error_flag
        )
        {
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    // Construct references to the output array values.
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        BOOST_PP_CAT(out_arr_value_t,n) & BOOST_PP_CAT(out_arr_value,n) =\
                            *reinterpret_cast<BOOST_PP_CAT(out_arr_value_t,n) *>(iter.get_data(n));
                    BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DEF, ~)
                    #undef BOOST_NUMPY_DEF

                    // Construct references to the input array values.
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        BOOST_PP_CAT(in_arr_value_t,n) & BOOST_PP_CAT(in_arr_value,n) =\
                            *reinterpret_cast<BOOST_PP_CAT(in_arr_value_t,n) *>(iter.get_data(MappingDefinition::out::arity + n));
                    BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DEF, ~)
                    #undef BOOST_NUMPY_DEF

                    // Call the to-be-exposed function and implicitly convert
                    // between types, in case function types differ from array
                    // value types.
                    typename FTypes::return_type ret = f_caller.call(
                          self
                        , BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_WIRING_MODEL__in_arr_value, ~)
                    );

                    if(ret.size() != OUT_ARITY)
                    {
                        std::cerr << "Function returned std::vector of wrong "
                                  << "size! Must be " << OUT_ARITY << ", but "
                                  << "was " << ret.size() << "!"
                                  << std::endl;
                        error_flag = true;
                        return;
                    }

                    // Set the return values to the output array.
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        BOOST_PP_CAT(out_arr_value,n) = BOOST_PP_CAT(out_arr_value_t,n)(ret[n]);
                    BOOST_PP_REPEAT(OUT_ARITY, BOOST_NUMPY_DEF, ~)
                    #undef BOOST_NUMPY_DEF

                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    };
};

#undef BOOST_NUMPY_DSTREAM_WIRING_MODEL__in_arr_value

#undef IN_ARITY
#undef OUT_ARITY

#endif // BOOST_PP_ITERATION_DEPTH == 2
#endif // BOOST_PP_ITERATION_DEPTH == 1

#endif // BOOST_PP_IS_ITERATING
