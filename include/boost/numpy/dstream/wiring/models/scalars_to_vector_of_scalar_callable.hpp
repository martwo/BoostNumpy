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
struct scalars_to_vector_of_scalar_callable_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/scalars_to_vector_of_scalar_callable.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class MappingDefinition, class FTypes>
struct scalars_to_vector_of_scalar_callable_impl_select
{
    typedef typename dstream::mapping::detail::out_mapping<typename MappingDefinition::out>::template arity_is_equal_to<1>::type
            is_single_output_t;

    typedef typename boost::mpl::if_<
                is_single_output_t
              , boost::mpl::integral_c<unsigned, 0>
              , boost::mpl::integral_c<unsigned, 1>
            >::type
            output_style_t;

    typedef typename scalars_to_vector_of_scalar_callable_arity<MappingDefinition::in::arity>::template impl<
                  output_style_t::value
                , MappingDefinition
                , FTypes
                >::type
            type;
};

template <class MappingDefinition, class FTypes>
struct scalars_to_vector_of_scalar_callable
  : scalars_to_vector_of_scalar_callable_impl_select<MappingDefinition, FTypes>::type
{
    typedef scalars_to_vector_of_scalar_callable<MappingDefinition, FTypes>
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
        typedef detail::scalars_to_vector_of_scalar_callable<MappingDefinition, FTypes>
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
#elif BOOST_PP_ITERATION_FLAGS() == 1

#define IN_ARITY BOOST_PP_ITERATION()

template <>
struct scalars_to_vector_of_scalar_callable_arity<IN_ARITY>
{
    BOOST_STATIC_CONSTANT(unsigned, in_arity = IN_ARITY);

    template <
          int output_style
        , class MappingDefinition
        , class FTypes
    >
    struct impl;

    //--------------------------------------------------------------------------
    // Partial specialization for an output style, where the output is a 1d
    // array.
    template <
          class MappingDefinition
        , class FTypes
    >
    struct impl<0, MappingDefinition, FTypes>
    {
        typedef impl<0, MappingDefinition, FTypes>
                type;

        typedef scalars_to_vector_of_scalar_callable_api<MappingDefinition, FTypes>
                api;
    };
};

#undef IN_ARITY

#endif // BOOST_PP_IS_ITERATING
