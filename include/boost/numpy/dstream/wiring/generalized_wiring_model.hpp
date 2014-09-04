/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/generalized_wiring_model.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the generalized_wiring_model template that will be
 *        used as a default wiring model if the user does not provide a wiring
 *        model.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_WIRING_GENERALIZED_WIRING_MODEL_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_GENERALIZED_WIRING_MODEL_HPP_INCLUDED

#include <boost/mpl/bitor.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numpy/mpl/types_from_fctptr_signature.hpp>
#include <boost/numpy/dstream/wiring.hpp>
#include <boost/numpy/dstream/wiring/converter/arg_from_core_shape_data.hpp>
#include <boost/numpy/dstream/wiring/converter/arg_type_to_array_dtype.hpp>
#include <boost/numpy/dstream/wiring/converter/return_type_to_array_dtype.hpp>
#include <boost/numpy/dstream/wiring/converter/return_to_core_shape_data.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {

namespace detail {

template <class MappingDefinition, class FTypes>
struct generalized_wiring_model_api
{
    template <unsigned Idx>
    struct out_arr_value_type
    {
        typedef typename FTypes::return_type
                return_type;
        typedef typename converter::detail::return_type_to_array_dtype<typename MappingDefinition::out, return_type, Idx>::type
                type;
    };

    template <unsigned Idx>
    struct in_arr_value_type
    {
        typedef typename numpy::mpl::fct_arg_type<FTypes, Idx>::type
                arg_type;
        typedef typename converter::detail::arg_type_to_array_dtype<arg_type>::type
                type;
    };

    template <unsigned Idx>
    struct out_arr_iter_operand_flags
    {
        // This should be appropriate for 99% of all cases. But one might have
        // this as a general "flag selector" interface.
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
        // This should be appropriate for 99% of all cases. But one might have
        // this as a general "flag selector" interface.
        typedef typename numpy::detail::iter_operand::flags::READONLY
                type;
    };

    template <class LoopService>
    struct iter_flags
    {
        // If any array data type is boost::python::object, the REF_OK iterator
        // operand flag needs to be set.
        // Note: This could lead to the requirement that
        //       the python GIL cannot released during the iteration!
        typedef typename boost::mpl::if_<
                  typename LoopService::object_arrays_are_involved
                , numpy::detail::iter::flags::REFS_OK
                , numpy::detail::iter::flags::NONE
                >::type
                refs_ok_flag;

        typedef boost::mpl::bitor_<
                  typename numpy::detail::iter::flags::DONT_NEGATE_STRIDES
                , refs_ok_flag
                >
                type;
    };

    BOOST_STATIC_CONSTANT(numpy::order_t, order = numpy::KEEPORDER);

    BOOST_STATIC_CONSTANT(numpy::casting_t, casting = numpy::SAME_KIND_CASTING);

    BOOST_STATIC_CONSTANT(intptr_t, buffersize = 0);
};

template <unsigned in_arity>
struct generalized_wiring_model_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/generalized_wiring_model.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class MappingDefinition, class FTypes>
struct select_generalized_wiring_model_impl
{
    typedef typename generalized_wiring_model_arity<MappingDefinition::in::arity>::template impl<
              FTypes::has_void_return
            , MappingDefinition
            , FTypes
            >::type
            apply;
};

}// namespace detail

struct generalized_wiring_model_selector
  : wiring_model_selector_type
{
    typedef generalized_wiring_model_selector
            type;

    template <
         class MappingDefinition
       , class FTypes
    >
    struct select
    {
        typedef typename detail::select_generalized_wiring_model_impl<MappingDefinition, FTypes>::apply
                type;
    };
};

}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_GENERALIZED_WIRING_MODEL_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define IN_ARITY BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_DEF__in_arr_value(z, n, data) \
    BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(arg_converter_t,n)::apply(iter, MappingDefinition::out::arity + n, in_core_shapes[n])

template <>
struct generalized_wiring_model_arity<IN_ARITY>
{
    template <
          bool fct_has_void_return
        , class MappingDefinition
        , class FTypes
    >
    struct impl;

    //--------------------------------------------------------------------------
    // Partial specialization for functions returning void.
    template <class MappingDefinition, class FTypes>
    struct impl<true, MappingDefinition, FTypes>
      : wiring_model_base<MappingDefinition, FTypes>
    {
        typedef impl<true, MappingDefinition, FTypes>
                type;

        typedef generalized_wiring_model_api<MappingDefinition, FTypes>
                api;

        // Define the arg_from_core_shape_data converter types for all the
        // input arguments.
        #define BOOST_NUMPY_DSTREAM_DEF(z, n, data)                            \
            typedef typename converter::detail::arg_from_core_shape_data_converter< \
                      typename numpy::mpl::fct_arg_type<FTypes, n>::type       \
                    , typename api::template in_arr_value_type<n>::type        \
                    >::type                                                    \
                    BOOST_PP_CAT(arg_converter_t,n);
        BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DEF, ~)
        #undef BOOST_NUMPY_DSTREAM_DEF

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
            , std::vector< std::vector<intptr_t> > const & out_core_shapes
            , std::vector< std::vector<intptr_t> > const & in_core_shapes
            , bool & error_flag
        )
        {
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    std::cout << "Calling f from generalized:";

                    f_caller.call(
                        self
                      , BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DEF__in_arr_value, ~)
                    );

                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for functions returning non-void.
    template <class MappingDefinition, class FTypes>
    struct impl<false, MappingDefinition, FTypes>
      : wiring_model_base<MappingDefinition, FTypes>
    {
        typedef impl<false, MappingDefinition, FTypes>
                type;

        typedef generalized_wiring_model_api<MappingDefinition, FTypes>
                api;

        // Define the arg_from_core_shape_data converter types for all the
        // input arguments.
        #define BOOST_NUMPY_DSTREAM_DEF(z, n, data)                            \
            typedef typename converter::detail::arg_from_core_shape_data_converter< \
                      typename numpy::mpl::fct_arg_type<FTypes, n>::type       \
                    , typename api::template in_arr_value_type<n>::type        \
                    >::type                                                    \
                    BOOST_PP_CAT(arg_converter_t,n);
        BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DEF, ~)
        #undef BOOST_NUMPY_DSTREAM_DEF

        // Define the return value converter type, that will be used to transfer
        // the function's return data into the output arrays.
        typedef typename converter::detail::return_to_core_shape_data_converter<
                  typename MappingDefinition::out
                , typename FTypes::return_type
                >::type
                return_to_core_shape_data_t;

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
            , std::vector< std::vector<intptr_t> > const & out_core_shapes
            , std::vector< std::vector<intptr_t> > const & in_core_shapes
            , bool & error_flag
        )
        {
            do {
                intptr_t size = iter.get_inner_loop_size();
                while(size--)
                {
                    if(! return_to_core_shape_data_t::template apply<api>(
                          f_caller.call(
                                self
                              , BOOST_PP_REPEAT(IN_ARITY, BOOST_NUMPY_DSTREAM_DEF__in_arr_value, ~)
                          )
                        , iter
                        , out_core_shapes
                    ))
                    {
                        error_flag = true;
                        return;
                    }

                    iter.add_inner_loop_strides_to_data_ptrs();
                }
            } while(iter.next());
        }
    };
};

#undef BOOST_NUMPY_DSTREAM_DEF__in_arr_value

#undef IN_ARITY

#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
