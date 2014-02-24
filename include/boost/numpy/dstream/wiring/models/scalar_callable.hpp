/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/wiring/models/scalar_callable.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines the scalar_callable template for
 *        wiring one scalar function or scalar class member function that takes
 *        single scalar input values and returns one scalar output value.
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

#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
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

template <unsigned in_arity>
struct scalar_callable_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/scalar_callable.hpp>, 1))
#include BOOST_PP_ITERATE()

template <
    class MappingDefinition
  , class FTypes
>
struct scalar_callable
  : scalar_callable_arity<MappingDefinition::in::arity>::template scalar_callable_impl<FTypes::has_void_return, MappingDefinition, FTypes>
{
    typedef scalar_callable<MappingDefinition, FTypes>
            type;

    typedef typename scalar_callable_arity<MappingDefinition::in::arity>::template scalar_callable_impl<FTypes::has_void_return, MappingDefinition, FTypes>
            base_t;
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
//     - all input arrays of the mapping definition are scalars
//     - the output mapping consists of zero or one output array
//     - the output array is a scalar array
//     - all input function types are scalars (including bool type)
//     - the output function type is a scalar (including bool type) or is void
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
                      typename boost::mpl::bool_<MappingDefinition::maps_to_void>::type
                    , typename type_traits::is_same<typename FTypes::out_t, void>::type
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

#define N BOOST_PP_ITERATION()

template <>
struct scalar_callable_arity<N>
{
    BOOST_STATIC_CONSTANT(unsigned, in_arity = N);

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
      : wiring_model_type
    {
        template <unsigned Idx>
        struct out_arr_value_type
        {
            typedef FTypes::out_t
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

        //______________________________________________________________________
        // The call method of the wiring model does the iteration and the
        // actual wiring using the wiring model configuration.
        template <class Class, class FCaller>
        static
        void
        iterate(
              Class & self
            , FCaller const & f_caller
            , numpy::detail::iter & iter
            , std::vector< std::vector<intptr_t> > const & core_shapes
            , bool & error_flag
        )
        {
            //------------------------------------------------------------------
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
                    #define BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__GET_ITER_DATA_N(z, n, data) \
                        BOOST_PP_COMMA_IF(n) *reinterpret_cast<typename FTypes:: BOOST_PP_CAT(in_t,n) *>(iter.get_data(n))
                    FCaller::callable_ptr_t::call(
                              f_caller.bfunc
                            , &self
                            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__GET_ITER_DATA_N, ~)
                    );
                    #undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__GET_ITER_DATA_N

                    iter.add_strides_to_data_ptrs(MappingModel::n_op, &MappingModel::op_value_strides[0]);
                }
            } while(iter.next());
        }
    };

    //--------------------------------------------------------------------------
    // Partial specialization for mapping models mapping to non-void.
    template <
          class MappingModel
        , class Class
    >
    struct scalar_callable_impl<false, MappingModel, Class>
      : base_wiring_model<MappingModel, Class, wiring_model_config_t>
    {
        typedef base_wiring_model<MappingModel, Class, wiring_model_config_t>
                base_wiring_model_t;

        //----------------------------------------------------------------------
        #define BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__in_arr_dshape_vtype(z, n, data) \
            BOOST_PP_COMMA_IF(n) typename MappingModel::in_arr_dshape_ ## n ::value_type
        typedef boost::numpy::detail::callable_caller<
              N
            , Class
            , typename MappingModel::out_arr_dshape::value_type
            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__in_arr_dshape_vtype, ~)
            > callable_caller_t;
        #undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__in_arr_dshape_vtype
        typedef typename callable_caller_t::callable_ptr_t callable_t;

        //______________________________________________________________________
        scalar_callable_impl(wiring_model_config_t const & wmc)
          : base_wiring_model_t(wmc)
        {}

        //______________________________________________________________________
        // The call method of the wiring model does the iteration and the
        // actual wiring using the wiring model configuration.
        static
        void
        call(Class & self, wiring_model_config_t const & config, boost::numpy::detail::iter & iter, bool & error_flag)
        {
            //------------------------------------------------------------------
            // Get the configuration:
            //-- The pointer to the C++ class method.
            callable_caller_t callable_caller(config.get_setting(0));

            //------------------------------------------------------------------
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

                    #define BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__GET_ITER_DATA_N(z, n, data) \
                        BOOST_PP_COMMA_IF(n) *reinterpret_cast<typename MappingModel:: BOOST_PP_CAT(in_arr_dshape_,n) ::value_type *>(iter.get_data( BOOST_PP_ADD(n, 1) ))
                    *reinterpret_cast<typename MappingModel::out_arr_dshape::value_type *>(iter.get_data(0)) =
                        callable_t::call(
                              callable_caller.bfunc
                            , &self
                            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__GET_ITER_DATA_N, ~)
                        );
                    #undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__GET_ITER_DATA_N

                    iter.add_strides_to_data_ptrs(MappingModel::n_op, &MappingModel::op_value_strides[0]);
                }
            } while(iter.next());
        }
    };
};

#undef N

#endif // !BOOST_PP_IS_ITERATING
