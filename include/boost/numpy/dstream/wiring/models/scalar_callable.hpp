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

#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/detail/config.hpp>
#include <boost/numpy/detail/callable_caller.hpp>
#include <boost/numpy/dstream/wiring.hpp>
#include <boost/numpy/dstream/mapping/models/_NxS_to_S.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace model {

namespace detail {

template <unsigned in_arity> struct scalar_callable_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/scalar_callable.hpp>, 1))
#include BOOST_PP_ITERATE()

}/*namespace detail*/

//==============================================================================
template <
    class MappingModel
  , class Class
>
struct scalar_callable
  : detail::scalar_callable_arity<MappingModel::in_arity>::template scalar_callable_impl<MappingModel::maps_to_void, MappingModel, Class>
{
    typedef scalar_callable<MappingModel, Class>
            scalar_callable_t;

    typedef scalar_callable_t
            type;

    typedef typename detail::scalar_callable_arity<MappingModel::in_arity>::template scalar_callable_impl<MappingModel::maps_to_void, MappingModel, Class>
            scalar_callable_impl_t;

    typedef typename scalar_callable_impl_t::base_wiring_model_t::wiring_model_config_t
            wiring_model_config_t;

    scalar_callable(wiring_model_config_t const & wmc)
      : scalar_callable_impl_t(wmc)
    {}
};

struct scalar_callable_selector
  : wiring_model_selector_type
{
    template <
         class MappingModel
       , class Class
    >
    struct wiring_model
    {
        typedef scalar_callable<MappingModel, Class>
                type;
    };
};

}// namespace model
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE_HPP_INCLUDED
// EOF
//==============================================================================
// Partial template specializations for the model_NxS_to_S mapping models.
#elif BOOST_PP_ITERATION_FLAGS() == 1

#define N BOOST_PP_ITERATION()

template <>
struct scalar_callable_arity<N>
{
    BOOST_STATIC_CONSTANT(unsigned, in_arity = N);

    //----------------------------------------------------------------------
    // Define the required settings type (i.e. one setting) and the
    // corresponding configuration type.
    typedef boost::numpy::detail::settings<1>::type
            wiring_model_settings_t;
    typedef typename boost::numpy::detail::config<wiring_model_settings_t>::type
            wiring_model_config_t;

    template <
          bool maps_to_void
        , class MappingModel
        , class Class
    >
    struct scalar_callable_impl;

    //--------------------------------------------------------------------------
    // Partial specialization for mapping models mapping to void.
    template <
          class MappingModel
        , class Class
    >
    struct scalar_callable_impl<true, MappingModel, Class>
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
            , void
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
                        BOOST_PP_COMMA_IF(n) *reinterpret_cast<typename MappingModel:: BOOST_PP_CAT(in_arr_dshape_,n) ::value_type *>(iter.get_data(n))
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

#undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_SCALAR_CALLABLE__GET_ITER_DATA_N

#undef N

#endif // !BOOST_PP_IS_ITERATING
