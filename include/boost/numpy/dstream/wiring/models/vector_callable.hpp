/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/wiring/models/vector_callable.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines the vector_callable template for
 *        wiring one function or a class method that returns a std::vector of
 *        values to a particular input-output mapping.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE_HPP_INCLUDED

#include <iostream>

#include <vector>

#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/detail/config.hpp>
#include <boost/numpy/detail/callable_caller.hpp>
#include <boost/numpy/dstream/wiring.hpp>
#include <boost/numpy/dstream/mapping/models/NxS_to_X.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace model {

namespace detail {

template <unsigned in_arity> struct vector_callable_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/vector_callable.hpp>, 1))
#include BOOST_PP_ITERATE()

template <
    class MappingModel
  , class Class
>
struct vector_callable
  : vector_callable_arity<MappingModel::in_arity>::template vector_callable_impl<MappingModel, Class>
{
    typedef vector_callable<MappingModel, Class>
            vector_callable_t;

    typedef vector_callable_t
            type;

    typedef typename detail::vector_callable_arity<MappingModel::in_arity>::template vector_callable_impl<MappingModel, Class>
            vector_callable_impl_t;

    typedef typename vector_callable_impl_t::base_wiring_model_t::wiring_model_config_t
            wiring_model_config_t;

    vector_callable(wiring_model_config_t const & wmc)
      : vector_callable_impl_t(wmc)
    {}
};

}/*namespace detail*/

struct vector_callable
  : wiring_model_selector_type
{
    template <
         class MappingModel
       , class Class
    >
    struct wiring_model
    {
        typedef detail::vector_callable<MappingModel, Class>
                type;
    };
};

}// namespace model
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE_HPP_INCLUDED
// EOF
//==============================================================================
#elif BOOST_PP_ITERATION_FLAGS() == 1

#define N BOOST_PP_ITERATION()

template <>
struct vector_callable_arity<N>
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
          class MappingModel
        , class Class
    >
    struct vector_callable_impl
      : base_wiring_model<MappingModel, Class, wiring_model_config_t>
    {
        typedef base_wiring_model<MappingModel, Class, wiring_model_config_t>
                base_wiring_model_t;

        //----------------------------------------------------------------------
        #define BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__in_arr_dshape_vtype(z, n, data) \
            BOOST_PP_COMMA_IF(n) typename MappingModel::in_arr_dshape_ ## n ::value_type
        typedef boost::numpy::detail::callable_caller<
              N
            , Class
            , std::vector<typename MappingModel::out_arr_dshape::value_type>
            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__in_arr_dshape_vtype, ~)
            > callable_caller_t;
        #undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__in_arr_dshape_vtype
        typedef typename callable_caller_t::callable_ptr_t callable_t;

        //______________________________________________________________________
        vector_callable_impl(wiring_model_config_t const & wmc)
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

            // Get the requested length of the 1D output array.
            intptr_t const X = MappingModel::out_arr_dshape::template shape<0>();

            //------------------------------------------------------------------
            // Do the iteration loop over the array.
            // Note: The iterator flags is set with EXTERNAL_LOOP. So each
            //       iteration is a chunk of data of size
            //       iter.get_inner_loop_size() holding the data of the inner
            //       most loop. By construction of the mapping model, the size
            //       of the inner loop is a multiple of X (the length of the
            //       second axis of the output array).
            do {
                intptr_t size = iter.get_inner_loop_size();

                // Ensure, the inner loop size is a multiple of X.
                if(size % X)
                {
                    // Note: We can't call python C-API functions here, because
                    //       we don't own the python GIL.
                    std::cerr << "The size of the inner loop is not a multiple "
                                 "of " << X << "! It is " << size << "."
                              << std::endl;
                    error_flag = true;
                    return;
                }

                while(size)
                {
                    // Get the result vector from the class method.
                    #define BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__GET_ITER_DATA_N(z, n, data) \
                        BOOST_PP_COMMA_IF(n) *reinterpret_cast<typename MappingModel:: BOOST_PP_CAT(in_arr_dshape_,n) ::value_type *>(iter.get_data( BOOST_PP_ADD(n, 1) ))
                    std::vector<typename MappingModel::out_arr_dshape::value_type> res_vec =
                        callable_t::call(
                              callable_caller.bfunc
                            , &self
                            , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__GET_ITER_DATA_N, ~)
                        );
                    #undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__GET_ITER_DATA_N

                    if(res_vec.size() != X)
                    {
                        std::cerr << "The length of the std::vector returned "
                                     "from the callable is not " << X << "! "
                                     "It is " << res_vec.size() << "."
                                  << std::endl;
                        error_flag = true;
                        return;
                    }

                    // Fill the output array with the result vector values.
                    for(intptr_t x=0; x<X; ++x)
                    {
                        *reinterpret_cast<typename MappingModel::out_arr_dshape::value_type *>(iter.get_data(0) + x*iter.get_stride(0)) = res_vec[x];
                    }

                    // Move on to the next value set.
                    iter.add_strides_to_data_ptrs(MappingModel::n_op, &MappingModel::op_value_strides[0]);
                    size -= X;
                }
            } while(iter.next());
        }
    };
};

#undef N

#endif // !BOOST_PP_IS_ITERATING
