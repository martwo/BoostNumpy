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
#if !BOOST_PP_IS_ITERATING
#ifndef BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE_HPP_INCLUDED

#include <iostream>

#include <vector>

#include <boost/assert.hpp>
#include <boost/python.hpp>
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
#include <boost/numpy/detail/pygil.hpp>
#include <boost/numpy/dstream/wiring.hpp>
#include <boost/numpy/dstream/mapping/models/_NxS_to_X.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace model {

template <
    class MappingModel
  , class Class
  , class ConfigID
  , typename OutT
  , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_, = BOOST_NUMPY_PP_MPL_VOID BOOST_PP_INTERCEPT)
>
struct vector_callable
  : base_wiring_model<MappingModel>
{};

//------------------------------------------------------------------------------
// Partial template specializations for the model_NxS_to_X mapping models.
#define BOOST_PP_ITERATION_PARAMS_1 (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/vector_callable.hpp>, 0))
#include BOOST_PP_ITERATE()

}// namespace model
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE_HPP_INCLUDED
// EOF
//==============================================================================
#elif BOOST_PP_ITERATION_DEPTH() == 1 \
   && BOOST_PP_ITERATION_FLAGS() == 0

#define BOOST_PP_ITERATION_PARAMS_2 (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/wiring/models/vector_callable.hpp>, 0))
#include BOOST_PP_ITERATE()

#elif BOOST_PP_ITERATION_DEPTH() == 2 \
   && BOOST_PP_ITERATION_FLAGS() == 0

#define N BOOST_PP_FRAME_ITERATION(1)
#define X BOOST_PP_FRAME_ITERATION(2)

#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__GET_ITER_DATA_N(z, n, data) \
    BOOST_PP_COMMA_IF(n) *reinterpret_cast<BOOST_PP_CAT(InT_, n) *>(iter.get_data( BOOST_PP_ADD(n, 1) ))

#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__SET_OUT_ARR_VALUE_X(z, x, data) \
    *reinterpret_cast<OutT*>(iter.get_data(0) + x*iter.get_stride(0)) = res_vec[x];

#define BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__COMMA_X(z, n, x) \
    BOOST_PP_COMMA() x

template <
    class Class
  , class ConfigID
  , typename OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_)
>
struct vector_callable<
    BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(mapping::model::_, N), xS_to_), X)
  , Class
  , ConfigID
  , OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
> : base_wiring_model<BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(mapping::model::_, N), xS_to_), X)>
{
    //--------------------------------------------------------------------------
    // Define the required settings type (i.e. one setting) and the
    // corresponding configuration type.
    typedef ConfigID config_id_t;
    typedef boost::numpy::detail::settings<1>::type wiring_model_settings_t;
    typedef typename boost::numpy::detail::config<wiring_model_settings_t, config_id_t>::type config_t;

    //--------------------------------------------------------------------------
    // Define the required types for calling the class method.
    typedef boost::numpy::detail::callable_caller<
        mapping_model_t::in_arity
      , Class
      , std::vector<OutT>
      , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
      > callable_caller_t;
    typedef typename callable_caller_t::callable_ptr_t callable_t;

    //--------------------------------------------------------------------------
    // Declare the number of value jumps for each array for a jump to the next
    // set of data values.
    static int const op_value_jumps[BOOST_PP_ADD(1, N)];

    //__________________________________________________________________________
    // The call method of the wiring model does the iteration and the
    // actual wiring using the wiring model configuration.
    static
    void
    call(Class & self, config_t const & cfg, boost::numpy::detail::iter & iter, bool & error_flag)
    {
        //----------------------------------------------------------------------
        // Get the configuration:
        //-- The pointer to the C++ class method.
        callable_caller_t callable_caller(cfg.get_setting(0));

        //----------------------------------------------------------------------
        // Do the iteration loop over the array.
        // Note: The iterator flags is set with EXTERNAL_LOOP. So each iteration
        //       is a chunk of data of size iter.get_inner_loop_size() holding
        //       the data of the inner most loop. By construction of the mapping
        //       model, the size of the inner loop is a multiple of X (the
        //       length of the second axis of the output array).
        do {
            intptr_t size = iter.get_inner_loop_size();

            // Ensure, the inner loop size is a multiple of X.
            if(size % X)
            {
                // Note: We can't call python C-API functions here, because we
                //       don't own the python GIL.
                std::cerr << "The size of the inner loop is not a multiple of "
                             BOOST_PP_STRINGIZE(X) "!"
                          << std::endl;
                error_flag = true;
                return;
            }

            while(size)
            {
                // Get the result vector from the class method.
                std::vector<OutT> res_vec =
                    callable_t::call(
                          callable_caller.bfunc
                        , &self
                        , BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__GET_ITER_DATA_N, ~)
                    );
                if(res_vec.size() != X)
                {
                    std::cerr << "The length of the std::vector returned from "
                                 "the callable is not " BOOST_PP_STRINGIZE(X) "!"
                              << std::endl;
                    error_flag = true;
                    return;
                }
                // Fill the output array with the result vector values.
                BOOST_PP_REPEAT(X, BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__SET_OUT_ARR_VALUE_X, ~)

                // Move on to the next value set.
                iter.add_strides_to_data_ptrs(mapping_model_t::n_op, op_value_jumps);
                size -= X;
            }
        } while(iter.next());
    }
};

//______________________________________________________________________________
// Define the values of the op_value_jumps static constant array.
template <
    class Class
  , class ConfigID
  , typename OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, typename InT_)
>
int const
vector_callable<
    BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(mapping::model::_, N), xS_to_), X)
  , Class
  , ConfigID
  , OutT
  , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)
>
::op_value_jumps[BOOST_PP_ADD(1, N)]
=
{X BOOST_PP_REPEAT(N, BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__COMMA_X, X)};

#undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__COMMA_ONE
#undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__SET_OUT_ARR_VALUE_X
#undef BOOST_NUMPY_DSTREAM_WIRING_MODEL_VECTOR_CALLABLE__GET_ITER_DATA_N

#undef X
#undef N

#endif // !BOOST_PP_IS_ITERATING
