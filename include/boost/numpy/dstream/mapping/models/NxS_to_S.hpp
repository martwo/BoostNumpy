/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/mapping/models/NxS_to_S.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines a mapping model template for functions that expact
 *        only scalar data shaped input arrays and that are returning a scalar
 *        shaped output array.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S_HPP_INCLUDED

#include <stdint.h>

#include <vector>

#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>

#include <boost/type_traits/is_same.hpp>

#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/types.hpp>
#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/dstream/mapping.hpp>
#include <boost/numpy/dstream/dshape.hpp>

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__ITER_FLAGS
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__ITER_FLAGS                 \
    boost::numpy::detail::iter::DONT_NEGATE_STRIDES
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__ORDER
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__ORDER                      \
    KEEPORDER
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__CASTING
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__CASTING                    \
    SAME_KIND_CASTING
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__BUFFERSIZE
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__BUFFERSIZE                 \
    0
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__OUT_ARR_ITER_OPERAND_FLAGS
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__OUT_ARR_ITER_OPERAND_FLAGS \
    boost::numpy::detail::iter_operand::WRITEONLY                              \
    | boost::numpy::detail::iter_operand::NBO                                  \
    | boost::numpy::detail::iter_operand::ALIGNED
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__IN_ARR_ITER_OPERAND_FLAGS
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__IN_ARR_ITER_OPERAND_FLAGS  \
    boost::numpy::detail::iter_operand::READONLY
#endif

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace model {

#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__INDSHAPE(z, n, data)       \
    BOOST_PP_COMMA_IF(n) scalar_dshape< BOOST_PP_CAT(InT_, n) >

template <
      int InArity
    , class OutT
    , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, class InT_, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
struct NxS_to_S
  : base_mapping_model<
          InArity
        , scalar_dshape<OutT>
        , BOOST_PP_REPEAT(BOOST_NUMPY_LIMIT_INPUT_ARITY, BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__INDSHAPE, ~)
    >
{
    typedef boost::is_same<OutT, void> maps_to_void_t;
    BOOST_STATIC_CONSTANT(bool, maps_to_void = maps_to_void_t::value);

    //--------------------------------------------------------------------------
    // Declare the iterator configuration needed for this mapping model as
    // static constants.
    static boost::numpy::detail::iter_flags_t const iter_flags;
    static order_t                            const order;
    static casting_t                          const casting;
    static intptr_t                           const buffersize;
    BOOST_STATIC_CONSTANT(int, n_op = (!maps_to_void) + InArity);
    BOOST_STATIC_CONSTANT(int, n_iter_axes = 1);

    //--------------------------------------------------------------------------
    // Declare the strides between the values of each array operand.
    static std::vector<int> const op_value_strides;

    //--------------------------------------------------------------------------
    // Declare the iter operand flags for the output array.
    static boost::numpy::detail::iter_operand_flags_t const out_arr_iter_operand_flags;

    //--------------------------------------------------------------------------
    // Declare the iter operand flags for all the input arrays.
    template <int idx>
    struct in_arr_iter_operand_flags
    {
        static boost::numpy::detail::iter_operand_flags_t const value;
    };

    //__________________________________________________________________________
    /**
     * \brief Sets the iteration shape to the given std::vector<intptr_t> object
     *     for an iteration where the first axis has n_axis_1_elements
     *     elements.
     */
    static void
    set_itershape(std::vector<intptr_t> & itershape, intptr_t n_axis_1_elements)
    {
        itershape.resize(n_iter_axes);
        itershape[0] = n_axis_1_elements;
    }

    //__________________________________________________________________________
    /**
     * \brief Function to set the broadcasting rules for the output ndarray
     *     object to the given std::vector<int> object.
     */
    static void
    set_out_arr_bcr(std::vector<int> & bcr)
    {
        bcr.resize(n_iter_axes);
        bcr[0] = 0;
    }

    //__________________________________________________________________________
    /**
     * \brief Function template for setting the broadcasting rules of the
     *     idx'th input ndarray objects based on the dimension of the particular
     *     ndarray object, so it fits the mapping of this mapping model.
     */
    template <int idx>
    static void
    set_in_arr_bcr(std::vector<int> & bcr, ndarray const & arr)
    {
        bcr.resize(n_iter_axes);
        int const nd = arr.get_nd();
        switch(nd)
        {
            case 0:
                // Broadcast the 0-dimensional value over the first axis.
                bcr[0] = -1;
                break;
            case 1:
                // Use each individual value of the 1-dimensional input
                // array for iterating over the first axis.
                bcr[0] = 0;
                break;
            default:
                PyErr_SetString(PyExc_ValueError,
                    "The dimension of the input array can only "
                    "be 0 or 1!");
                python::throw_error_already_set();
        }
    }
};

#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__INDSHAPE

//______________________________________________________________________________
// Define the iterator configuration for this mapping model.
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS    \
    template <                                                                 \
          int InArity                                                          \
        , class OutT                                                           \
        , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, class InT_)      \
    >

#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC      \
    < InArity                                                                  \
    , OutT                                                                     \
    , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)                \
    >
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS
boost::numpy::detail::iter_flags_t const
NxS_to_S BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
::iter_flags = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__ITER_FLAGS;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS
order_t const
NxS_to_S BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
::order = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__ORDER;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS
casting_t const
NxS_to_S BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
::casting = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__CASTING;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS
intptr_t const
NxS_to_S BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
::buffersize = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__BUFFERSIZE;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS
std::vector<int> const
NxS_to_S BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
::op_value_strides = std::vector<int>((!maps_to_void) + InArity, 1);
//------------------------------------------------------------------------------
// Define the iter operand flags for the output array.
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS
boost::numpy::detail::iter_operand_flags_t const
NxS_to_S BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
::out_arr_iter_operand_flags = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__OUT_ARR_ITER_OPERAND_FLAGS;
//------------------------------------------------------------------------------
// Define the (same) iter operand flags for all the input arrays.
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS
template <int idx>
boost::numpy::detail::iter_operand_flags_t const
NxS_to_S BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
::in_arr_iter_operand_flags<idx>
::value = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__IN_ARR_ITER_OPERAND_FLAGS;

#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_SPEC
#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S__GENERAL_TEMPLATE_PARAMS

}// namespace model
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_S_HPP_INCLUDED
