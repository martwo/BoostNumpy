/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/mapping/models/_NxS_to_X.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines a mapping model template (and its specializations
 *        for the different numbers of input arrays and output data shapes) for
 *        functions that expact only scalar data shaped input arrays and that
 *        are returning a one-dimensional data shaped output array of arbitrary
 *        length.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X_HPP_INCLUDED

#include <stdint.h>

#include <vector>

#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <boost/numpy/pp.hpp>
#include <boost/numpy/limits.hpp>
#include <boost/numpy/detail/iter.hpp>
#include <boost/numpy/dstream/dshape.hpp>
#include <boost/numpy/dstream/mapping.hpp>

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ITER_FLAGS
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ITER_FLAGS                 \
    boost::numpy::detail::iter::EXTERNAL_LOOP                                  \
    | boost::numpy::detail::iter::DONT_NEGATE_STRIDES
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ORDER
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ORDER                      \
    KEEPORDER
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__CASTING
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__CASTING                    \
    SAME_KIND_CASTING
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__BUFFERSIZE
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__BUFFERSIZE                 \
    0
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__OUT_ARR_ITER_OPERAND_FLAGS
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__OUT_ARR_ITER_OPERAND_FLAGS \
    boost::numpy::detail::iter_operand::WRITEONLY                              \
    | boost::numpy::detail::iter_operand::NBO                                  \
    | boost::numpy::detail::iter_operand::ALIGNED
#endif

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__IN_ARR_ITER_OPERAND_FLAGS
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__IN_ARR_ITER_OPERAND_FLAGS  \
    boost::numpy::detail::iter_operand::READONLY
#endif

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace model {

template <
      int InArity
    , class OutArrDShape
    , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, class InArrDShape_, = BOOST_NUMPY_PP_MPL_VOID BOOST_PP_INTERCEPT)
>
struct _NxS_to_X
  : base_mapping_model<
          InArity
        , OutArrDShape
        , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InArrDShape_)
    >
{
    typedef boost::is_same<OutArrDShape, void_dshape> maps_to_void_t;
    BOOST_STATIC_CONSTANT(bool, maps_to_void = maps_to_void_t::value);

    typedef OutArrDShape out_arr_dshape_t;

    //--------------------------------------------------------------------------
    // Declare the iterator configuration needed for this mapping model as
    // static constants.
    static boost::numpy::detail::iter_flags_t const iter_flags;
    static order_t                            const order;
    static casting_t                          const casting;
    static intptr_t                           const buffersize;
    BOOST_STATIC_CONSTANT(int, n_op = 1 + InArity);
    BOOST_STATIC_CONSTANT(int, n_iter_axes = 2);

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
     * \brief Sets the iteration shape to the given std::vector<int> object
     *     for an iteration where the first axis has n_axis_1_elements
     *     elements.
     */
    static void
    set_itershape(std::vector<intptr_t> & itershape, intptr_t n_axis_1_elements)
    {
        itershape.resize(n_iter_axes);
        itershape[0] = n_axis_1_elements;
        itershape[1] = out_arr_dshape_t::template shape<0>();
    }

    //__________________________________________________________________________
    /**
     * \brief Function to sets the broadcasting rules for the output ndarray
     *     object to the given std::vector<int> object.
     */
    static void
    set_out_arr_bcr(std::vector<int> & bcr)
    {
        bcr.resize(n_iter_axes);
        bcr[0] = 0;
        bcr[1] = 1;
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
                // Broadcast the 0-dimensional value over the first and second
                // axis.
                bcr[0] = -1;
                bcr[1] = -1;
                break;
            case 1:
                // Use each individual value of the 1-dimensional input
                // array for iterating over the first axis and broadcast that
                // value over the second axis.
                bcr[0] = 0;
                bcr[1] = -1;
                break;
            default:
                PyErr_SetString(PyExc_ValueError,
                    "The dimension of the input array can only "
                    "be 0 or 1!");
                python::throw_error_already_set();
        }
    }
};

//______________________________________________________________________________
// Define the iterator configuration for this mapping model.
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS    \
    template <                                                                 \
          int InArity                                                          \
        , class OutArrDShape                                                   \
        , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, class InArrDShape_)\
    >

#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC      \
    <                                                                          \
          InArity                                                              \
        , OutArrDShape                                                         \
        , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InArrDShape_)    \
    >

//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
boost::numpy::detail::iter_flags_t const
_NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::iter_flags = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ITER_FLAGS;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
order_t const
_NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::order = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ORDER;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
casting_t const
_NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::casting = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__CASTING;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
intptr_t const
_NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::buffersize = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__BUFFERSIZE;
//------------------------------------------------------------------------------
// Define the iter operand flags for the output array.
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
boost::numpy::detail::iter_operand_flags_t const
_NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::out_arr_iter_operand_flags = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__OUT_ARR_ITER_OPERAND_FLAGS;
//------------------------------------------------------------------------------
// Define the (same) iter operand flags for all the input arrays.
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
template <int idx>
boost::numpy::detail::iter_operand_flags_t const
_NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::in_arr_iter_operand_flags<idx>
::value = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__IN_ARR_ITER_OPERAND_FLAGS;

#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC

//______________________________________________________________________________
// Define typedef's for N number of input arrays and for a (X,)-data-shaped
// output array named _NxS_to_X where N and X are substituted with integer
// values.
#define BOOST_PP_DEF(z, N, X) \
    typedef _NxS_to_X<                                                         \
          N                                                                    \
        , BOOST_PP_CAT(BOOST_PP_CAT(_,X),_dshape)                              \
        BOOST_NUMPY_PP_COMMA_IF_LIST(N, scalar_dshape)                         \
        BOOST_NUMPY_PP_COMMA_IF_LIST(BOOST_PP_SUB(BOOST_NUMPY_LIMIT_INPUT_ARITY, N), BOOST_NUMPY_PP_MPL_VOID)\
    > BOOST_PP_CAT(BOOST_PP_CAT(BOOST_PP_CAT(_,N),xS_to_),X);

#define BOOST_PP_LOCAL_LIMITS (1, BOOST_NUMPY_LIMIT_INPUT_ARITY)
#define BOOST_PP_LOCAL_MACRO(X)                                                \
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_ADD(BOOST_NUMPY_LIMIT_INPUT_ARITY, 1), BOOST_PP_DEF, X)
#include BOOST_PP_LOCAL_ITERATE()

#undef BOOST_PP_DEF

}// namespace model
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X_HPP_INCLUDED
