/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - 2014
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/mapping/models/NxS_to_X.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines a mapping model template for mapping N scalar input
 *        values to a 1-dimensional output array with X elements.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/type_traits/is_same.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/dstream/dshape.hpp>
#include <boost/numpy/dstream/mapping.hpp>
#include <boost/numpy/mpl/types.hpp>

#include <boost/numpy/dstream/wiring/models/vector_callable.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace model {

#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__INDSHAPE(z, n, data)       \
    BOOST_PP_COMMA_IF(n) scalar_dshape< BOOST_PP_CAT(InT_, n) >

#ifndef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ITER_FLAGS
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ITER_FLAGS                 \
    boost::numpy::detail::iter::DONT_NEGATE_STRIDES
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

template <
      int N
    , int X
    , class OutT
    , BOOST_PP_ENUM_BINARY_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, class InT_, = numpy::mpl::unspecified BOOST_PP_INTERCEPT)
>
struct NxS_to_X
  : base_mapping_model<
          N
        , dshape_1d<X, OutT>
        , BOOST_PP_REPEAT(BOOST_NUMPY_LIMIT_INPUT_ARITY, BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__INDSHAPE, ~)
    >
{
    // Define the default wiring model selector suitable for this mapping model.
    struct default_wiring_model_selector
    {
        typedef wiring::model::vector_callable
                type;
    };

    // We assume, that this mapping model will never map to a void output. If
    // a void output is needed that NxS_to_S mapping model should be used
    // instead!
    BOOST_STATIC_CONSTANT(bool, maps_to_void = false);

    //--------------------------------------------------------------------------
    // Declare the iterator configuration needed for this mapping model as
    // static constants.
    static boost::numpy::detail::iter_flags_t const iter_flags;
    static order_t                            const order;
    static casting_t                          const casting;
    static intptr_t                           const buffersize;
    BOOST_STATIC_CONSTANT(int, n_op = 1 + N);
    BOOST_STATIC_CONSTANT(int, n_iter_axes = 2);

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
        itershape[1] = X;
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

#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__INDSHAPE

//______________________________________________________________________________
// Define the iterator configuration for this mapping model.
#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS    \
    template <                                                                 \
          int N                                                                \
        , int X                                                                \
        , class OutT                                                           \
        , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, class InT_)      \
    >

#define BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC      \
    < N                                                                        \
    , X                                                                        \
    , OutT                                                                     \
    , BOOST_PP_ENUM_PARAMS(BOOST_NUMPY_LIMIT_INPUT_ARITY, InT_)                \
    >

//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
boost::numpy::detail::iter_flags_t const
NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::iter_flags = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ITER_FLAGS;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
order_t const
NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::order = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__ORDER;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
casting_t const
NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::casting = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__CASTING;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
intptr_t const
NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::buffersize = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__BUFFERSIZE;
//------------------------------------------------------------------------------
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
std::vector<int> const
NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::op_value_strides = std::vector<int>(1 + N, X);
//------------------------------------------------------------------------------
// Define the iter operand flags for the output array.
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
boost::numpy::detail::iter_operand_flags_t const
NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::out_arr_iter_operand_flags = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__OUT_ARR_ITER_OPERAND_FLAGS;
//------------------------------------------------------------------------------
// Define the (same) iter operand flags for all the input arrays.
BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS
template <int idx>
boost::numpy::detail::iter_operand_flags_t const
NxS_to_X BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
::in_arr_iter_operand_flags<idx>
::value = BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__IN_ARR_ITER_OPERAND_FLAGS;

#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_SPEC
#undef BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X__GENERAL_TEMPLATE_PARAMS

namespace detail {

template <int in_arity, int X, class IOTypes>
struct select_base;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/mapping/models/NxS_to_X.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail

template <int X>
struct NxS_to_
  : mapping::mapping_model_selector_type
{
    template <class IOTypes>
    struct select
      : detail::select_base<IOTypes::in_arity, X, IOTypes>
    {};
};

}// namespace model
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_MODEL_NXS_TO_X_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

// Specialization for input arity N.
template <int X, class IOTypes>
struct select_base<N, X, IOTypes>
{
    // IOTypes::out_t is a container type. If the container type is STL conform
    // it must have the typedef ``value_type``.
    typedef NxS_to_X<N, X, typename IOTypes::out_t::value_type, BOOST_PP_ENUM_PARAMS(N, typename IOTypes::in_t_)>
            type;
};

#undef N

#endif // !BOOST_PP_IS_ITERATING
