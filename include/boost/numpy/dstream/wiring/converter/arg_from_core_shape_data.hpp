/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/arg_from_core_shape_data.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the arg_from_core_shape_data template that should
 *        construct an argument object (which can also be a reference) holding
 *        the function argument data of the corresponding ndarray iterator
 *        operand.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !BOOST_PP_IS_ITERATING

#ifndef BOOST_NUMPY_DSTREAM_WIRING_ARG_FROM_CORE_SHAPE_DATA_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_ARG_FROM_CORE_SHAPE_DATA_HPP_INCLUDED

#include <vector>

#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/mpl/is_std_vector.hpp>
#include <boost/numpy/dstream/wiring/detail/iter_data_ptr.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace converter {

template <class FctArgT, class CoreShape, class ArrDataHoldingT, class Enable=void>
struct arg_from_core_shape_data
{
    typedef arg_from_core_shape_data<FctArgT, CoreShape, ArrDataHoldingT, Enable>
            type;

    // The arg_from_core_shape_data needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_arg_from_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_ARGUMENT_TYPE_FctArgT_AND_ARRAY_CORE_SHAPE_CoreShape_AND_ARRAY_DATA_TYPE_ArrDataHoldingT, (FctArgT, CoreShape, ArrDataHoldingT));
};

namespace detail {

template <class FctArgT>
struct bp_object_arg_from_bp_object_core_shape_data
{
    typedef bp_object_arg_from_bp_object_core_shape_data<FctArgT>
            type;

    typedef FctArgT
            arg_t;
    typedef typename remove_reference<arg_t>::type
            bare_arg_t;

    bp_object_arg_from_bp_object_core_shape_data(
        numpy::detail::iter &         iter
      , size_t const                  iter_op_idx
      , std::vector<intptr_t> const & //core_shape
    )
      : iter_(iter)
      , iter_op_idx_(iter_op_idx)
    {}

    inline
    arg_t
    operator()()
    {
        // The array operand is suppost to be an object array. That means, it
        // stores pointer values to PyObject instances.
        uintptr_t * data = reinterpret_cast<uintptr_t*>(iter_.get_data(iter_op_idx_));
        boost::python::object obj(boost::python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*data)));
        return obj;
    }

    numpy::detail::iter & iter_;
    size_t const          iter_op_idx_;
};

template <class FctArgT, class CoreShape, class ArrDataHoldingT>
struct bp_object_arg_from_core_shape_data
{
    typedef typename mapping::detail::is_scalar<CoreShape>::type
            is_scalar_core_shape;

    typedef typename boost::mpl::eval_if<
              typename boost::mpl::and_<
                  is_scalar_core_shape
                , typename is_same<ArrDataHoldingT, python::object>::type
              >::type
            , bp_object_arg_from_bp_object_core_shape_data<FctArgT>

            , numpy::mpl::unspecified
            >::type
            type;
};

//------------------------------------------------------------------------------

template <class FctArgT, class ArrDataHoldingT>
struct scalar_arg_from_scalar_core_shape_data
{
    typedef scalar_arg_from_scalar_core_shape_data<FctArgT, ArrDataHoldingT>
            type;

    typedef FctArgT
            arg_t;

    scalar_arg_from_scalar_core_shape_data(
        numpy::detail::iter &         iter
      , size_t const                  iter_op_idx
      , std::vector<intptr_t> const & //core_shape
    )
      : iter_(iter)
      , iter_op_idx_(iter_op_idx)
    {}

    inline
    arg_t
    operator()()
    {
        ArrDataHoldingT & arr_value = *reinterpret_cast<ArrDataHoldingT *>(iter_.get_data(iter_op_idx_));
        return arr_value;
    }

    numpy::detail::iter & iter_;
    size_t const          iter_op_idx_;
};

template <class FctArgT, class CoreShape, class ArrDataHoldingT>
struct scalar_arg_from_core_shape_data
{
    typedef typename mapping::detail::is_scalar<CoreShape>::type
            is_scalar_core_shape;

    typedef typename boost::mpl::eval_if<
              is_scalar_core_shape
            , scalar_arg_from_scalar_core_shape_data<FctArgT, ArrDataHoldingT>

            , numpy::mpl::unspecified
            >::type
            type;
};

//------------------------------------------------------------------------------

template <class FctArgT, class ScalarT, class ArrDataHoldingT, unsigned nd>
struct std_vector_of_scalar_arg_from_scalar_core_shape_data;

template <class ArgT, unsigned nd>
struct std_vector_of_bp_object_arg_from_bp_object_core_shape_data;

// Define nd specializations for dimensions J to Z, i.e. up to 18 dimensions.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, 18, <boost/numpy/dstream/wiring/converter/arg_from_core_shape_data.hpp>, 1))
#include BOOST_PP_ITERATE()

template <class FctArgT, class ScalarT, class CoreShape, class ArrDataHoldingT, unsigned nd>
struct std_vector_of_scalar_arg_from_core_shape_data
{
    typedef typename mapping::detail::is_core_shape_of_dim<CoreShape, nd>::type
            is_core_shape_of_dim_nd;

    typedef typename is_scalar<ArrDataHoldingT>::type
            is_scalar_arr_data_holding_type;

    typedef typename boost::mpl::eval_if<
              typename boost::mpl::and_<
                is_core_shape_of_dim_nd
              , is_scalar_arr_data_holding_type
              >::type
            , std_vector_of_scalar_arg_from_scalar_core_shape_data<FctArgT, ScalarT, ArrDataHoldingT, nd>

            , numpy::mpl::unspecified
            >::type
            type;
};

template <class ArgT, class CoreShape, class ArrDataHoldingT, unsigned nd>
struct std_vector_of_bp_object_arg_from_core_shape_data
{
    typedef typename mapping::detail::is_core_shape_of_dim<CoreShape, nd>::type
            is_core_shape_of_dim_nd;

    typedef typename is_same<ArrDataHoldingT, python::object>::type
            is_bp_object_arr_data_holding_type;

    typedef typename boost::mpl::eval_if<
              typename boost::mpl::and_<
                is_core_shape_of_dim_nd
              , is_bp_object_arr_data_holding_type
              >::type
            , std_vector_of_bp_object_arg_from_bp_object_core_shape_data<ArgT, nd>

            , numpy::mpl::unspecified
            >::type
            type;
};

//------------------------------------------------------------------------------

template <class ArgT, class NestedVectorT, class CoreShape, class ArrDataHoldingT, unsigned nd>
struct std_vector_arg_from_core_shape_data
{
    typedef typename remove_reference<NestedVectorT>::type
            bare_vector_t;
    typedef typename bare_vector_t::value_type
            vector_value_t;
    typedef typename remove_reference<vector_value_t>::type
            vector_bare_value_t;

    typedef typename boost::mpl::if_<
              typename is_scalar<vector_bare_value_t>::type
            , std_vector_of_scalar_arg_from_core_shape_data<ArgT, vector_value_t, CoreShape, ArrDataHoldingT, nd>

            , typename boost::mpl::if_<
                typename is_same<vector_bare_value_t, python::object>::type
              , std_vector_of_bp_object_arg_from_core_shape_data<ArgT, CoreShape, ArrDataHoldingT, nd>

              , typename boost::mpl::eval_if<
                  typename numpy::mpl::is_std_vector<vector_bare_value_t>::type
                , std_vector_arg_from_core_shape_data<ArgT, vector_value_t, CoreShape, ArrDataHoldingT, nd+1>

                , numpy::mpl::unspecified
                >::type
              >::type
            >::type
            type;
};

//------------------------------------------------------------------------------

template <class FctArgT, class CoreShape, class ArrDataHoldingT>
struct select_arg_from_core_shape_data_converter
{
    typedef typename remove_reference<FctArgT>::type
            bare_arg_t;

    typedef typename boost::mpl::if_<
              typename is_same<bare_arg_t, python::object>::type
            , bp_object_arg_from_core_shape_data<FctArgT, CoreShape, ArrDataHoldingT>

            , typename boost::mpl::if_<
                typename is_scalar<bare_arg_t>::type
              , scalar_arg_from_core_shape_data<FctArgT, CoreShape, ArrDataHoldingT>

              , typename boost::mpl::eval_if<
                  typename numpy::mpl::is_std_vector<bare_arg_t>::type
                , std_vector_arg_from_core_shape_data<FctArgT, FctArgT, CoreShape, ArrDataHoldingT, 1>

                , numpy::mpl::unspecified
                >::type
              >::type
            >::type
            type;
};

template <class FctArgT, class CoreShape, class ArrDataHoldingT>
struct arg_from_core_shape_data_converter
{
    typedef typename select_arg_from_core_shape_data_converter<FctArgT, CoreShape, ArrDataHoldingT>::type
            builtin_converter_selector;
    typedef typename boost::mpl::eval_if<
              is_same<typename builtin_converter_selector::type, numpy::mpl::unspecified>
            , ::boost::numpy::dstream::wiring::converter::arg_from_core_shape_data<FctArgT, CoreShape, ArrDataHoldingT>
            , builtin_converter_selector
            >::type
            type;
};

}// namespace detail
}// namespace converter
}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_ARG_FROM_CORE_SHAPE_DATA_HPP_INCLUDED
#else

#if BOOST_PP_ITERATION_FLAGS() == 1

#define ND BOOST_PP_ITERATION()

#define BOOST_NUMPY_DSTREAM_vec_def_p1(z, n, data) std::vector<
#define BOOST_NUMPY_DSTREAM_vec_def_p2(z, n, data) >

#define BOOST_NUMPY_DSTREAM_for_dim_begin(z, n, nd) \
    BOOST_PP_REPEAT(BOOST_PP_SUB(nd,n), BOOST_NUMPY_DSTREAM_vec_def_p1, ~) \
    ScalarT \
    BOOST_PP_REPEAT(BOOST_PP_SUB(nd,n), BOOST_NUMPY_DSTREAM_vec_def_p2, ~) \
    BOOST_PP_CAT(v,n); \
    BOOST_PP_CAT(v,n).reserve(core_shape_[n]); \
    for(dim_indices_[n]=0; dim_indices_[n] < core_shape_[n]; ++dim_indices_[n]) {

#define BOOST_NUMPY_DSTREAM_for_dim_end(z, n, nd) \
    BOOST_PP_CAT(v, BOOST_PP_SUB(BOOST_PP_SUB(nd,n),1)).push_back( \
        BOOST_PP_CAT(v,BOOST_PP_SUB(nd,n))); }

template <class FctArgT, class ScalarT, class ArrDataHoldingT>
struct std_vector_of_scalar_arg_from_scalar_core_shape_data<FctArgT, ScalarT, ArrDataHoldingT, ND>
{
    typedef std_vector_of_scalar_arg_from_scalar_core_shape_data<FctArgT, ScalarT, ArrDataHoldingT, ND>
            type;

    typedef FctArgT
            arg_t;

    std_vector_of_scalar_arg_from_scalar_core_shape_data(
        numpy::detail::iter &         iter
      , size_t const                  iter_op_idx
      , std::vector<intptr_t> const & core_shape
    )
      : iter_(iter)
      , iter_op_idx_(iter_op_idx)
      , core_shape_(core_shape)
        // Get the strides of the argument ndarray. Note: This contains the
        // strides for all dimensions, i.e. also for the loop dimensions.
        // The strides for the core dimensions are the last entries in this
        // vector.
      , strides_(iter_.get_operand(iter_op_idx_).get_strides_vector())
      , dim_indices_(std::vector<intptr_t>(ND))
    {}

    inline
    arg_t
    operator()()
    {
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_for_dim_begin, ND)
        ArrDataHoldingT & BOOST_PP_CAT(v,ND) = *reinterpret_cast<ArrDataHoldingT *>(wiring::detail::iter_data_ptr<ND, 0>::get(iter_, iter_op_idx_, dim_indices_, strides_));
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_for_dim_end, ND)

        return v0;
    }

    numpy::detail::iter &         iter_;
    size_t const                  iter_op_idx_;
    std::vector<intptr_t> const & core_shape_;
    std::vector<intptr_t> const   strides_;
    std::vector<intptr_t>         dim_indices_;
};

template <class ArgT>
struct std_vector_of_bp_object_arg_from_bp_object_core_shape_data<ArgT, ND>
{
    typedef std_vector_of_bp_object_arg_from_bp_object_core_shape_data<ArgT, ND>
            type;

    typedef ArgT
            arg_t;

    typedef python::object
            ScalarT;

    std_vector_of_bp_object_arg_from_bp_object_core_shape_data(
        numpy::detail::iter &         iter
      , size_t const                  iter_op_idx
      , std::vector<intptr_t> const & core_shape
    )
      : iter_(iter)
      , iter_op_idx_(iter_op_idx)
      , core_shape_(core_shape)
        // Get the strides of the argument ndarray. Note: This contains the
        // strides for all dimensions, i.e. also for the loop dimensions.
        // The strides for the core dimensions are the last entries in this
        // vector.
      , strides_(iter_.get_operand(iter_op_idx_).get_strides_vector())
      , dim_indices_(std::vector<intptr_t>(ND))
    {}

    inline
    arg_t
    operator()()
    {
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_for_dim_begin, ND)
        uintptr_t * data = reinterpret_cast<uintptr_t*>(wiring::detail::iter_data_ptr<ND, 0>::get(iter_, iter_op_idx_, dim_indices_, strides_));
        boost::python::object BOOST_PP_CAT(v,ND)(boost::python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*data)));
        BOOST_PP_REPEAT(ND, BOOST_NUMPY_DSTREAM_for_dim_end, ND)

        return v0;
    }

    numpy::detail::iter &         iter_;
    size_t const                  iter_op_idx_;
    std::vector<intptr_t> const & core_shape_;
    std::vector<intptr_t> const   strides_;
    std::vector<intptr_t>         dim_indices_;
};

#undef BOOST_NUMPY_DSTREAM_for_dim_end
#undef BOOST_NUMPY_DSTREAM_for_dim_begin
#undef BOOST_NUMPY_DSTREAM_vec_def_p2
#undef BOOST_NUMPY_DSTREAM_vec_def_p1

#undef ND

#endif // BOOST_PP_ITERATION_FLAGS() == 1

#endif // BOOST_PP_IS_ITERATING
