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
#ifndef BOOST_NUMPY_DSTREAM_WIRING_ARG_FROM_CORE_SHAPE_DATA_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_ARG_FROM_CORE_SHAPE_DATA_HPP_INCLUDED

#include <vector>

#include <boost/mpl/and.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/remove_reference.hpp>

#include <boost/numpy/mpl/is_std_vector.hpp>

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

    static
    arg_t
    apply(
        numpy::detail::iter &         iter
      , size_t                        iter_op_idx
      , std::vector<intptr_t> const & core_shape
    )
    {
        // The array operand is suppost to be an object array. That means, it
        // stores pointer values to PyObject instances.
        uintptr_t * data = reinterpret_cast<uintptr_t*>(iter.get_data(iter_op_idx));

        boost::python::object obj(boost::python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*data)));
        return obj;
    }
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

    static
    arg_t
    apply(
        numpy::detail::iter &         iter
      , size_t                        iter_op_idx
      , std::vector<intptr_t> const & core_shape
    )
    {
        ArrDataHoldingT & arr_value = *reinterpret_cast<ArrDataHoldingT *>(iter.get_data(iter_op_idx));
        return arr_value;
    }
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

template <class FctArgT, class ScalarT, class ArrDataHoldingT>
struct std_vector_of_scalar_arg_from_scalar_core_shape_data<FctArgT, ScalarT, ArrDataHoldingT, 1>
{
    typedef std_vector_of_scalar_arg_from_scalar_core_shape_data<FctArgT, ScalarT, ArrDataHoldingT, 1>
            type;

    typedef FctArgT
            arg_t;
    typedef typename remove_reference<arg_t>::type
            vector_t;

    // Note: The function argument type (i.e. std::vector) cannot be a
    //       reference but the scalar values can be references to the actual
    //       stored data.
    static
    arg_t
    apply(
        numpy::detail::iter &         iter
      , size_t                        iter_op_idx
      , std::vector<intptr_t> const & core_shape
    )
    {
        if(core_shape.size() != 1)
        {
            PyErr_SetString(PyExc_ValueError,
                "The core shape of the argument array must be of dimension 1!");
            python::throw_error_already_set();
        }
        intptr_t const N = core_shape[0];
        intptr_t const op_item_stride = iter.get_item_size(iter_op_idx);
        vector_t v;
        v.reserve(N);
        for(intptr_t i=0; i<N; ++i)
        {
            ArrDataHoldingT & value = *reinterpret_cast<ArrDataHoldingT *>(iter.get_data(iter_op_idx) + i*op_item_stride);
            v.push_back(value);
        }
        return v;
    }
};



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

//------------------------------------------------------------------------------

template <class ArgT, unsigned nd>
struct std_vector_of_bp_object_arg_from_bp_object_core_shape_data;

template <class ArgT>
struct std_vector_of_bp_object_arg_from_bp_object_core_shape_data<ArgT, 1>
{
    typedef std_vector_of_bp_object_arg_from_bp_object_core_shape_data<ArgT, 1>
            type;

    typedef ArgT
            arg_t;

    static
    arg_t
    apply(
        numpy::detail::iter &         iter
      , size_t                        iter_op_idx
      , std::vector<intptr_t> const & core_shape
    )
    {
        if(core_shape.size() != 1)
        {
            PyErr_SetString(PyExc_ValueError,
                "The core shape of the argument array must be of dimension 1!");
            python::throw_error_already_set();
        }
        intptr_t const N = core_shape[0];
        intptr_t const op_item_stride = iter.get_item_size(iter_op_idx);
        std::vector<python::object> v;
        v.reserve(N);
        for(intptr_t i=0; i<N; ++i)
        {
            uintptr_t * data = reinterpret_cast<uintptr_t*>(iter.get_data(iter_op_idx) + i*op_item_stride);
            boost::python::object obj(boost::python::detail::borrowed_reference(reinterpret_cast<PyObject*>(*data)));
            v.push_back(obj);
        }
        return v;
    }
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
