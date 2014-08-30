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

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {
namespace converter {

template <class FctArgT, class ArrDataHoldingT, class Enable=void>
struct arg_from_core_shape_data
{
    typedef arg_from_core_shape_data<FctArgT, ArrDataHoldingT, Enable>
            type;

    // The arg_from_core_shape_data needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_arg_from_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_ARGUMENT_TYPE_FctArgT_AND_ARRAY_DATA_TYPE_ArrDataHoldingT, (FctArgT, ArrDataHoldingT));
};

template <class FctArgT, class ArrDataHoldingT, class Enable=void>
struct boost_python_object_arg_from_core_shape_data
{
    typedef boost_python_object_arg_from_core_shape_data<FctArgT, ArrDataHoldingT, Enable>
            type;

    // The boost_python_object_arg_from_core_shape_data needs to be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_boost_python_object_arg_from_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_ARGUMENT_TYPE_FctArgT_AND_ARRAY_DATA_HOLDING_TYPE_ArrDataHoldingT, (FctArgT, ArrDataHoldingT));
};

template <class FctArgT, class ArrDataHoldingT, class Enable=void>
struct std_vector_of_boost_python_object_arg_from_core_shape_data
{
    typedef std_vector_of_boost_python_object_arg_from_core_shape_data<FctArgT, ArrDataHoldingT, Enable>
            type;

    // The std_vector_of_boost_python_object_arg_from_core_shape_data needs to
    // be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_std_vector_of_boost_python_object_arg_from_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_ARGUMENT_TYPE_FctArgT_AND_ARRAY_DATA_HOLDING_TYPE_ArrDataHoldingT, (FctArgT, ArrDataHoldingT));
};

template <class FctArgT, class ArrDataHoldingT, class Enable=void>
struct std_vector_of_scalar_arg_from_core_shape_data
{
    typedef std_vector_of_scalar_arg_from_core_shape_data<FctArgT, ArrDataHoldingT, Enable>
            type;

    // The std_vector_of_boost_python_object_arg_from_core_shape_data needs to
    // be specialized.
    // Trigger a compilation error with a meaningful message.
    BOOST_MPL_ASSERT_MSG(false,
        THE_std_vector_of_scalar_arg_from_core_shape_data_CONVERTER_NEED_TO_BE_SPECIALIZED_FOR_FUNCTION_ARGUMENT_TYPE_FctArgT_AND_ARRAY_DATA_HOLDING_TYPE_ArrDataHoldingT, (FctArgT, ArrDataHoldingT));
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

template <class FctArgT>
struct std_vector_of_bp_object_arg_from_bp_object_core_shape_data
{
    typedef std_vector_of_bp_object_arg_from_bp_object_core_shape_data<FctArgT>
            type;
    typedef FctArgT
            arg_t;
    typedef typename remove_reference<arg_t>::type
            bare_arg_t;

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

template <class FctArgT, class ArrDataHoldingT>
struct bp_object_arg_from_core_shape_data
{
    typedef typename boost::mpl::eval_if<
              typename is_same<ArrDataHoldingT, python::object>::type
            , bp_object_arg_from_bp_object_core_shape_data<FctArgT>
            , ::boost::numpy::dstream::wiring::converter::boost_python_object_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>
            >::type
            type;
};

template <class FctArgT, class ArrDataHoldingT>
struct std_vector_of_bp_object_arg_from_core_shape_data
{
    typedef typename boost::mpl::eval_if<
              typename is_same<ArrDataHoldingT, python::object>::type
            , std_vector_of_bp_object_arg_from_bp_object_core_shape_data<FctArgT>
            , ::boost::numpy::dstream::wiring::converter::std_vector_of_boost_python_object_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>
            >::type
            type;
};

template <class FctArgT, class ArrDataHoldingT>
struct scalar_arg_from_core_shape_data
{
    typedef scalar_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>
            type;
    typedef FctArgT
            arg_t;
    typedef typename remove_reference<arg_t>::type
            bare_arg_t;

    arg_t
    apply(
        numpy::detail::iter &         iter
      , size_t                        iter_op_idx
      , std::vector<intptr_t> const & core_shape
    )
    {
        arg_t arr_value = *reinterpret_cast<bare_arg_t *>(iter.get_data(iter_op_idx));
        return arr_value;
    }
};

//------------------------------------------------------------------------------

template <class FctArgT, class ArrDataHoldingT>
struct std_vector_of_scalar_arg_from_scalar_core_shape_data
{
    typedef std_vector_of_scalar_arg_from_scalar_core_shape_data<FctArgT, ArrDataHoldingT>
            type;
    typedef FctArgT
            arg_t;
    typedef typename remove_reference<FctArgT>::type
            bare_arg_t;

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
        bare_arg_t v;
        v.reserve(N);
        for(intptr_t i=0; i<N; ++i)
        {
            typename bare_arg_t::value_type * value_ptr = reinterpret_cast<typename bare_arg_t::value_type *>(iter.get_data(iter_op_idx) + i*op_item_stride);
            v.push_back(*value_ptr);
        }
        return v;
    }
};

template <class FctArgT, class ArrDataHoldingT>
struct std_vector_of_scalar_arg_from_core_shape_data
{
    typedef typename remove_reference<FctArgT>::type
            bare_arg_t;
    typedef typename boost::mpl::eval_if<
              typename is_same<typename remove_reference<typename bare_arg_t::value_type>::type, ArrDataHoldingT>::type
            , std_vector_of_scalar_arg_from_scalar_core_shape_data<FctArgT, ArrDataHoldingT>
            , ::boost::numpy::dstream::wiring::converter::std_vector_of_scalar_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>
            >::type
            type;
};

//------------------------------------------------------------------------------

template <class FctArgT, class ArrDataHoldingT>
struct select_arg_from_core_shape_data_converter
{
    typedef typename remove_reference<FctArgT>::type
            bare_arg_t;

    typedef typename boost::mpl::eval_if<
              typename is_same<bare_arg_t, python::object>::type
            , bp_object_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>

            , typename boost::mpl::eval_if<
                typename boost::mpl::and_< typename is_scalar<bare_arg_t>::type, typename is_same<bare_arg_t, ArrDataHoldingT>::type >::type
              , scalar_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>

              , typename boost::mpl::eval_if<
                  typename numpy::mpl::is_std_vector_of_scalar<bare_arg_t>::type
                , std_vector_of_scalar_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>

                , typename boost::mpl::eval_if<
                    typename numpy::mpl::is_std_vector_of<bare_arg_t, python::object>::type
                  , std_vector_of_bp_object_arg_from_core_shape_data<FctArgT, ArrDataHoldingT>

                  , ::boost::numpy::dstream::wiring::converter::arg_from_core_shape_data<FctArgT, ArrDataHoldingT>
                  >::type
                >::type
              >::type
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
