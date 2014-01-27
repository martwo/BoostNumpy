/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/out_arr_transforms/column_tuple.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines an output arrar transformation template for slicing
 *        a 2-dimensional output array into a tuple of arrays where each array
 *        represents one column of the original output array.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_COLUMN_TUPLE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_COLUMN_TUPLE_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#include <boost/numpy/detail/prefix.hpp>

#include <boost/python/slice.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/dstream/out_arr_transform.hpp>
#include <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis_if_single_input_and_scalarize.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace out_arr_transforms {

namespace detail {
template <int InArity, class MappingModel>
struct column_tuple_base;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/out_arr_transforms/column_tuple.hpp>))
#include BOOST_PP_ITERATE()

}// namespace detail

template <class MappingModel>
struct column_tuple
  : detail::column_tuple_base<MappingModel::in_arity, MappingModel>
{
    typedef column_tuple<MappingModel>
            type;
};

struct column_tuple_selector
  : out_arr_transforms::out_arr_transform_selector_type
{
    template <class MappingModel>
    struct out_arr_transform
    {
        typedef column_tuple<MappingModel>
                type;
    };
};

}/*namespace out_arr_transforms*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORMS_COLUMN_TUPLE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

// Partial specialization for input arity N.
template <class MappingModel>
struct column_tuple_base<N, MappingModel>
  : out_arr_transform_base<MappingModel>
{
    inline static int
    apply(ndarray & out_arr, BOOST_PP_ENUM_PARAMS(N, ndarray const & in_arr_))
    {
        intptr_t const axis = 2; // this could become an option.

        intptr_t const nd = out_arr.get_nd();
        if(! nd >= axis)
        {
            PyErr_SetString(PyExc_ValueError,
                "The dimension of the output array must be at least 2 in order "
                "to use the column_tuple output array transformation!");
            python::throw_error_already_set();
        }

        // Get the number of columns (= the length of the axis).
        intptr_t const n_axis = out_arr.shape(axis-1);

        dtype const dt = out_arr.get_dtype();

        // Create the output list where we will hold each column before
        // converting it into a tuple. Note: The detour through the list could
        // be avoiding by using MPL techniques. But this could limit the
        // maximal length of the tuple.
        python::list out_list;
        for(intptr_t i=0; i<n_axis; ++i)
        {
            // Create a slice object for the i'th column. I.e. a list of slice
            // objects - one for each dimension of the output array (up to the
            // axis to slice on).
            python::list slist;
            for(intptr_t s=0; s<axis-1; ++s)
            {
                slist.append(python::slice());
            }
            slist.append(python::slice(i,i+1));

            PyObject* pycolumn = PyObject_GetItem((PyObject*)out_arr.ptr(), (PyObject*)slist.ptr());
            ndarray column = python::extract<ndarray>(pycolumn);
            Py_DECREF(pycolumn);

            // If axis is the last dimension, each element of the column will be
            // an array with one element. So we'll get rid of the last dimension
            // by simply reshaping the column array.
            if(axis == nd)
            {
                python::list new_shape;
                for(intptr_t s=0; s<nd-1; ++s)
                {
                    new_shape.append(column.shape(s));
                }
                column = column.reshape(new_shape);
            }

            squeeze_first_axis_if_single_input_and_scalarize<MappingModel>::apply(column, BOOST_PP_ENUM_PARAMS(N, in_arr_));

            out_list.append(column);
        }
        python::tuple out_tuple(out_list);

        out_arr = python::object(out_tuple);
        return 1;
    }
};

#undef N

#endif // BOOST_PP_IS_ITERATING
