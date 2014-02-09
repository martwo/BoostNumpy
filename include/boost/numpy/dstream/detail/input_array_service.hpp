/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/detail/input_array_service.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the input_array_service template that provides
 *        functionalities for an input array of a specified core shape.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_DETAIL_INPUT_ARRAY_SERVICE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_INPUT_ARRAY_SERVICE_HPP_INCLUDED

#include <iostream>

#include <vector>

#include <boost/numpy/ndarray.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <class CoreShape>
struct input_array_service
{
    typedef CoreShape core_shape_t;

    input_array_service(ndarray const & arr)
      : arr_(arr)
      , arr_shape_(arr.get_shape_vector())
    {
        // Prepend ones to the shape while the shape dimensionality is less than
        // the core shape dimensionality.
        bool shape_changed = false;
        while(int(arr_shape_.size()) < core_shape_t::nd::value)
        {
            arr_shape_.insert(arr_shape_.begin(), 1);
            shape_changed = true;
        }
        if(shape_changed)
        {
            std::cout << "input_array_service: reshape array" << std::endl;
            arr_ = arr_.reshape(arr_shape_);
        }

        // Calculate the number of loop dimensions the input array already
        // provides.
        arr_loop_nd_ = arr_shape_.size() - core_shape_t::nd::value;
    }

    ndarray const &
    get_arr() const
    {
        return arr_;
    }

    inline
    int
    get_arr_nd() const
    {
        return arr_shape_.size();
    }

    inline
    std::vector<intptr_t>
    get_arr_shape() const
    {
        return arr_shape_;
    }

    inline
    int
    get_arr_loop_nd() const
    {
        return arr_loop_nd_;
    }

    /**
     * \brief Returns the loop shape of the input array.
     */
    std::vector<intptr_t>
    get_arr_loop_shape() const
    {
        std::vector<intptr_t> arr_loop_shape(arr_loop_nd_);
        for(int i=0; i<arr_loop_nd_; ++i)
        {
            arr_loop_shape[i] = arr_shape_[i];
        }
        return arr_loop_shape;
    }

    /**
     * \brief Sets the broadcasting rules for the loop dimensions of the input
     *     array to normal numpy broadcasting rules.
     */
    void
    set_arr_bcr(int const loop_nd)
    {
        arr_bcr_.resize(loop_nd);
        for(int loop_axis=0; loop_axis<loop_nd; ++loop_axis)
        {
            if((loop_nd - loop_axis) > arr_loop_nd_)
            {
                arr_bcr_[loop_axis] = -1;
            }
            else
            {
                arr_bcr_[loop_axis] = loop_axis - (loop_nd - arr_loop_nd_);
            }
        }
    }

    int * const
    get_arr_bcr_data()
    {
        return &(arr_bcr_.front());
    }

    ndarray arr_;
    std::vector<intptr_t> arr_shape_;
    int arr_loop_nd_;
    std::vector<int> arr_bcr_;
};

}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_DETAIL_INPUT_ARRAY_SERVICE_HPP_INCLUDED
