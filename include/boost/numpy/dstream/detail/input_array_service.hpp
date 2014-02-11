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

#include <boost/assert.hpp>

#include <boost/numpy/ndarray.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <class ArrayDefinition>
class input_array_service
{
  public:
    typedef ArrayDefinition
            array_definition_t;

    typedef typename array_definition_t::core_shape_type
            core_shape_t;

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

        // Extract the core shape of the input array.
        arr_core_shape_.resize(core_shape_t::nd::value);
        std::copy(arr_shape_.end() - core_shape_t::nd::value, arr_shape_.end(), arr_core_shape_.begin());
    }

    inline
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
     * \brief Returns the shape of the input array of only the (last) core
     *     dimensions.
     */
    std::vector<intptr_t>
    get_arr_core_shape() const
    {
        return arr_core_shape_;
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

    int const * const
    get_arr_bcr_data() const
    {
        return &(arr_bcr_.front());
    }

    /**
     * \brief Returns the length of the input array's core dimension that has
     *     the given id. If several core dimensions have the same id,
     *     the maximum length of all these core dimensions will be returned.
     *     If the given dimension id is not found for this input array, 0 will
     *     be returned.
     */
    intptr_t
    get_len_of_core_dim(int const id) const
    {
        // TODO: This loop can be unrolled using MPL. The maximal possible core
        //       dimensionality is BOOST_MPL_LIMIT_VECTOR_SIZE.
        std::vector<int> const core_shape_desc = core_shape_t::as_std_vector();
        BOOST_ASSERT(core_shape_desc.size() == arr_core_shape_.size());
        intptr_t len = 0;
        for(int i=0; i<core_shape_t::nd::value; ++i)
        {
            if(core_shape_desc[i] == id && len < arr_core_shape_[i])
            {
                len = arr_core_shape_[i];
            }
        }
        return len;
    }

  protected:
    ndarray arr_;
    std::vector<intptr_t> arr_shape_;
    std::vector<intptr_t> arr_core_shape_;
    int arr_loop_nd_;
    std::vector<int> arr_bcr_;
};

}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_DETAIL_INPUT_ARRAY_SERVICE_HPP_INCLUDED
