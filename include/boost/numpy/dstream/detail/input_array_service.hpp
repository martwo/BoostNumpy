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

#include <sstream>
#include <vector>

#include <boost/numpy/detail/logging.hpp>
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
      : arr_core_shape_ids_(core_shape_t::as_std_vector())
      , arr_(arr)
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
            BOOST_NUMPY_LOG("input_array_service: reshape array")
            arr_ = arr_.reshape(arr_shape_);
        }

        // Calculate the number of loop dimensions the input array already
        // provides.
        arr_loop_nd_ = arr_shape_.size() - core_shape_t::nd::value;

        // Extract the core shape of the input array.
        arr_core_shape_.resize(core_shape_t::nd::value);
        std::copy(arr_shape_.end() - core_shape_t::nd::value, arr_shape_.end(), arr_core_shape_.begin());

        // Validate that the input array's fixed sized core dimensions, i.e. with
        // dimension id values greater than 0, have the correct lengths.
        for(int i=0; i<core_shape_t::nd::value; ++i)
        {
            if(arr_core_shape_ids_[i] > 0 && arr_core_shape_[i] != arr_core_shape_ids_[i])
            {
                std::stringstream msg;
                msg << "The "<< (i+1) <<"th fixed sized array dimension has "
                    << "the wrong length! Is " << arr_core_shape_[i] << ", "
                    << "but must be " << arr_core_shape_ids_[i] << "!";
                PyErr_SetString(PyExc_ValueError, msg.str().c_str());
                python::throw_error_already_set();
            }
        }
    }

    /**
     * \brief Prepends a loop dimension with one iteration.
     */
    inline
    void
    prepend_loop_dimension()
    {
        arr_shape_.insert(arr_shape_.begin(), 1);
        arr_ = arr_.reshape(arr_shape_);
        ++arr_loop_nd_;
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

    int *
    get_arr_bcr_data()
    {
        return &(arr_bcr_.front());
    }

    int const * const
    get_arr_bcr_data() const
    {
        return &(arr_bcr_.front());
    }

    std::vector<int> const &
    get_arr_core_shape_ids() const
    {
        return arr_core_shape_ids_;
    }

    /**
     * \brief Returns the length of the input array's core dimension that has
     *     the given id. If several core dimensions have the same id,
     *     the maximum length of all these core dimensions will be returned.
     *     If the given dimension id is not found for this input array, 0 will
     *     be returned.
     */
    intptr_t
    get_core_dim_len(int const id) const
    {
        // TODO: This loop can be unrolled using MPL. The maximal possible core
        //       dimensionality is BOOST_MPL_LIMIT_VECTOR_SIZE.
        intptr_t len = 0;
        for(int i=0; i<core_shape_t::nd::value; ++i)
        {
            if(arr_core_shape_ids_[i] == id && len < arr_core_shape_[i])
            {
                len = arr_core_shape_[i];
            }
        }
        return len;
    }

  protected:
    std::vector<int> const arr_core_shape_ids_;
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
