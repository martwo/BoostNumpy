/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/detail/output_array_service.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the output_array_service template that provides
 *        functionalities for an output array of a specified core shape.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_DETAIL_OUTPUT_ARRAY_SERVICE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_OUTPUT_ARRAY_SERVICE_HPP_INCLUDED

#include <algorithm>
#include <sstream>
#include <vector>

#include <boost/assert.hpp>
#include <boost/python/object_fwd.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/detail/utils.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

template <
      class ArrayDefinition
    , class LoopService
>
class output_array_service
{
  public:
    typedef ArrayDefinition
            array_definition_t;

    typedef typename array_definition_t::core_shape_type
            core_shape_t;

    typedef typename array_definition_t::value_type
            value_type;

    typedef LoopService loop_service_t;

    output_array_service(
          loop_service_t const & loop_service
        , python::object const & out_obj
        , ndarray::flags         flags = ndarray::NONE
    )
      : loop_service_(loop_service)
      , arr_(from_object(python::object()))
    {
        // Generate the shape of the array based on its core shape and the
        // information from the loop service.
        std::vector<intptr_t> const loop_shape = loop_service_.get_loop_shape();
        int const loop_nd = loop_shape.size();
        arr_shape_.resize(loop_nd + core_shape_t::nd::value);
        std::copy(loop_shape.begin(), loop_shape.end(), arr_shape_.begin());
        for(int i=0; i<core_shape_t::nd::value; ++i)
        {
            // Get core dimension id of the i'th core dimension.
            int const id = core_shape_t::id(i);
            if(id > 0)
            {
                // The core dimension is a fixed size dimension of size id.
                arr_shape_[loop_nd+i] = id;
            }
            else
            {
                // The core dimension id refers to a core dimension of one of
                // the input arrays. Ask the loop service about its length.
                intptr_t len = loop_service_.get_core_dim_len(id);
                BOOST_ASSERT(len > 0);
                arr_shape_[loop_nd+i] = len;
            }
        }

        // Create the output array object. Either from the provided bp::object
        // or a new one.
        if(out_obj != python::object())
        {
            // An output array was already provided by the user. Check its
            // correct shape.
            ndarray arr = from_object(out_obj, flags);
            if(! arr.has_shape(arr_shape_))
            {
                std::stringstream msg;
                msg << "The provided output array does not have the required "
                    << "shape " << numpy::detail::pprint_shape(arr_shape_) << "!";
                PyErr_SetString(PyExc_ValueError, msg.str().c_str());
                python::throw_error_already_set();
            }
            arr_ = arr;
        }
        else
        {
            // No output array was provided, create a new array.
            dtype const dt = dtype::get_builtin< value_type >();
            arr_ = zeros(arr_shape_, dt);
        }

        // Set the broadcasting rules for the output array. By construction all
        // axes are already present, so no broadcasting (i.e. -1 axes) need to
        // be done.
        arr_bcr_.resize(loop_nd);
        for(int loop_axis=0; loop_axis<loop_nd; ++loop_axis)
        {
            arr_bcr_[loop_axis] = loop_axis;
        }
    }

    inline
    ndarray const &
    get_arr() const
    {
        return arr_;
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

  protected:
    loop_service_t const & loop_service_;
    ndarray arr_;
    std::vector<intptr_t> arr_shape_;
    std::vector<int> arr_bcr_;
};

}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_DETAIL_OUTPUT_ARRAY_SERVICE_HPP_INCLUDED
