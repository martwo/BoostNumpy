/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/converter/arg_type_to_core_shape/std_vector_of_bp_object.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the arg_type_to_core_shape template for
 *        converting a std::vector< boost::python::object > type to a
 *        1-dimensional core shape type. The core dimension has the name I.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_STD_VECTOR_TO_BP_OBJECT_TO_CORE_SHAPE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_STD_VECTOR_TO_BP_OBJECT_TO_CORE_SHAPE_HPP_INCLUDED

#include <vector>

#include <boost/python/object_fwd.hpp>

#include <boost/numpy/mpl/is_std_vector_of.hpp>
#include <boost/numpy/dstream/mapping/converter/arg_type_to_core_shape_fwd.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace converter {

template <class T>
struct arg_type_to_core_shape<
    T
  , typename enable_if< typename numpy::mpl::is_std_vector_of< T, python::object >::type >::type
>
{
    typedef mapping::detail::core_shape<1>::shape< dim::I >
            type;
};

}// namespace converter
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_STD_VECTOR_TO_BP_OBJECT_TO_CORE_SHAPE_HPP_INCLUDED
