/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/converter/return_type_to_out_mapping_fwd.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file forward declares the return_type_to_out_mapping converter
 *        facility for converting a function's return type to an output mapping
 *        type. This allows the user to define his own converter template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_FWD_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_FWD_HPP_INCLUDED

#include <boost/numpy/dstream/mapping.hpp>
#include <boost/numpy/dstream/detail/core_shape.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace converter {

template <class T, class Enable>
struct return_type_to_out_mapping;

}// namespace converter
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_RETURN_TYPE_TO_OUT_MAPPING_FWD_HPP_INCLUDED
