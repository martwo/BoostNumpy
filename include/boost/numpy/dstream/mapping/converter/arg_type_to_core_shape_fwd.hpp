/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/converter/arg_type_to_core_shape_fwd.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file forward declares the arg_type_to_core_shape converter
 *        facility for converting a function's argument type to a core shape
 *        type. This allows the user to define his own converter template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_ARG_TYPE_TO_CORE_SHAPE_FWD_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_ARG_TYPE_TO_CORE_SHAPE_FWD_HPP_INCLUDED

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace converter {

template <class T, class Enable>
struct arg_type_to_core_shape;

}// namespace converter
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_MAPPING_CONVERTER_ARG_TYPE_TO_CORE_SHAPE_FWD_HPP_INCLUDED
