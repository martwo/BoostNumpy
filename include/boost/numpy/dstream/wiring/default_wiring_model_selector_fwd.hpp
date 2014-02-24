/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring/default_wiring_model_selector_fwd.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file forward-declares the default_wiring_model_selector template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_WIRING_DEFAULT_WIRING_MODEL_SELECTOR_FWD_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_DEFAULT_WIRING_MODEL_SELECTOR_FWD_HPP_INCLUDED

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {

template <
      class MappingDefinition
    , class FTypes
    , class Enable = void
>
struct default_wiring_model_selector;

}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_DEFAULT_WIRING_MODEL_SELECTOR_FWD_HPP_INCLUDED
