/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/wiring.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines templates for data stream wiring functionalty.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_WIRING_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_HPP_INCLUDED

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {

struct wiring_model_type
{};

struct wiring_model_selector_type
{};

//==============================================================================
/**
 * @brief The base_wiring_model template provides a base for a certain
 *        wiring model. Wiring is the part that defines what C++ class member
 *        functions should be called using the values from the input arrays.
 *        Usually the user needs to define his own wiring model, because it is
 *        highly dependent on the implementation of the C++ class interface that
 *        is going to be exposed to Python. But for simple C++ class interfaces
 *        some wiring model are pre-defined. For example if all input values
 *        feed one C++ class member function and returning one output value.
 */
template <
      class MappingModel
    , class Class
    , class WiringModelConfig
>
struct base_wiring_model
  : wiring_model_type
{
    typedef base_wiring_model<MappingModel, Class, WiringModelConfig>
            base_wiring_model_t;

    typedef MappingModel
            mapping_model_t;

    typedef Class
            class_t;

    typedef WiringModelConfig
            wiring_model_config_t;

    // The config_ variable holds the configuration of the wiring model.
    wiring_model_config_t const config_;

    //__________________________________________________________________________
    base_wiring_model(wiring_model_config_t const & config)
      : config_(config)
    {}

    //__________________________________________________________________________
    wiring_model_config_t const &
    GetConfig() const
    {
        return config_;
    }
};

}/*namespace wiring*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_WIRING_HPP_INCLUDED
