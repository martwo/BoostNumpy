/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/wiring.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines templates for data stream wiring functionalty.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_WIRING_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_WIRING_HPP_INCLUDED

#include <boost/utility/enable_if.hpp>

#include <boost/numpy/dstream/wiring/default_wiring_model_selector_fwd.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace wiring {

struct wiring_model_type
{};

struct wiring_model_selector_type
{};

struct null_wiring_model_selector
  : wiring_model_selector_type
{};

// This wiring model selector selects the null_wiring_model_selector which is
// just a place holder when the MappingDefinition is unspecified.
template <class MappingDefinition, class FTypes>
struct default_wiring_model_selector<
    MappingDefinition
  , FTypes
  , typename enable_if<
        typename boost::is_same<MappingDefinition, numpy::mpl::unspecified>::type
    >::type
>
{
    typedef null_wiring_model_selector
            type;
};

//==============================================================================
/**
 * @brief The wiring_model_base template provides a base for a certain
 *        wiring model. Wiring is the part that defines what C++ (member)
 *        function should be called using the values from the input arrays.
 *        Usually the user needs to define his own wiring model, because it is
 *        highly dependent on the implementation of the C++ function that
 *        is going to be exposed to Python. But for simple C++ functions
 *        some wiring model are pre-defined. For example if all input values
 *        feed one C++ (member) function and returning one output value.
 */
template <
      class MappingDefinition
    , class FTypes
>
struct wiring_model_base
  : wiring_model_type
{
    typedef wiring_model_base<MappingDefinition, FTypes>
            type;

    typedef MappingDefinition
            mapping_definition_t;

    typedef FTypes
            f_types_t;
};

}// namespace wiring
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_WIRING_HPP_INCLUDED
