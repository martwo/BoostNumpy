/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/mapping/detail/definition.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines templates for mapping definitions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_DEFINITION_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_DEFINITION_HPP_INCLUDED

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numpy/dstream/mapping/detail/in.hpp>
#include <boost/numpy/dstream/mapping/detail/out.hpp>
#include <boost/numpy/dstream/mapping/detail/core_shape.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace mapping {
namespace detail {

template <
      class OutMapping
    , class InMapping
>
struct definition
{
    typedef OutMapping out;
    typedef InMapping in;

    typedef typename boost::mpl::if_c<
              out::arity
            , boost::mpl::false_
            , boost::mpl::true_
            >::type
            maps_to_void_t;
    BOOST_STATIC_CONSTANT(bool, maps_to_void = maps_to_void_t::value);
};

template <class OutCoreShapeTuple, class InCoreShapeTuple>
struct make_definition
{
    typedef typename make_out_mapping<OutCoreShapeTuple::len::value>::template impl<OutCoreShapeTuple>::type
            out_mapping_t;
    typedef typename make_in_mapping<InCoreShapeTuple::len::value>::template impl<InCoreShapeTuple>::type
            in_mapping_t;
    typedef definition<out_mapping_t, in_mapping_t>
            type;
};

// Construct a mapping definition from two core_shape_tuple types.
template <class InCoreShapeTuple, class OutCoreShapeTuple>
typename boost::lazy_enable_if<
    boost::mpl::and_< is_core_shape_tuple<InCoreShapeTuple>
                    , is_core_shape_tuple<OutCoreShapeTuple>
    >
  , make_definition<OutCoreShapeTuple, InCoreShapeTuple>
>::type
operator>>(InCoreShapeTuple const &, OutCoreShapeTuple const &)
{
    std::cout << "Creating definition type" << std::endl;
    return typename make_definition<OutCoreShapeTuple, InCoreShapeTuple>::type();
}

}// namespace detail
}// namespace mapping
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_MAPPING_DETAIL_DEFINITION_HPP_INCLUDED
