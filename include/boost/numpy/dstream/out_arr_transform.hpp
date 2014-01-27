/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/out_arr_transform.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines the out_arr_transform base template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORM_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORM_HPP_INCLUDED

namespace boost {
namespace numpy {
namespace dstream {
namespace out_arr_transforms {

struct out_arr_transform_type
{};

struct out_arr_transform_selector_type
{};

template <class MappingModel>
struct out_arr_transform_base
  : out_arr_transform_type
{
    BOOST_STATIC_CONSTANT(int, in_arity = MappingModel::in_arity);

    typedef MappingModel
            mapping_model_t;
};

}/*namespace out_arr_transforms*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_OUT_ARR_TRANSFORM_HPP_INCLUDED
