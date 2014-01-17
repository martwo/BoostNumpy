/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \file    boost/numpy/dstream/detail/callable_base.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * \brief This file defines the base class template of all dstream callable
 *        objects.
 *        The base class template is dependent on a member function flag and a
 *        void-return flag.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_BASE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_BASE_HPP_INCLUDED

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

struct callable_type
{
    typedef callable_type
            callable_type_t;
};

template <bool is_mf>
struct callable_mf_base
  : callable_type
{
    typedef callable_mf_base<is_mf>
            callable_mf_base_t;

    BOOST_STATIC_CONSTANT(bool, is_member_function = is_mf);
};

template <bool is_member_function, bool void_return>
struct callable_base
  : callable_mf_base<is_member_function>
{
    typedef callable_base<is_member_function, void_return>
            callable_base_t;

    BOOST_STATIC_CONSTANT(bool, has_void_return = void_return);
};

}/*namespace detail*/
}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_DETAIL_CALLABLE_BASE_HPP_INCLUDED
