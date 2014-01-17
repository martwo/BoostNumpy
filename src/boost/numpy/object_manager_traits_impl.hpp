/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 * 2010-2012
 *     Jim Bosch
 *
 * @file boost/numpy/object_manager_traits_impl.hpp
 * @brief Macro that defines the source-file implementation of get_pytype() for
 *        the specialized object_manager_traits struct.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL_HPP_INCLUDED
#define BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL_HPP_INCLUDED

#if !defined(BOOST_NUMPY_INTERNAL_IMPL)
    ERROR_object_manager_traits_impl_hpp_is_for_internal_source_file_usage_only
#endif // !BOOST_NUMPY_INTERNAL_IMPL

#define BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL(manager, pytypeobj)             \
namespace boost { namespace python { namespace converter {                     \
PyTypeObject const *                                                           \
object_manager_traits<manager>::get_pytype() {                                 \
    return reinterpret_cast<PyTypeObject const *>(&pytypeobj);                 \
}                                                                              \
}/*converter*/}/*python*/}/*boost*/

#define BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL__BP_OBJECT(manager, bpobj)      \
namespace boost { namespace python { namespace converter {                     \
PyTypeObject const *                                                           \
object_manager_traits<manager>::get_pytype() {                                 \
    return reinterpret_cast<PyTypeObject const *>(bpobj.ptr());                \
}                                                                              \
}/*converter*/}/*python*/}/*boost*/

#endif // !BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL_HPP_INCLUDED
