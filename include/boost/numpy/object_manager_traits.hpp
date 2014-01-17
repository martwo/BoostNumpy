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
 * @file boost/numpy/object_manager_traits.hpp
 * @brief Macro that specializes object_manager_traits by requiring a
 *        source-file implementation of get_pytype().
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_OBJECT_MANAGER_TRAITS_HPP_INCLUDED
#define BOOST_NUMPY_OBJECT_MANAGER_TRAITS_HPP_INCLUDED

#define BOOST_NUMPY_OBJECT_MANAGER_TRAITS(manager)                             \
namespace boost { namespace python { namespace converter {                     \
template <>                                                                    \
struct object_manager_traits<manager>                                          \
{                                                                              \
    BOOST_STATIC_CONSTANT(bool, is_specialized = true);                        \
                                                                               \
    static inline                                                              \
    python::detail::new_reference                                              \
    adopt(PyObject* x)                                                         \
    {                                                                          \
        return python::detail::new_reference(python::pytype_check((PyTypeObject*)get_pytype(), x));\
    }                                                                          \
                                                                               \
    static                                                                     \
    bool                                                                       \
    check(PyObject* x)                                                         \
    {                                                                          \
        return ::PyObject_IsInstance(x, (PyObject*)get_pytype());              \
    }                                                                          \
                                                                               \
    static                                                                     \
    manager*                                                                   \
    checked_downcast(PyObject* x)                                              \
    {                                                                          \
        return python::downcast<manager>((checked_downcast_impl)(x, (PyTypeObject*)get_pytype()));\
    }                                                                          \
                                                                               \
    static                                                                     \
    PyTypeObject const *                                                       \
    get_pytype();                                                              \
};                                                                             \
}/*converter*/}/*python*/}/*boost*/

#endif // !BOOST_NUMPY_OBJECT_MANAGER_TRAITS_HPP_INCLUDED

