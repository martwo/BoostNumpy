/**
 * $Id$
 *
 * Copyright (C)
 * 2015 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/python/make_tuple_from_container.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines the boost::python::make_tuple_from_container
 *        template for constructing a boost::python::tuple object from a given
 *        STL container, that implements the iterator interface.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_PYTHON_MAKE_TUPLE_FROM_CONTAINER_HPP_INCLUDED
#define BOOST_NUMPY_PYTHON_MAKE_TUPLE_FROM_CONTAINER_HPP_INCLUDED

#include <iterator>

#include <boost/python.hpp>
#include <boost/python/tuple.hpp>

namespace boost {
namespace python {

template <typename IterT>
tuple
make_tuple_from_container(
    IterT const & begin
  , IterT const & end
)
{
    typename IterT::difference_type const n = std::distance(begin, end);
    if(n < 0)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "The distance between the begin and end iterators must be >=0!");
        throw_error_already_set();
    }

    tuple result((detail::new_reference)::PyTuple_New((n>=0 ? n : -n)));
    IterT it(begin);
    for(size_t idx=0; it!=end; ++idx, ++it)
    {
        PyTuple_SET_ITEM(result.ptr(), idx, incref(object(*it).ptr()));
    }

    return result;
}

}//namespace python
}//namespace boost

#endif // ! BOOST_NUMPY_PYTHON_MAKE_TUPLE_FROM_CONTAINER_HPP_INCLUDED
