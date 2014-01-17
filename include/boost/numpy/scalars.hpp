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
 * @file boost/numpy/scalars.hpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <martin.wolf@icecube.wisc.edu>
 * @brief This file defines numpy array scalar types in boost::python.
 *
 *        Supported numpy array scalar types:
 *
 *            - numpy.void
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_SCALARS_HPP_INCLUDED
#define BOOST_NUMPY_SCALARS_HPP_INCLUDED

#include <boost/python.hpp>
#include <boost/numpy/object_manager_traits.hpp>
#include <boost/numpy/dtype.hpp>

// TODO: rename file to void_.hpp

namespace boost {
namespace numpy {

/**
 *  @brief A boost::python "object manager" (subclass of object) for numpy.void.
 */
class void_ : public python::object
{
  public:
    /**
     *  @brief Construct a new array scalar with the given size and void dtype.
     *
     *  Data is initialized to zero.  One can create a standalone scalar object
     *  with a certain dtype "dt" with:
     *  @code
     *  void_ scalar = void_(dt.get_itemsize()).view(dt);
     *  @endcode
     */
     explicit void_(Py_ssize_t size);

     BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(void_, object);

     /// @brief Return a view of the scalar with the given dtype.
     void_ view(dtype const & dt) const;

     /// @brief Copy the scalar (deep for all non-object fields).
     void_ copy() const;
};

}/*numpy*/
}/*boost*/

BOOST_NUMPY_OBJECT_MANAGER_TRAITS(boost::numpy::void_);

#endif // !BOOST_NUMPY_SCALARS_HPP_INCLUDED
