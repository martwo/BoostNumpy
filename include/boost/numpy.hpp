/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 * 2010-2012
 *     Jim Bosch
 *
 * @file boost/numpy.hpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <martin.wolf@fysik.su.se>
 * @brief This file is the main public header file for boost::numpy. It includes
 *        the boost/python.hpp header file.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_HPP_INCLUDED
#define BOOST_NUMPY_HPP_INCLUDED

#include <boost/numpy/detail/prefix.hpp>

namespace boost {
namespace numpy {

//______________________________________________________________________________
/**
 *  @brief Initialize the Numpy C-API
 *
 *  This must be called before using anything in boost::numpy;
 *  It should probably be the first line inside BOOST_PYTHON_MODULE.
 *
 *  @internal This just calls the Numpy C-API functions "import_array()"
 *            and "import_ufunc()", and then calls
 *            dtype::register_scalar_converters().
 */
void initialize(bool register_scalar_converters=true);

}/*numpy*/
}/*boost*/

#endif // !BOOST_NUMPY_HPP_INCLUDED
