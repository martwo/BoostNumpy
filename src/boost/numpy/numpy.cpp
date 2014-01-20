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
 * @file    boost/numpy/numpy.cpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <martin.wolf@icecube.wisc.edu>,
 *          Jim Bosch
 *
 * @brief This file implements the boost::numpy utility functions.
 *        The numpy C-API is also living in this object file.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#define BOOST_NUMPY_INTERNAL_IMPL
#define BOOST_NUMPY_INTERNAL_IMPL_MAIN
#include <boost/numpy/internal_impl.hpp>

#include <boost/numpy/numpy_c_api.hpp>
#include <boost/numpy/dtype.hpp>

#include <boost/assert.hpp>
#include <boost/version.hpp>

namespace boost {
namespace numpy {

void initialize(bool register_scalar_converters)
{
    // Check if the boost version is at least 1.38. For previous versions this
    // library has not been tested.
    BOOST_ASSERT(BOOST_VERSION >= 103800);

    // Check if the numpy version is at least 1.6. We don't support prior
    // versions because they have an other iterator API.
    BOOST_ASSERT(NPY_FEATURE_VERSION >= 0x00000006);

    import_array();
    //import_ufunc();

    if(register_scalar_converters)
    {
        dtype::register_scalar_converters();
    }
}

}/*numpy*/
}/*boost*/
