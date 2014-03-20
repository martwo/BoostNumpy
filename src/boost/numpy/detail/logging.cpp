/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/detail/logging.cpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file implements the basic logging functionalities of
 *        boost::numpy.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#include <iostream>
#include <string>

namespace boost {
namespace numpy {
namespace detail {

void
log(
      std::string const & file
    , int line
    , std::string const & func
    , std::string const & msg)
{
#ifndef NDEBUG
    std::cout << "DEBUG: In file " << file << ", line "<< line << ", in function " << func << ": " << msg << std::endl;
#endif
}

}// namespace detail
}// namespace numpy
}// namespace boost
