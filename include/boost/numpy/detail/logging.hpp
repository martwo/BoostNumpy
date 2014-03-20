/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/detail/logging.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines basic logging facilities for boost::numpy.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_LOGGING_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_LOGGING_HPP_INCLUDED

#include <string>
#include <sstream>

namespace boost {
namespace numpy {
namespace detail {

void
log(
      std::string const & file
    , int line
    , std::string const & func
    , std::string const & msg
);

}// namespace detail
}// namespace numpy
}// namespace boost

#define BOOST_NUMPY_DETAIL_STREAM_LOGGER(file, line, func, msg)                \
    {                                                                          \
        std::ostringstream oss;                                                \
        oss << msg;                                                            \
        boost::numpy::detail::log(file, line, func, oss.str());                \
    }

#ifdef NDEBUG
    #define BOOST_NUMPY_LOG(msg)
#else
    #define BOOST_NUMPY_LOG(msg) \
        BOOST_NUMPY_DETAIL_STREAM_LOGGER(__FILE__, __LINE__, __PRETTY_FUNCTION__, msg)
#endif

#endif // BOOST_NUMPY_DETAIL_LOGGING_HPP_INCLUDED
