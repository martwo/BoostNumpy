/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/pp.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines some preprocessor macros for boost::numpy. These
 *        are usually used for code repetitions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_PP_HPP_INCLUDED
#define BOOST_NUMPY_PP_HPP_INCLUDED

#include <boost/mpl/void.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/preprocessor/seq/cat.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <boost/numpy/types.hpp>

//______________________________________________________________________________
#define BOOST_NUMPY_PP_OB <
#define BOOST_NUMPY_PP_CB >

//______________________________________________________________________________
#define BOOST_NUMPY_PP_MPL_VOID boost::mpl::void_::type

//______________________________________________________________________________
#define BOOST_NUMPY_PP_REPEAT_DATA(z, n, data)                                 \
    data

//______________________________________________________________________________
#define BOOST_NUMPY_PP_REPEAT_DATA_AS_LIST(n, data)                            \
    BOOST_PP_ENUM_TRAILING(n, BOOST_NUMPY_PP_REPEAT_DATA, data)

//______________________________________________________________________________
#define BOOST_NUMPY_PP_MPL_VOID_LIST(n)                                        \
    BOOST_NUMPY_PP_REPEAT_DATA_AS_LIST(n, BOOST_NUMPY_PP_MPL_VOID)

//______________________________________________________________________________
#define BOOST_NUMPY_PP_COMMA_IF_LIST(n, data)                                  \
    BOOST_PP_COMMA_IF(n) BOOST_PP_ENUM(n, BOOST_NUMPY_PP_REPEAT_DATA, data)

//______________________________________________________________________________
#define BOOST_NUMPY_PP_ENUM_PARAMS_TRAILING_IF(n, data)                        \
    BOOST_PP_COMMA_IF(n) BOOST_PP_ENUM_PARAMS(n, data)

//______________________________________________________________________________
#define BOOST_NUMPY_PP_SEQ_TO_STR(seq)                                         \
    BOOST_PP_STRINGIZE(BOOST_PP_SEQ_CAT(seq))

#endif // !BOOST_NUMPY_PP_HPP_INCLUDED
