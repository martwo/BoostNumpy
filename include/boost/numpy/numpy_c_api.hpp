/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * @file boost/numpy.hpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <martin.wolf@icecube.wisc.edu>
 * @brief This file includes the numpy C API for usage by boost::numpy.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_NUMPY_C_API_HPP_INCLUDED
#define BOOST_NUMPY_NUMPY_C_API_HPP_INCLUDED

#include <Python.h>

// If this header file is only included for declaration purposes the numpy
// python modules do not need to be imported. This should be done only once
// per shared object library. For boost::numpy this is numpy.cpp.
#if !defined(BOOST_NUMPY_INTERNAL_IMPL_MAIN)
    #define NO_IMPORT_ARRAY
//    #define NO_IMPORT_UFUNC
#endif

// Do not use the deprecated old numpy API if there is the new numpy API
// (as of v1.7) available.
#include <numpy/numpyconfig.h>
#ifdef NPY_1_7_API_VERSION
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

#include <numpy/arrayobject.h>
// Note: The numpy/__ufunc_api.h header file included by numpy/ufuncobject.h
//       has probably a bug: The "static int _import_umath(void)" function is
//       not put into an "#if !defined(NO_IMPORT_UFUNC) && !defined(NO_IMPORT)"
//       condition statement causing an unused function compiler warning when
//       including numpy/ufuncobject.h into more than one internal source file.
//
//       Since we do not need it yet, we just do not include it at all.
//#include <numpy/ufuncobject.h>

// Include a header file that defines the new numpy constants as of numpy
// version 1.7.
#if NPY_FEATURE_VERSION < 0x00000007
#include <boost/numpy/post_v1_7_constants.hpp>
#endif

#endif // !BOOST_NUMPY_NUMPY_C_API_HPP_INCLUDED
