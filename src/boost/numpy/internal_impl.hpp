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
 * \file   boost/numpy/internal_impl.hpp
 * \author Martin Wolf <martin.wolf@icecube.wisc.edu>,
 *         Jim Bosch
 * \brief This header file is included FIRST by all internal implementation
 *        source files of the boost::numpy library itself.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_INTERNAL_IMPL_HPP_INCLUDED
#define BOOST_NUMPY_INTERNAL_IMPL_HPP_INCLUDED

#if !defined(BOOST_NUMPY_INTERNAL_IMPL)
    ERROR_internal_impl_hpp_is_for_internal_source_file_usage_only
#endif // !BOOST_NUMPY_INTERNAL_IMPL

#define PY_ARRAY_UNIQUE_SYMBOL BOOST_NUMPY_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL BOOST_NUMPY_UFUNC_API

#endif // ! BOOST_NUMPY_INTERNAL_IMPL_HPP_INCLUDED
