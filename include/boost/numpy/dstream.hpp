/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines templates for data stream functionalty.
 *        A data stream (dstream) describes how input data (via several input
 *        ndarrays) will be mapped to output data (usually one output ndarray)
 *        via a particular "mapping model", and what C++ methods are used to
 *        generate the output based on the input, managed by the so called
 *        "wiring model".
 *        The iteration over the stream (i.e. the arrays) is done over the
 *        first axis of all the ndarrays.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_HPP_INCLUDED

// Include the none output array transformation template, so the user does not
// need to include it by hand.
#include <boost/numpy/dstream/out_arr_transforms/none.hpp>

// Notes about output array transforms:
//     Output array transformation templates are used to transform the output
//     array after doing the data stream iteration and before returning the
//     output array to the user. They are defined within the
//     boost::numpy::dstream::out_arr_transform namespace.
//     The following transform templates are pre-defined:
//
//     none
//         If the output array should not be transformed at all.
//
//     scalarize
//         If the output array should be transformed into a numpy scalar
//         if it is a zero-dimensional ndarray or a one-dimensional ndarray
//         holding only one element.

#include <boost/numpy/dstream/callable.hpp>
#include <boost/numpy/dstream/class_method_pp_ui.hpp>
#include <boost/numpy/dstream/function_pp_ui.hpp>

#endif // !BOOST_NUMPY_DSTREAM_HPP_INCLUDED
