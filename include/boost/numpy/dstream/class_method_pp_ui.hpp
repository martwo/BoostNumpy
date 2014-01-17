/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/class_method_pp_ui.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines pre-processor macros for a nicer user interface to
 *        the dstream templates for exposing class member functions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_CLASS_METHOD_PP_UI_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_CLASS_METHOD_PP_UI_HPP_INCLUDED

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/seq/size.hpp>

#include <boost/numpy/pp.hpp>
#include <boost/numpy/dstream/callable.hpp>
#include <boost/numpy/dstream/callable_pp_ui.hpp>
#include <boost/numpy/dstream/defaults.hpp>
#include <boost/numpy/dstream/out_arr_transforms/squeeze_first_axis_if_single_input_and_scalarize.hpp>

//______________________________________________________________________________
/**
 * \brief This macro defines an easy to use interface to the
 *     boost::numpy::dstream::callable template together with
 *     boost::python::class_.def.
 *     It can be used as so:
 *     \code
 *     .def(
 *         BOOST_NUMPY_DSTREAM_CLASS_METHOD_ADVANCED(
 *               (m)(y)(_)(f)(u)(n)(c)
 *             , MyClass
 *             , TheMappingModel
 *             , TheWiringModel
 *             , (setting1)(setting2)(...)
 *             , OutArrTransform
 *             , OutType,(InType1)(InType2)(...)
 *             , "out",("arg1")("arg2")(...)
 *             , BOOST_NUMPY_DSTREAM_DEFAULT_MIN_N_TASKS_PER_THREAD
 *         )
 *         , "This is the docstring.")
 *     \endcode
 */
#define BOOST_NUMPY_DSTREAM_CLASS_METHOD_ADVANCED(                             \
      _func_name_seq                                                           \
    , _cls                                                                     \
    , _mapping_model                                                           \
    , _wiring_model, _cfg_setting_seq                                          \
    , _out_arr_transform                                                       \
    , _out_type, _in_type_seq                                                  \
    , _out_name, _in_name_seq                                                  \
    , _min_n_tasks_per_thread)                                                 \
                                                                               \
      BOOST_NUMPY_PP_SEQ_TO_STR(_func_name_seq)                                \
    , boost::numpy::dstream::callable<                                         \
          _cls                                                                 \
        , _mapping_model                                                       \
        , _wiring_model                                                        \
        , _out_arr_transform                                                   \
        , BOOST_NUMPY_DSTREAM_MAKE_CONFIG_ID(_func_name_seq)                   \
        , _out_type                                                            \
        , BOOST_PP_REPEAT(BOOST_PP_SEQ_SIZE(_in_type_seq), BOOST_NUMPY_DSTREAM_CAT_SEQ_BY_COMMA_IF, _in_type_seq)\
        >::make(                                                               \
              BOOST_PP_REPEAT(BOOST_PP_SEQ_SIZE(_cfg_setting_seq), BOOST_NUMPY_DSTREAM_CAT_SEQ_BY_COMMA_IF, BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_SETTINGS_TO_CFGS(_cfg_setting_seq))\
            , _out_name                                                        \
            , BOOST_PP_REPEAT(BOOST_PP_SEQ_SIZE(_in_name_seq), BOOST_NUMPY_DSTREAM_CAT_SEQ_BY_COMMA_IF, _in_name_seq)\
            , _min_n_tasks_per_thread                                          \
           )

//______________________________________________________________________________
/**
 * \brief This macro is a simplified version of the
 *     BOOST_NUMPY_DSTREAM_CLASS_METHOD_ADVANCED macro.
 *     It uses squeeze_first_axis_if_single_input_and_scalarize as output
 *     array transformation and
 *     BOOST_NUMPY_DSTREAM_DEFAULT_MIN_N_TASKS_PER_THREAD for the minimum number
 *     of tasks per thread.
 *     It can be used as so:
 *     \code
 *     .def(
 *         BOOST_NUMPY_DSTREAM_CLASS_METHOD(
 *               (m)(y)(_)(f)(u)(n)(c)
 *             , MyClass
 *             , TheMappingModel
 *             , TheWiringModel
 *             , (setting1)(setting2)(...)
 *             , OutType,(InType1)(InType2)(...)
 *             , "out",("arg1")("arg2")(...)
 *         )
 *         , "This is the docstring.")
 *     \endcode
 */
#define BOOST_NUMPY_DSTREAM_CLASS_METHOD(                                      \
      _func_name_seq                                                           \
    , _cls                                                                     \
    , _mapping_model                                                           \
    , _wiring_model, _cfg_setting_seq                                          \
    , _out_type, _in_type_seq                                                  \
    , _out_name, _in_name_seq)                                                 \
                                                                               \
    BOOST_NUMPY_DSTREAM_CLASS_METHOD_ADVANCED(                                 \
          _func_name_seq                                                       \
        , _cls                                                                 \
        , _mapping_model                                                       \
        , _wiring_model, _cfg_setting_seq                                      \
        , boost::numpy::dstream::out_arr_transform::squeeze_first_axis_if_single_input_and_scalarize\
        , _out_type, _in_type_seq                                              \
        , _out_name, _in_name_seq                                              \
        , BOOST_NUMPY_DSTREAM_DEFAULT_MIN_N_TASKS_PER_THREAD                   \
        )

#endif // !BOOST_NUMPY_DSTREAM_CLASS_METHOD_PP_UI_HPP_INCLUDED
