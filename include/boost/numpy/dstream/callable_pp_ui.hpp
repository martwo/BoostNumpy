/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/callable_pp_ui.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines pre-processor macros for a nicer user interface to
 *        the dstream templates for exposing functions or class member
 *        functions.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_CALLABLE_PP_UI_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_CALLABLE_PP_UI_HPP_INCLUDED

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/comma_if.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/size.hpp>
#include <boost/preprocessor/seq/transform.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/pp.hpp>

#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__0 '0'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__1 '1'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__2 '2'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__3 '3'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__4 '4'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__5 '5'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__6 '6'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__7 '7'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__8 '8'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__9 '9'

#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__A 'A'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__B 'B'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__C 'C'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__D 'D'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__E 'E'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__F 'F'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__G 'G'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__H 'H'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__I 'I'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__J 'J'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__K 'K'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__L 'L'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__M 'M'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__N 'N'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__O 'O'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__P 'P'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__Q 'Q'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__R 'R'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__S 'S'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__T 'T'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__U 'U'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__V 'V'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__W 'W'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__X 'X'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__Y 'Y'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__Z 'Z'

#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__a 'a'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__b 'b'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__c 'c'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__d 'd'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__e 'e'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__f 'f'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__g 'g'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__h 'h'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__i 'i'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__j 'j'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__k 'k'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__l 'l'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__m 'm'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__n 'n'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__o 'o'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__p 'p'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__q 'q'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__r 'r'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__s 's'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__t 't'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__u 'u'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__v 'v'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__w 'w'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__x 'x'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__y 'y'
#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__z 'z'

#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR___ '_'

#define BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__none '~'

//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_CAT_SEQ_BY_COMMA_IF(z, n, seq)                     \
    BOOST_PP_COMMA_IF(n) BOOST_PP_SEQ_ELEM(n, seq)
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_CHAR_TO_INT(s, data, el)             \
    BOOST_PP_CAT(BOOST_NUMPY_DSTREAM_CONST_CHAR_EXPR__, el)
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_CHARS_TO_INTS(seq)                   \
    BOOST_PP_SEQ_TRANSFORM(BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_CHAR_TO_INT, BOOST_PP_EMPTY, seq)
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_CFG_PREFIX ( boost::numpy::detail::cfg() =
#define BOOST_NUMPY_DSTREAM_CFG_POSTFIX )
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_SETTING_TO_CFG(s, data, el)          \
    BOOST_NUMPY_DSTREAM_CFG_PREFIX el BOOST_NUMPY_DSTREAM_CFG_POSTFIX
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_SETTINGS_TO_CFGS(seq)                \
    BOOST_PP_SEQ_TRANSFORM(BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_SETTING_TO_CFG, BOOST_PP_EMPTY, seq)
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_SEQ_ELEM_NONE (none)
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_SEQ_FILLUP_WITH_NONE(seq, n_max)                   \
    seq BOOST_PP_REPEAT_FROM_TO(                                               \
            BOOST_PP_SEQ_SIZE(seq), n_max,                                     \
            BOOST_NUMPY_PP_REPEAT_DATA, BOOST_NUMPY_DSTREAM_SEQ_ELEM_NONE      \
        )
//______________________________________________________________________________
#define BOOST_NUMPY_DSTREAM_MAKE_CONFIG_ID(_func_name_seq)                     \
    boost::numpy::detail::config_id<                                           \
        BOOST_PP_REPEAT(                                                       \
              BOOST_NUMPY_LIMIT_MAX_FUNC_NAME_LENGTH                           \
            , BOOST_NUMPY_DSTREAM_CAT_SEQ_BY_COMMA_IF                          \
            , BOOST_NUMPY_DSTREAM_TRANSFORM_SEQ_CHARS_TO_INTS(                 \
                  BOOST_NUMPY_DSTREAM_SEQ_FILLUP_WITH_NONE(                    \
                        _func_name_seq                                         \
                      , BOOST_NUMPY_LIMIT_MAX_FUNC_NAME_LENGTH                 \
                  )                                                            \
              )                                                                \
        )                                                                      \
    >

#endif // !BOOST_NUMPY_DSTREAM_CALLABLE_PP_UI_HPP_INCLUDED
