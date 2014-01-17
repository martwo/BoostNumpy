/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @file    boost/numpy/dstream/threading.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <martin.wolf@fysik.su.se>
 *
 * @brief This file defines a helper mechanism to extract parameters for the def
 *        and classdef function templates.
 *        It uses the tuple_extract code from boost/python/detail/def_helper.hpp
 *        and defines a boost::numpy specific def_helper template.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_THREADING_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_THREADING_HPP_INCLUDED

#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>

#include <boost/numpy/dstream/defaults.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace threading {

struct thread_ability_selector_type
{};

// We derive the thread_ability_type from the thread_ability_selector_type
// because a thread ability can of course always select itself as thread
// ability.
// This allows to specify either a thread ability selector or a thread ability
// to the def/classdef functions.
struct thread_ability_type
  : thread_ability_selector_type
{};

}/*namespace threading*/

template <bool b>
struct threads_allowed
  : boost::mpl::bool_<b>
{};

template <int i>
struct min_n_tasks_per_thread
  : boost::mpl::int_<i>
{};

template <class ThreadsAllowed, class MinNTasksPerThread>
struct thread_ability
  : threading::thread_ability_type
{
    typedef thread_ability<ThreadsAllowed, MinNTasksPerThread>
            type;

    typedef ThreadsAllowed
            threads_allowed_t;

    typedef MinNTasksPerThread
            min_n_tasks_per_thread_t;
};

struct no_threads
  : threading::thread_ability_type
{
    typedef thread_ability<threads_allowed<false>, min_n_tasks_per_thread<0> >
            type;
};

template <int n>
struct min_thread_chunck_size
  : threading::thread_ability_type
{
    typedef thread_ability<threads_allowed<true>, min_n_tasks_per_thread<n> >
            type;
};

struct default_thread_ability
  : threading::thread_ability_type
{
    typedef thread_ability
            < threads_allowed<true>
            , min_n_tasks_per_thread<BOOST_NUMPY_DSTREAM_DEFAULT_MIN_N_TASKS_PER_THREAD>
            >
            type;
};

}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_THREADING_HPP_INCLUDED
