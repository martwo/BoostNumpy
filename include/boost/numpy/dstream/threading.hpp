/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file    boost/numpy/dstream/threading.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @brief This file defines types for describing the thread ability of a
 *        generalized universal function.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_THREADING_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_THREADING_HPP_INCLUDED

#include <boost/mpl/bool.hpp>
#include <boost/mpl/integral_c.hpp>

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
// This allows to specify either a thread ability selector or an actual thread
// ability to the dstream::def function.
struct thread_ability_type
  : thread_ability_selector_type
{};

template <bool b>
struct threads_allowed
  : boost::mpl::bool_<b>
{};

template <unsigned n>
struct min_n_tasks_per_thread
  : boost::mpl::integral_c<unsigned, n>
{};

}// namespace threading

template <
      class ThreadsAllowed
    , class MinNTasksPerThread = threading::min_n_tasks_per_thread<BOOST_NUMPY_DSTREAM_DEFAULT_MIN_N_TASKS_PER_THREAD>
>
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

template <unsigned n>
struct min_thread_size
  : threading::thread_ability_selector_type
{
    typedef thread_ability<
                  threading::threads_allowed<true>
                , threading::min_n_tasks_per_thread<n>
            >
            type;
};

struct allow_threads
  : threading::thread_ability_selector_type
{
    typedef thread_ability<
                  threading::threads_allowed<true>
            >
            type;
};

struct no_threads
  : threading::thread_ability_selector_type
{
    typedef thread_ability<
                  threading::threads_allowed<false>
                , threading::min_n_tasks_per_thread<0>
            >
            type;
};

// By default, the GU functions should not have thread ability.
struct default_thread_ability
  : threading::thread_ability_selector_type
{
    typedef no_threads::type
            type;
};

}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_THREADING_HPP_INCLUDED
