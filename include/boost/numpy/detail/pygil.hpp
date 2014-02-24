/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/detail/pygil.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the PyGIL class for holding the python global
 *        interpreter lock. Thanks to Steve Jackson for the code inspiration.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DETAIL_PYGIL_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_PYGIL_HPP_INCLUDED

#include <boost/python.hpp>

namespace boost {
namespace numpy {
namespace detail {

class PyGIL
{
  public:
    PyGIL(bool auto_acquire=true)
    {
        if(auto_acquire)
        {
            acquire();
        }
    }

    ~PyGIL()
    {
        if(is_acquired())
        {
            release();
        }
    }

    void
    acquire()
    {
        state_ = PyGILState_Ensure();
        acquired_ = true;
    }

    void
    release()
    {
        PyGILState_Release(state_);
        acquired_ = false;
    }

    bool
    is_acquired() const
    {
        return acquired_;
    }

  private:
    bool acquired_;
    PyGILState_STATE state_;
};

}// namespace detail
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DETAIL_PYGIL_HPP_INCLUDED
