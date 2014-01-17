/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/detail/pygil.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
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

//==============================================================================
class PyGIL
{
  public:
    //__________________________________________________________________________
    PyGIL(bool auto_acquire=true)
    {
        if(auto_acquire)
        {
            acquire();
        }
    }

    //__________________________________________________________________________
    ~PyGIL()
    {
        if(is_acquired())
        {
            release();
        }
    }

    //__________________________________________________________________________
    void
    acquire()
    {
        state_ = PyGILState_Ensure();
        acquired_ = true;
    }

    //__________________________________________________________________________
    void
    release()
    {
        PyGILState_Release(state_);
        acquired_ = false;
    }

    //__________________________________________________________________________
    bool
    is_acquired() const
    {
        return acquired_;
    }

  private:
    bool acquired_;
    PyGILState_STATE state_;
};

}/*detail*/
}/*numpy*/
}/*boost*/

#endif // !BOOST_NUMPY_DETAIL_PYGIL_HPP_INCLUDED
