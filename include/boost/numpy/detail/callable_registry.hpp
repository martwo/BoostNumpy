/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * @file    boost/numpy/detail/registry.hpp
 * @version $Revision$
 * @date    $Date$
 * @author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * @brief This file defines the registry functionality for boost::numpy.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 *
 * FIXME: This file is deprecated!!
 */
#ifndef BOOST_NUMPY_DETAIL_REGISTRY_HPP_INCLUDED
#define BOOST_NUMPY_DETAIL_REGISTRY_HPP_INCLUDED

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/mpl/push_back.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>

#include <boost/numpy/pp.hpp>

namespace boost {
namespace numpy {
namespace detail {
namespace registry {

// The registry_vector is declared static in order to allocate memory for it in
// each source file seperately.
static std::vector< boost::shared_ptr<void> > registry_vector;

typedef boost::mpl::vector<> empty_registry_index_vec_t;

template <class RV>
struct add_index
{
    typedef add_index<RV> type;
    typedef boost::mpl::push_back<RV, BOOST_NUMPY_PP_MPL_VOID> apply;
};

struct none_registry
{
    typedef none_registry type;
    typedef empty_registry_index_vec_t index_vec_t;
    typedef boost::mpl::int_<-1> index_t;
};

template <class RV>
struct registry
{
    typedef registry<RV> type;
    typedef typename add_index<RV>::apply index_vec_t;
    typedef boost::mpl::int_<boost::mpl::size<index_vec_t>::type::value - 1> index_t;

    //__________________________________________________________________________
    static
    void
    add_item(boost::shared_ptr<void> const & ptr, std::string const & errmsg)
    {
        // Raise an exception if the insertion into the registry_vector
        // would be at the wrong place. Unfortunately, this can only be done at
        // runtime but will be done, when the python module is imported. So
        // programming errors can
        if(index_t::value != registry_vector.size())
        {
            std::string s = "boost::numpy: "
                "The registry index does not fit the size of the registry "
                "vector. ";
            s += errmsg;
            PyErr_SetString(PyExc_RuntimeError, s.c_str());
            python::throw_error_already_set();
        }

        registry_vector.push_back(ptr);
    }

    //__________________________________________________________________________
    template <class T>
    static
    T const &
    get_item()
    {
        return *reinterpret_cast<T*>(registry_vector[index_t::value].get());
    }
};

}/*namespace registry*/
}/*namespace detail*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DETAIL_REGISTRY_HPP_INCLUDED
