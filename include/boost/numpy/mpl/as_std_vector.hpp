/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/mpl/as_std_vector.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines the as_std_vector MPL utility function for creating
 *     a std::vector object for a given MPL vector, assuming that the values of
 *     the types that are hold by the MPL vector all matches the std::vector
 *     value type.
 *
 *     This file is distributed under the Boost Software License,
 *     Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *     http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_MPL_AS_STD_VECTOR_HPP_INCLUDED
#define BOOST_NUMPY_MPL_AS_STD_VECTOR_HPP_INCLUDED

#include <vector>

#include <boost/mpl/limits/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

namespace boost {
namespace numpy {
namespace mpl {

template<class T, class MPLVector, int Size>
struct as_std_vector
{};

//______________________________________________________________________________
// Template specialization for an empty MPL vector.
template<class T, class MPLVector>
struct as_std_vector<T, MPLVector, 0>
{
    typedef MPLVector      mpl_vector_t;
    typedef std::vector<T> std_vector_t;

    static std_vector_t
    apply()
    {
        return std_vector_t();
    }
};

//______________________________________________________________________________
// Template specialization for a MPL vector of size 1.
template<class T, class MPLVector>
struct as_std_vector<T, MPLVector, 1>
{
    typedef MPLVector      mpl_vector_t;
    typedef std::vector<T> std_vector_t;

    static std_vector_t
    apply()
    {
        return std_vector_t(1, boost::mpl::at<mpl_vector_t, boost::mpl::int_<0> >::type::value);
    }
};

//______________________________________________________________________________
// Template specialization for a MPL vector of size greater than or equal 2.
#define BOOST_PP_DEF(z, n, data)                                               \
    std_vector.push_back(boost::mpl::at<mpl_vector_t, boost::mpl::int_<n> >::type::value);
#define BOOST_PP_LOCAL_LIMITS (2, BOOST_MPL_LIMIT_VECTOR_SIZE)
#define BOOST_PP_LOCAL_MACRO(n)                                                \
    template<class T, class MPLVector>                                         \
    struct as_std_vector<T, MPLVector, n>                                      \
    {                                                                          \
        typedef MPLVector      mpl_vector_t;                                   \
        typedef std::vector<T> std_vector_t;                                   \
                                                                               \
        inline static std_vector_t                                             \
        apply()                                                                \
        {                                                                      \
            std_vector_t std_vector(n);                                        \
            BOOST_PP_REPEAT(n, BOOST_PP_DEF, ~)                                \
            return std_vector;                                                 \
        }                                                                      \
    };
#include BOOST_PP_LOCAL_ITERATE()
#undef BOOST_PP_DEF

}/*mpl*/
}/*numpy*/
}/*boost*/

#endif // !BOOST_NUMPY_MPL_AS_STD_VECTOR_HPP_INCLUDED
