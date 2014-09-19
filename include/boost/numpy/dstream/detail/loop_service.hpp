/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/detail/loop_service.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the loop_service_arity<N>::loop_service template
 *        that provides functionalities for iterating over the loop dimensions
 *        of a set of input arrays.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_DETAIL_LOOP_SERVICE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DETAIL_LOOP_SERVICE_HPP_INCLUDED

#include <algorithm>
#include <set>
#include <sstream>
#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/numpy/detail/logging.hpp>
#include <boost/numpy/detail/max.hpp>
#include <boost/numpy/dstream/detail/input_array_service.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {

struct max_loop_shape_selector
{
    max_loop_shape_selector(std::vector<intptr_t> const & arr_shape, int const arr_loop_nd)
      : arr_shape_(arr_shape)
      , arr_loop_nd_(arr_loop_nd)
    {}

    std::vector<intptr_t>
    get_arr_loop_shape() const
    {
        std::vector<intptr_t> arr_loop_shape(arr_loop_nd_);
        std::copy(arr_shape_.begin(), arr_shape_.begin() + arr_loop_nd_, arr_loop_shape.begin());
        return arr_loop_shape;
    }

    std::vector<intptr_t> const & arr_shape_;
    int const arr_loop_nd_;
};

inline
bool operator>(max_loop_shape_selector const & lhs, max_loop_shape_selector const & rhs)
{
    return (lhs.arr_loop_nd_ > rhs.arr_loop_nd_);
}

template <int Arity>
struct loop_service_arity;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (3, (1, BOOST_NUMPY_LIMIT_INPUT_ARITY, <boost/numpy/dstream/detail/loop_service.hpp>))
#include BOOST_PP_ITERATE()

}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // ! BOOST_NUMPY_DSTREAM_DETAIL_LOOP_SERVICE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

template <>
struct loop_service_arity<N>
{
    BOOST_STATIC_CONSTANT(int, arity = N);

    template <BOOST_PP_ENUM_PARAMS_Z(1, N, class InArrDef)>
    class loop_service
    {
      public:
        //----------------------------------------------------------------------
        // Define a boost::mpl::bool_ type specifying if any input array is an
        // an object array. This information could be needed to set iterator
        // flags correctly.
        // Note: By default, boost::mpl::or_ has only a maximal arity of 5, so
        //       we have to construct a nested sequence of boost::mpl::or_<.,.>
        //       with always, two arguments.
        #define BOOST_NUMPY_DEF_pre_or(z, n, data) \
            typename boost::mpl::or_<
        #define BOOST_NUMPY_DEF_arr_dtype_is_bp_object(n) \
            boost::is_same<typename BOOST_PP_CAT(InArrDef,n)::value_type, python::object>
        #define BOOST_NUMPY_DEF_post_or(z, n, data) \
            BOOST_PP_COMMA() BOOST_NUMPY_DEF_arr_dtype_is_bp_object(BOOST_PP_ADD(n,1)) >::type
        typedef BOOST_PP_REPEAT(BOOST_PP_SUB(N,1), BOOST_NUMPY_DEF_pre_or, ~)
                BOOST_NUMPY_DEF_arr_dtype_is_bp_object(0)
                BOOST_PP_REPEAT(BOOST_PP_SUB(N,1), BOOST_NUMPY_DEF_post_or, ~)
                object_arrays_are_involved;
        #undef BOOST_NUMPY_DEF_post_or
        #undef BOOST_NUMPY_DEF_arr_dtype_is_bp_object
        #undef BOOST_NUMPY_DEF_pre_or
        //----------------------------------------------------------------------

        #define BOOST_NUMPY_DEF(z, n, data) \
            BOOST_PP_COMMA_IF(n) BOOST_PP_CAT(_in_arr_service_,n) ( BOOST_PP_CAT(in_arr_service_,n) )
        loop_service( BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, input_array_service< InArrDef, > & in_arr_service_) )
          : _is_virtual_loop(false)
          , BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
        {
            // Calculate the loop shape. It's just the biggest loop shape of all
            // individual input array loop shapes.
            #define BOOST_NUMPY_DEF(z, n, data) \
                BOOST_PP_COMMA_IF(n) max_loop_shape_selector( \
                      BOOST_PP_CAT(_in_arr_service_,n).get_arr_shape() \
                    , BOOST_PP_CAT(_in_arr_service_,n).get_arr_loop_nd() )
            _loop_shape = boost::numpy::detail::max(
                BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
            ).get_arr_loop_shape();
            #undef BOOST_NUMPY_DEF

            // Make sure, that the loop shape as at least 1 dimension with one
            // iteration.
            if(_loop_shape.size() == 0)
            {
                BOOST_NUMPY_LOG("Do a virtual loop")

                _is_virtual_loop = true;
                _loop_shape.push_back(1);
                #define BOOST_NUMPY_DEF(z, n, data) \
                    BOOST_PP_CAT(in_arr_service_,n).prepend_loop_dimension();
                BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
                #undef BOOST_NUMPY_DEF
            }

            // Set the broadcasting rules for all input arrays.
            #define BOOST_NUMPY_DEF(z, n, data) \
                BOOST_PP_CAT(_in_arr_service_,n) .set_arr_bcr(get_loop_nd());
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
            #undef BOOST_NUMPY_DEF

            // Check if the lengths of all the core dimensions of all the
            // input arrays are compatible to each other.
            // 1. Get a unique set of all used dimension ids.
            std::set<int> ids;
            #define BOOST_NUMPY_DEF(z, n, data) \
                std::vector<int> const & BOOST_PP_CAT(in_arr_core_shape_ids_,n) = BOOST_PP_CAT(in_arr_service_,n).get_arr_core_shape_ids();\
                ids.insert( BOOST_PP_CAT(in_arr_core_shape_ids_,n).begin(), BOOST_PP_CAT(in_arr_core_shape_ids_,n).end());
            BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
            #undef BOOST_NUMPY_DEF
            // 2. Loop through the dimension ids and for each id, collect the
            //    list of dimension lengths of all input arrays.
            std::set<int>::const_iterator it;
            std::set<int>::const_iterator const ids_end = ids.end();
            for(it=ids.begin(); it!=ids_end; ++it)
            {
                int const id = *it;
                if(id <= 0)
                {
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        intptr_t const BOOST_PP_CAT(len_,n) = BOOST_PP_CAT(_in_arr_service_,n).get_core_dim_len(id);
                    BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
                    #undef BOOST_NUMPY_DEF
                    // 3. Get the maximum of all dimension lengths.
                    intptr_t const max_len = numpy::detail::max(BOOST_PP_ENUM_PARAMS_Z(1, N, len_));
                    // 4. Compare if all dimension length are equal to this
                    //    maximal value or are of size 0 (i.e. not defined for
                    //    an array), or 1 (i.e. broadcast-able).
                    #define BOOST_NUMPY_DEF(z, n, data) \
                        BOOST_PP_IF(n, &&, ) (BOOST_PP_CAT(len_,n) == max_len || BOOST_PP_CAT(len_,n) == 0 || BOOST_PP_CAT(len_,n) == 1)
                    if( ! ( BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~) ) )
                    #undef BOOST_NUMPY_DEF
                    {
                        std::stringstream msg;
                        msg << "One of the variable sized array dimensions has "
                            << "the wrong length! It must be of length 1 or "
                            << max_len << "!";
                        PyErr_SetString(PyExc_ValueError, msg.str().c_str());
                        python::throw_error_already_set();
                    }
                }
            }

            // TODO: Check if the loop dimension length of all the input arrays
            //       are compatible to each other. This is just to prevend an
            //       cryptic iterator error message to the user.
        }

        inline
        int
        get_loop_nd() const
        {
            return _loop_shape.size();
        }

        inline
        std::vector<intptr_t>
        get_loop_shape() const
        {
            return _loop_shape;
        }

        intptr_t *
        get_loop_shape_data()
        {
            return &(_loop_shape.front());
        }

        intptr_t const * const
        get_loop_shape_data() const
        {
            return &(_loop_shape.front());
        }

        intptr_t
        get_loop_size() const
        {
            size_t const loop_nd = _loop_shape.size();
            intptr_t loop_size = (loop_nd == 0 ? 0 : 1);
            for(size_t i=0; i<loop_nd; ++i)
            {
                loop_size *= _loop_shape[i];
            }
            return loop_size;
        }

        /**
         * \brief Returns the maximum length of the core dimension that has the
         *     given id. All input arrays are searched and the maximum number
         *     is returned.
         *     If the given dimension id is not found for all the input array,
         *     0 will be returned.
         */
        intptr_t
        get_core_dim_len(int const id) const
        {
            return boost::numpy::detail::max(
                BOOST_PP_ENUM_BINARY_PARAMS_Z(1, N, _in_arr_service_, .get_core_dim_len(id) BOOST_PP_INTERCEPT)
            );
        }

        inline
        bool
        is_virtual_loop() const
        {
            return _is_virtual_loop;
        }

      protected:
        std::vector<intptr_t> _loop_shape;
        bool _is_virtual_loop;
        #define BOOST_NUMPY_DEF(z, n, data) \
            input_array_service< BOOST_PP_CAT(InArrDef,n) > & BOOST_PP_CAT(_in_arr_service_,n) ;
        BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~)
        #undef BOOST_NUMPY_DEF
    };
};

#undef N

#endif // BOOST_PP_IS_ITERATING
