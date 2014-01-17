/**
 * $Id$
 *
 * Copyright (C)
 * 2013
 *     Martin Wolf <martin.wolf@icecube.wisc.edu>
 *     and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * \file    boost/numpy/dstream/dshape.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <martin.wolf@icecube.wisc.edu>
 *
 * \brief This file defines templates for data shape of a data stream.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DSTREAM_DSHAPE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_DSHAPE_HPP_INCLUDED

#include <stdint.h>

#include <vector>

#include <boost/mpl/int.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/at.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/numpy/dtype.hpp>
#include <boost/numpy/ndarray.hpp>
#include <boost/numpy/dstream/limits.hpp>
#include <boost/numpy/mpl/as_std_vector.hpp>
#include <boost/numpy/mpl/types.hpp>

namespace boost {
namespace numpy {
namespace dstream {

namespace detail {

template <class ShapeVec>
struct dshape_shape_base
{
    typedef ShapeVec              vec;
    typedef boost::mpl::size<vec> nd;

    //__________________________________________________________________________
    /**
     * \brief Extracts the data shape from the MPL vector and returns it as a
     *     std::vector<T> object.
     */
    template <class T>
    inline static
    std::vector<T>
    as_std_vector()
    {
        return boost::numpy::mpl::as_std_vector<T, vec, nd::value>::apply();
    }

    //__________________________________________________________________________
    /**
     * \brief Extract the idx'th shape of the data shape MPL vector.
     */
    template <int idx>
    inline static
    intptr_t
    shape()
    {
        return boost::mpl::at<vec, boost::mpl::int_<idx> >::type::value;
    }
};

template <class T>
struct dshape_value_base
{
    typedef T value_type;
};

}/*namespace detail*/

//______________________________________________________________________________
/**
 * \brief The boost::numpy::dstream::dshape template provides a type for a
 *        particular ndarray shape, without the first axis, specifying the shape
 *        of the data per iteration (over the first axis).
 *
 *        The template parameter ShapeVec is a mpl::vector of mpl::int_ types,
 *        specifying the data shape of the ndarray.
 */
template <class ShapeVec, class T>
struct dshape
  : detail::dshape_shape_base<ShapeVec>
  , detail::dshape_value_base<T>
{};

//------------------------------------------------------------------------------
// Typedef the scalar data shape as scalar_dshape.
template <class T>
struct scalar_dshape
  : dshape<boost::mpl::vector<>, T>
{};

typedef scalar_dshape<void> void_dshape;

namespace utils {

//______________________________________________________________________________
/**
 * \brief Creates a boost::numpy::ndarray object from the
 *     given boost::python::object object. The dimension of the returned
 *     ndarray object is anything between 0 and the length of the data shape
 *     vector plus one (for the first axis).
 */
template <class DShape>
struct bpobj_to_ndarray
{
    static
    ndarray apply(python::object const & obj)
    {
        dtype const dt = dtype::get_builtin< typename DShape::value_type >();
        return from_object(obj, dt, 0, DShape::nd::value+1, ndarray::ALIGNED);
    }
};

//______________________________________________________________________________
/**
 * \brief Completes the ndarray shape based on a given data shape type.
 */
template <class DShape, int ND>
struct complete_shape;

//------------------------------------------------------------------------------
// Specialize the complete_shape class for each possible data shape dimension.
#define BOOST_PP_DEF(z, n, data)                                               \
    shape.push_back(DShape::template shape<n>());

#define BOOST_PP_LOCAL_LIMITS (0, BOOST_NUMPY_DSTREAM_LIMIT_DSHAPE_ND)
#define BOOST_PP_LOCAL_MACRO(n)                                                \
    template <class DShape>                                                    \
    struct complete_shape<DShape, n>                                           \
    {                                                                          \
        inline static                                                          \
        void apply(std::vector<intptr_t> & shape)                              \
        {                                                                      \
            BOOST_PP_REPEAT(n, BOOST_PP_DEF, ~)                                \
        }                                                                      \
    };
#include BOOST_PP_LOCAL_ITERATE()

#undef BOOST_PP_DEF
//------------------------------------------------------------------------------

}/*namespace utils*/

}/*namespace dstream*/
}/*namespace numpy*/
}/*namespace boost*/

#endif // !BOOST_NUMPY_DSTREAM_DSHAPE_HPP_INCLUDED
