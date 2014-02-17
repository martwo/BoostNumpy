/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \file    boost/numpy/dstream/detail/core_shape.hpp
 * \version $Revision$
 * \date    $Date$
 * \author  Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * \brief This file defines the
 *        boost::numpy::dstream::detail::core_shape::nd<ND>::shape template
 *        that provides a description of the core shape of dimensionality ND for
 *        an array. The maximal number of core dimensions is given by
 *        BOOST_MPL_LIMIT_VECTOR_SIZE.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#if !defined(BOOST_PP_IS_ITERATING)

#ifndef BOOST_NUMPY_DSTREAM_CORE_SHAPE_HPP_INCLUDED
#define BOOST_NUMPY_DSTREAM_CORE_SHAPE_HPP_INCLUDED

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/iteration/local.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/equal_to.hpp>

#include <boost/type_traits/is_base_of.hpp>

#include <boost/utility/enable_if.hpp>

#include <boost/numpy/limits.hpp>
#include <boost/numpy/mpl/unspecified.hpp>
#include <boost/numpy/mpl/as_std_vector.hpp>

namespace boost {
namespace numpy {
namespace dstream {
namespace detail {
namespace core_shape {

// Define core shape dimension names with pre-defined dimension ids.
namespace dim {

BOOST_STATIC_CONSTANT(int, A = -1);
BOOST_STATIC_CONSTANT(int, B = -2);
BOOST_STATIC_CONSTANT(int, C = -3);
BOOST_STATIC_CONSTANT(int, D = -4);
BOOST_STATIC_CONSTANT(int, E = -5);
BOOST_STATIC_CONSTANT(int, F = -6);
BOOST_STATIC_CONSTANT(int, G = -7);
BOOST_STATIC_CONSTANT(int, H = -8);
BOOST_STATIC_CONSTANT(int, I = -9);
BOOST_STATIC_CONSTANT(int, J = -10);
BOOST_STATIC_CONSTANT(int, K = -11);
BOOST_STATIC_CONSTANT(int, L = -12);
BOOST_STATIC_CONSTANT(int, M = -13);
BOOST_STATIC_CONSTANT(int, N = -14);
BOOST_STATIC_CONSTANT(int, O = -15);
BOOST_STATIC_CONSTANT(int, P = -16);
BOOST_STATIC_CONSTANT(int, Q = -17);
BOOST_STATIC_CONSTANT(int, R = -18);
BOOST_STATIC_CONSTANT(int, S = -19);
BOOST_STATIC_CONSTANT(int, T = -20);
BOOST_STATIC_CONSTANT(int, U = -21);
BOOST_STATIC_CONSTANT(int, V = -22);
BOOST_STATIC_CONSTANT(int, W = -23);
BOOST_STATIC_CONSTANT(int, X = -24);
BOOST_STATIC_CONSTANT(int, Y = -25);
BOOST_STATIC_CONSTANT(int, Z = -26);

}// namespace dim

namespace detail {

struct core_shape_type
{};

template <class T>
struct is_core_shape
{
    typedef typename boost::mpl::if_<
              is_base_of<core_shape_type, T>
            , boost::mpl::true_
            , boost::mpl::false_
            >::type
            type;
    BOOST_STATIC_CONSTANT(bool, value = type::value);
};

// The shape_vec_t type is a boost::mpl::vector of boost::mpl::int_ types,
// specifying the shape of the core dimensions of the ndarray.
// By convention, positive values specify fixed length
// dimensions. Negative values specify variable sized dimensions. The
// negative value specifies the key of that dimensions. Dimensions with the
// same key must have the same length or must have a size of one, so it can
// be broadcasted to the length of the other dimensions with the same key.
template <class ShapeVec>
struct core_shape_base
  : core_shape_type
{
    typedef ShapeVec shape_vec_t;
    typedef boost::mpl::size<shape_vec_t> nd;

    //__________________________________________________________________________
    /**
     * \brief Extracts the core shape from the MPL vector and returns it as a
     *     std::vector<int> object.
     */
    inline
    static
    std::vector<int>
    as_std_vector()
    {
        return boost::numpy::mpl::as_std_vector<int, shape_vec_t, nd::value>::apply();
    }

    //__________________________________________________________________________
    /**
     * \brief Extracts the idx'th id of the core shape description MPL vector.
     */
    inline
    static
    int
    id(int idx)
    {
        return as_std_vector()[idx];
    }
};

struct core_shape_tuple_type
{};

template <class T>
struct is_core_shape_tuple
{
    typedef typename boost::mpl::if_<
              is_base_of<core_shape_tuple_type, T>
            , boost::mpl::true_
            , boost::mpl::false_
            >::type
            type;
    BOOST_STATIC_CONSTANT(bool, value = type::value);
};

template <class T, int LEN>
struct is_len_of
{
    typedef typename boost::mpl::if_<
              boost::mpl::equal_to< typename T::len, boost::mpl::int_<LEN> >
            , boost::mpl::true_
            , boost::mpl::false_
            >::type
            type;
    BOOST_STATIC_CONSTANT(bool, value = type::value);
};

template <int LEN>
struct core_shape_tuple_base
  : core_shape_tuple_type
{
    typedef boost::mpl::int_<LEN> len;
};

template <int LEN>
struct core_shape_tuple;

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY, <boost/numpy/dstream/detail/core_shape.hpp>, 1))
#include BOOST_PP_ITERATE()

}// namespace detail

template <int ND>
struct nd {};

template <>
struct nd<0>
{
    template <class Key = numpy::mpl::unspecified>
    struct shape
      : detail::core_shape_base< boost::mpl::vector<> >
    {
        typedef shape<>
                type;
    };
};

#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (1, BOOST_NUMPY_LIMIT_CORE_SHAPE_ND, <boost/numpy/dstream/detail/core_shape.hpp>, 2))
#include BOOST_PP_ITERATE()

// Construct a core_shape_tuple of length 2 out of two core_shape types.
template <class CoreShapeLHS, class CoreShapeRHS>
typename boost::lazy_enable_if<
    boost::mpl::and_<detail::is_core_shape<CoreShapeLHS>
                   , detail::is_core_shape<CoreShapeRHS> >
  , detail::core_shape_tuple<2>::core_shapes<CoreShapeLHS, CoreShapeRHS>
>::type
operator,(CoreShapeLHS const &, CoreShapeRHS const &)
{
    std::cout << "Creating cshape_tuple<2> type" << std::endl;
    return detail::core_shape_tuple<2>::core_shapes<CoreShapeLHS, CoreShapeRHS>();
}

template <int LEN>
struct make_core_shape_tuple;

// Construct core_shape_tuples of length 3 and above.
#define BOOST_PP_ITERATION_PARAMS_1                                            \
    (4, (3, BOOST_NUMPY_LIMIT_INPUT_AND_OUTPUT_ARITY, <boost/numpy/dstream/detail/core_shape.hpp>, 3))
#include BOOST_PP_ITERATE()

// Construct a mapping definition from two core_shape_tuple types.
template <class InCoreShapeTuple, class OutCoreShapeTuple>
typename boost::enable_if<
    boost::mpl::and_< detail::is_core_shape_tuple<InCoreShapeTuple>
                    , detail::is_core_shape_tuple<OutCoreShapeTuple>
    >
    //, void//mapping::make_definition<OutCoreShapeTuple, InCoreShapeTuple>
>::type
operator>>(InCoreShapeTuple const &, OutCoreShapeTuple const &)
{
    //return typename mapping::make_definition<OutCoreShapeTuple, InCoreShapeTuple>::type();
}

}// namespace core_shape
}// namespace detail
}// namespace dstream
}// namespace numpy
}// namespace boost

#endif // !BOOST_NUMPY_DSTREAM_CORE_SHAPE_HPP_INCLUDED
#else

#define N BOOST_PP_ITERATION()

#if BOOST_PP_ITERATION_FLAGS() == 1

template <>
struct core_shape_tuple<N>
{
    template <BOOST_PP_ENUM_PARAMS_Z(1, N, class CoreShape)>
    struct core_shapes
      : core_shape_tuple_base<N>
    {
        typedef core_shapes<BOOST_PP_ENUM_PARAMS_Z(1, N, CoreShape)>
                type;

        #define BOOST_PP_LOCAL_MACRO(n) \
            typedef BOOST_PP_CAT(CoreShape,n) BOOST_PP_CAT(core_shape_type_,n);
        #define BOOST_PP_LOCAL_LIMITS (0, BOOST_PP_SUB(N,1))
        #include BOOST_PP_LOCAL_ITERATE()
        #undef BOOST_PP_LOCAL_LIMITS
        #undef BOOST_PP_LOCAL_MACRO
    };
};

#elif BOOST_PP_ITERATION_FLAGS() == 2

template <>
struct nd<N>
{
    #define BOOST_NUMPY_DEF(z, n, data) \
        BOOST_PP_COMMA_IF(n) boost::mpl::int_< BOOST_PP_CAT(Key,n) >
    template <BOOST_PP_ENUM_PARAMS_Z(1, N, int Key)>
    struct shape
      : detail::core_shape_base< boost::mpl::vector< BOOST_PP_REPEAT(N, BOOST_NUMPY_DEF, ~) > >
    {
        typedef shape<BOOST_PP_ENUM_PARAMS_Z(1, N, Key)>
                type;
    };
    #undef BOOST_NUMPY_DEF
};

#elif BOOST_PP_ITERATION_FLAGS() == 3

template <>
struct make_core_shape_tuple<N>
{
    template <class CoreShapeTuple, class CoreShape>
    struct impl
    {
        typedef detail::core_shape_tuple<N>::core_shapes<BOOST_PP_ENUM_PARAMS_Z(1, BOOST_PP_SUB(N,1), typename CoreShapeTuple::core_shape_type_), CoreShape>
                type;
    };
};

// Construct a cshape_tuple of length N out of a core shape tuple type of length
// N-1 and one core shape type.
template <class CoreShapeTuple, class CoreShape>
typename boost::lazy_enable_if<
    boost::mpl::and_<detail::is_core_shape_tuple<CoreShapeTuple>
                   , detail::is_len_of<CoreShapeTuple, BOOST_PP_SUB(N,1)>
                   , detail::is_core_shape<CoreShape> >
  , make_core_shape_tuple<N>::impl<CoreShapeTuple,CoreShape>
>::type
operator,(CoreShapeTuple const &, CoreShape const &)
{
    std::cout << "Creating core_shape_tuple<"<<N<<"> type" << std::endl;
    return typename make_core_shape_tuple<N>::impl<CoreShapeTuple,CoreShape>::type();
}

#endif

#undef N

#endif // BOOST_PP_IS_ITERATING
