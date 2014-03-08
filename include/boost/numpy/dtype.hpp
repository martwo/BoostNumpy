/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 * 2010-2012
 *     Jim Bosch
 *
 * @file boost/numpy/dtype.hpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>,
 *         Jim Bosch
 * @brief This file defines the boost::numpy::dtype object manager and some
 *        builtin type convertion between C++ and numpy types.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#ifndef BOOST_NUMPY_DTYPE_HPP_INCLUDED
#define BOOST_NUMPY_DTYPE_HPP_INCLUDED

#include <complex>

#include <boost/python.hpp>

#include <boost/numpy/object_manager_traits.hpp>

namespace boost {
namespace numpy {

/**
 *  @brief A boost.python "object manager" (subclass of object) for numpy.dtype.
 *
 *  @todo This could have a lot more interesting accessors.
 */
class dtype : public python::object
{
  private:
    static python::detail::new_reference convert(python::object const & arg, bool align);

  public:
    /**
     *  @brief Compare two dtypes for equivalence.
     *
     *  This is more permissive than equality tests.  For instance, if long and
     *  int are the same size, the dtypes corresponding to each will be
     *  equivalent, but not equal.
     */
    static bool equivalent(dtype const & a, dtype const & b);

    /**
     *  @brief Register from-Python converters for NumPy's built-in array scalar
     *         types.
     *
     *  This is usually called automatically by initialize(), and shouldn't be
     *  called twice (doing so just adds unused converters to the Boost.Python
     *  registry).
     */
    static void register_scalar_converters();

    /**
     *  @brief Get the built-in numpy dtype associated with the scalar
     *         type given through the template parameter.
     *
     *  This is perhaps the most useful part of the numpy API: it returns the
     *  dtype object corresponding to a built-in C++ type. This should work for
     *  any integer or floating point type supported by numpy, and will also
     *  work for std::complex if sizeof(std::complex<T>) == 2*sizeof(T).
     *
     *  It can also be useful for users to add explicit specializations for
     *  POD structs that return field-based dtypes.
     */
    template <typename T>
    static dtype get_builtin();

    BOOST_PYTHON_FORWARD_OBJECT_CONSTRUCTORS(dtype, python::object);

    /**
     * @brief Convert an arbitrary Python object to a data-type descriptor
     *        object.
     */
    template <typename T>
    explicit dtype(T arg, bool align=false)
      : python::object(convert(arg, align))
    {}

    /**
     * @brief Return the size of the data type in bytes.
     */
    int get_itemsize() const;
};

namespace detail {

//______________________________________________________________________________
template <int bits, bool is_unsigned>
dtype
get_int_dtype();

//______________________________________________________________________________
template <int bits>
dtype
get_float_dtype();

//______________________________________________________________________________
template <int bits>
dtype
get_complex_dtype();

//______________________________________________________________________________
template <typename T, bool is_integral=boost::is_integral<T>::value>
struct builtin_dtype;

template <typename T>
struct builtin_dtype<T, true>
{
    static dtype get()
    {
        return get_int_dtype< 8*sizeof(T), boost::is_unsigned<T>::value >();
    }
};

template <typename T>
struct builtin_dtype<T, false>
{
    static dtype get()
    {
        return get_float_dtype< 8*sizeof(T) >();
    }
};

template <typename T>
struct builtin_dtype< std::complex<T>, false >
{
    static dtype get()
    {
        return get_complex_dtype< 16*sizeof(T) >();
    }
};

template <>
struct builtin_dtype<bool, true>
{
    static dtype get();
};

}// namespace detail

template <typename T>
dtype
dtype::
get_builtin() { return detail::builtin_dtype<T>::get(); }

}// namespace numpy
}// namespace boost

BOOST_NUMPY_OBJECT_MANAGER_TRAITS(boost::numpy::dtype);

#endif // !BOOST_NUMPY_DTYPE_HPP_INCLUDED
