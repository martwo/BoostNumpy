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
#include <vector>

#include <boost/python.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>

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
     *
     *  @note: The dtype object returned is a reference to the singleton dtype
     *         object defined by numpy. One must not change the properties of
     *         this object. In case a brand new object is needed, the static
     *         function new_builtin<typename T>() should be used.
     */
    template <typename T>
    static dtype get_builtin();

    /**
     * @brief Creates a new dtype object with the same properties as the
     *        numpy's builtin dtype object corresponding to the scalar C++ type.
     */
    template <typename T>
    static dtype new_builtin();

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

    /**
     * @brief Checks if this data type object has fields associated.
     */
    bool has_fields() const;

    /**
     * @brief Checks if this data type object describes a flexible data type,
     *        i.e. it is of type NPY_STRING, NPY_UNICODE, or NPY_VOID.
     */
    bool is_flexible() const;

    /**
     * @brief Checks if this dtype object describes an array of values of type
     *        subdtype.
     */
    bool is_array() const;

    /**
     * @brief Returns the kind of the array.
     */
    char get_char() const;

    /**
     * @brief Returns the dtype's type number.
     */
    int get_type_num() const;

    /**
     * @brief Returns the dtype object of the array values this dtype object
     *        describes. In case this dtype object describes no array of values,
     *        a bare boost::python::object object (i.e. None) is returned.
     */
    python::object get_subdtype() const;

    /**
     * @brief Returns the shape of this dtype object as a python tuble. The
     *        returned tuple has only entries, if this dtype object describes
     *        a C-style contiguous array of type subdtype, i.e. if the
     *        ``is_array`` method returns ``true``.
     */
    python::tuple get_shape() const;

    std::vector<intptr_t>
    get_shape_vector() const;

    /**
     * @brief Returns a Python tuple containting the names of the fields in the
     *        order they have been added to this dtype object. If the
     *        ``has_fields`` method returns ``false``, this method returns an
     *        empty tuple.
     */
    python::tuple get_field_names() const;

    /**
     * @brief Returns the dtype object of the field having the specified name.
     *        This method must not be called when the ``has_fields`` method
     *        returns ``false``.
     */
    dtype get_field_dtype(python::str const & field_name) const;

    /**
     * @brief Returns the byte offset of the specified field from the beginning
     *        of the ndarray's item address.
     */
    intptr_t get_field_byte_offset(python::str const & field_name) const;

    /**
     * @brief Returns a vector containing the byte offsets of all the fields
     *        defined for this dtype object. The offsets are sorted ascending,
     *        i.e. by the order the fields appear in the ndarrays data.
     */
    std::vector<intptr_t> get_fields_byte_offsets() const;

    /**
     * @brief Adds the field named ``name`` having the data type ``dt`` to the
     *        end of this dtype object. This increases the itemsize of this
     *        dtype object automatically by the size of the field's dtype.
     */
    void add_field(std::string const & name, dtype const & dt);
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

template <typename T>
struct is_intish
{
    BOOST_STATIC_CONSTANT(bool, value =
        (::boost::type_traits::ice_or<
            ::boost::is_integral<T>::value,
            ::boost::is_enum<T>::value
         >::value));
};

//______________________________________________________________________________
template <typename T, bool is_integral>
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

template <>
struct builtin_dtype<void, false>
{
    static dtype get();
};

template <>
struct builtin_dtype<boost::python::object, false>
{
    static dtype get();
};

dtype construct_new_dtype(int type_num);

}// namespace detail

template <typename T>
dtype
dtype::
get_builtin()
{
    return detail::builtin_dtype<T, detail::is_intish<T>::value>::get();
}

template <typename T>
dtype
dtype::
new_builtin()
{
    return detail::construct_new_dtype(get_builtin<T>().get_type_num());
}

}// namespace numpy
}// namespace boost

BOOST_NUMPY_OBJECT_MANAGER_TRAITS(boost::numpy::dtype);

#endif // !BOOST_NUMPY_DTYPE_HPP_INCLUDED
