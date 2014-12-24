/**
 * $Id$
 *
 * Copyright (C)
 * 2013 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 * 2010-2012
 *     Jim Bosch
 *
 * @file boost/numpy/dtype.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>,
 *         Jim Bosch
 * @brief This file implements the boost::numpy::dtype object manager.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#define BOOST_NUMPY_INTERNAL_IMPL
#include <boost/numpy/internal_impl.hpp>

#include <iostream>

#include <boost/python/dict.hpp>

#include <boost/numpy/numpy_c_api.hpp>
#include <boost/numpy/object_manager_traits_impl.hpp>
#include <boost/numpy/dtype.hpp>

BOOST_NUMPY_OBJECT_MANAGER_TRAITS_IMPL(boost::numpy::dtype, PyArrayDescr_Type);

namespace boost {
namespace numpy {

//______________________________________________________________________________
python::detail::new_reference
dtype::
convert(python::object const & arg, bool align)
{
    PyArray_Descr* obj=NULL;
    if(align) {
        if(PyArray_DescrAlignConverter(arg.ptr(), &obj) < 0) {
            python::throw_error_already_set();
        }
    }
    else
    {
        if(PyArray_DescrConverter(arg.ptr(), &obj) < 0) {
            python::throw_error_already_set();
        }
    }
    return python::detail::new_reference(reinterpret_cast<PyObject*>(obj));
}

//______________________________________________________________________________
int
dtype::
get_itemsize() const
{
    return reinterpret_cast<PyArray_Descr*>(ptr())->elsize;
}
//______________________________________________________________________________
bool
dtype::
has_fields() const
{
    return PyDataType_HASFIELDS(reinterpret_cast<PyArray_Descr*>(ptr()));
}

//______________________________________________________________________________
bool
dtype::
is_flexible() const
{
    return PyDataType_ISFLEXIBLE(reinterpret_cast<PyArray_Descr*>(ptr()));
}

//______________________________________________________________________________
bool
dtype::
is_array() const
{
    return (reinterpret_cast<PyArray_Descr*>(ptr())->subarray != NULL);
}

//______________________________________________________________________________
char
dtype::
get_char() const
{
    return reinterpret_cast<PyArray_Descr*>(ptr())->kind;
}

//______________________________________________________________________________
int
dtype::
get_type_num() const
{
    return reinterpret_cast<PyArray_Descr*>(ptr())->type_num;
}

//______________________________________________________________________________
python::object
dtype::
get_subdtype() const
{
    if(reinterpret_cast<PyArray_Descr*>(ptr())->subarray == NULL)
    {
        return python::object();
    }

    return dtype(python::detail::borrowed_reference(reinterpret_cast<PyArray_Descr*>(ptr())->subarray->base));
}

//______________________________________________________________________________
python::tuple
dtype::
get_shape() const
{
    if(reinterpret_cast<PyArray_Descr*>(ptr())->subarray == NULL)
    {
        return python::make_tuple();
    }

    return python::tuple(python::detail::borrowed_reference(reinterpret_cast<PyArray_Descr*>(ptr())->subarray->shape));
}

//______________________________________________________________________________
std::vector<intptr_t>
dtype::
get_shape_vector() const
{
    python::tuple shape = get_shape();
    std::vector<intptr_t> shape_vec(python::len(shape));
    size_t const nd = shape_vec.size();
    for(size_t i=0; i<nd; ++i)
    {
        shape_vec[i] = python::extract<intptr_t>(shape[i]);
    }
    return shape_vec;
}

//______________________________________________________________________________
python::tuple
dtype::
get_field_names() const
{
    if(has_fields())
    {
        python::tuple names(python::detail::borrowed_reference(reinterpret_cast<PyArray_Descr*>(ptr())->names));
        return names;
    }
    return python::tuple();
}

//______________________________________________________________________________
dtype
dtype::
get_field_dtype(python::str const & field_name) const
{
    assert(has_fields());
    python::dict fields(python::detail::borrowed_reference(reinterpret_cast<PyArray_Descr*>(ptr())->fields));
    python::tuple field(fields.get(field_name));
    return dtype(field[0]);
}

//______________________________________________________________________________
intptr_t
dtype::
get_field_byte_offset(python::str const & field_name) const
{
    assert(has_fields());
    python::dict fields(python::detail::borrowed_reference(reinterpret_cast<PyArray_Descr*>(ptr())->fields));
    python::tuple field(fields.get(field_name));
    return python::extract<intptr_t>(field[1]);
}

//______________________________________________________________________________
std::vector<intptr_t>
dtype::
get_fields_byte_offsets() const
{
    python::tuple field_names = get_field_names();
    size_t N = python::len(field_names);
    std::vector<intptr_t> offsets(N);
    for(size_t i=0; i<N; ++i)
    {
        offsets[i] = get_field_byte_offset(python::str(field_names[i]));
    }
    return offsets;
}

//______________________________________________________________________________
void
dtype::
add_field(std::string const & name, dtype const & dt)
{
    if(!has_fields())
    {
        // Add a dictionary to the fields member.
        python::dict fields_dict;
        python::incref(fields_dict.ptr());
        reinterpret_cast<PyArray_Descr*>(ptr())->fields = fields_dict.ptr();

        python::tuple names = python::make_tuple();
        python::incref(names.ptr());
        reinterpret_cast<PyArray_Descr*>(ptr())->names = names.ptr();
    }

    python::str name_str(name);
    python::dict fields(python::detail::borrowed_reference(reinterpret_cast<PyArray_Descr*>(ptr())->fields));
    python::tuple field = python::make_tuple<dtype, intptr_t>(dt, get_itemsize());
    fields[name_str] = field;

    // Add the new field name to the names tuple member.
    // Note: We use new_reference here so the old name tuple gets PyDECREF'ed
    //       automatically, after old_names gets out of scope.
    python::tuple old_names(python::detail::new_reference(reinterpret_cast<PyArray_Descr*>(ptr())->names));
    python::list names_list(old_names);
    names_list.append(name_str);
    python::tuple new_names(names_list);
    python::incref(new_names.ptr());
    reinterpret_cast<PyArray_Descr*>(ptr())->names = new_names.ptr();

    // Increase the itemsize of this dtype object by the size of the given
    // field dtype.
    reinterpret_cast<PyArray_Descr*>(ptr())->elsize += dt.get_itemsize();
}

//______________________________________________________________________________
bool
dtype::
equivalent(dtype const & a, dtype const & b)
{
    return PyArray_EquivTypes(
        reinterpret_cast<PyArray_Descr*>(a.ptr()),
        reinterpret_cast<PyArray_Descr*>(b.ptr())
    );
}

namespace detail {

#define DTYPE_FROM_CODE(code) \
    dtype(boost::python::detail::new_reference(reinterpret_cast<PyObject*>(PyArray_DescrFromType(code))))

//______________________________________________________________________________
dtype
builtin_dtype<bool, true>::
get()
{
    return DTYPE_FROM_CODE(NPY_BOOL);
}

//______________________________________________________________________________
dtype
builtin_dtype<void, false>::
get()
{
    return DTYPE_FROM_CODE(NPY_VOID);
}

//______________________________________________________________________________
//
dtype
builtin_dtype<boost::python::object, false>::
get()
{
    return DTYPE_FROM_CODE(NPY_OBJECT);
}

//______________________________________________________________________________
template <int bits, bool is_unsigned>
struct builtin_int_dtype;

template <int bits>
struct builtin_float_dtype;

template <int bits>
struct builtin_complex_dtype;

//______________________________________________________________________________
template <int bits, bool is_unsigned>
dtype
get_int_dtype()
{
    return builtin_int_dtype<bits, is_unsigned>::get();
}

template <int bits>
dtype
get_float_dtype()
{
    return builtin_float_dtype<bits>::get();
}

template <int bits>
dtype
get_complex_dtype()
{
    return builtin_complex_dtype<bits>::get();
}

//______________________________________________________________________________
#define BUILTIN_INT_DTYPE(bits)                                                \
    template <> struct builtin_int_dtype< bits, false > {                      \
        static dtype get() { return DTYPE_FROM_CODE(NPY_INT ## bits); }        \
    };                                                                         \
    template <> struct builtin_int_dtype< bits, true > {                       \
        static dtype get() { return DTYPE_FROM_CODE(NPY_UINT ## bits); }       \
    };                                                                         \
    template dtype get_int_dtype< bits, false >();                             \
    template dtype get_int_dtype< bits, true >();

#define BUILTIN_FLOAT_DTYPE(bits)                                              \
    template <> struct builtin_float_dtype< bits > {                           \
        static dtype get() { return DTYPE_FROM_CODE(NPY_FLOAT ## bits); }      \
    };                                                                         \
    template dtype get_float_dtype< bits >();

#define BUILTIN_COMPLEX_DTYPE(bits)                                            \
    template <> struct builtin_complex_dtype< bits > {                         \
        static dtype get() { return DTYPE_FROM_CODE(NPY_COMPLEX ## bits); }    \
    };                                                                         \
    template dtype get_complex_dtype< bits >();

BUILTIN_INT_DTYPE(8);
BUILTIN_INT_DTYPE(16);
BUILTIN_INT_DTYPE(32);
BUILTIN_INT_DTYPE(64);

BUILTIN_FLOAT_DTYPE(32);
BUILTIN_FLOAT_DTYPE(64);

BUILTIN_COMPLEX_DTYPE(64);
BUILTIN_COMPLEX_DTYPE(128);

#if NPY_BITSOF_LONGDOUBLE > NPY_BITSOF_DOUBLE
template <>
struct builtin_float_dtype< NPY_BITSOF_LONGDOUBLE >
{
    static dtype get() { return DTYPE_FROM_CODE(NPY_LONGDOUBLE); }
};
template dtype get_float_dtype< NPY_BITSOF_LONGDOUBLE >();

template <>
struct builtin_complex_dtype< 2 * NPY_BITSOF_LONGDOUBLE >
{
    static dtype get() { return DTYPE_FROM_CODE(NPY_CLONGDOUBLE); }
};
template dtype get_complex_dtype< 2 * NPY_BITSOF_LONGDOUBLE >();
#endif

//______________________________________________________________________________
dtype
construct_new_dtype(int type_num)
{
    return dtype(boost::python::detail::new_reference(reinterpret_cast<PyObject*>(PyArray_DescrNewFromType(type_num))));
}

}/*detail*/

namespace {

template <typename T>
struct boost_numpy_array_scalar_converter
{
    //__________________________________________________________________________
    static
    PyTypeObject const *
    get_pytype()
    {
        // This implementation depends on the fact that get_builtin returns
        // pointers to objects numpy has declared statically, and that the
        // typeobj member also refers to a static object. That means we don't
        // need to do any reference counting.
        // In fact, I'm somewhat concerned that increasing the reference count
        // of any of these might cause leaks, because I don't think
        // boost::python ever decrements it, but it's probably a moot point if
        // everything is actually static.
        return reinterpret_cast<PyArray_Descr*>(dtype::get_builtin<T>().ptr())->typeobj;
    }

    //__________________________________________________________________________
    static
    void *
    convertible(PyObject * obj)
    {
        if(PyObject_TypeCheck(obj, const_cast<PyTypeObject*>(get_pytype())))
        {
            return obj;
        }

        return NULL;
    }

    //__________________________________________________________________________
    static
    void
    convert(PyObject * obj, boost::python::converter::rvalue_from_python_stage1_data * data)
    {
        void * storage = reinterpret_cast<boost::python::converter::rvalue_from_python_storage<T>*>(data)->storage.bytes;
        // We assume std::complex is a "standard layout" here and elsewhere;
        // not guaranteed by C++03 standard, but true in every known
        // implementation (and guaranteed by C++11).
        PyArray_ScalarAsCtype(obj, reinterpret_cast<T*>(storage));
        data->convertible = storage;
    }

    //__________________________________________________________________________
    static
    void
    declare()
    {
        boost::python::converter::registry::push_back(
            &convertible, &convert, boost::python::type_id<T>()
#ifndef BOOST_PYTHON_NO_PY_SIGNATURES
            , &get_pytype
#endif
        );
    }
};

}/*anonymous*/

//______________________________________________________________________________
void
dtype::
register_scalar_converters()
{
    boost_numpy_array_scalar_converter< bool >::declare();
    boost_numpy_array_scalar_converter< float >::declare();
    boost_numpy_array_scalar_converter< double >::declare();

#if NPY_BITSOF_LONGDOUBLE > NPY_BITSOF_DOUBLE
    boost_numpy_array_scalar_converter<long double>::declare();
    boost_numpy_array_scalar_converter< std::complex<long double> >::declare();
#endif

    boost_numpy_array_scalar_converter< npy_uint8 >::declare();
    boost_numpy_array_scalar_converter< npy_int8 >::declare();
    boost_numpy_array_scalar_converter< npy_uint16 >::declare();
    boost_numpy_array_scalar_converter< npy_int16 >::declare();
    boost_numpy_array_scalar_converter< npy_uint32 >::declare();
    boost_numpy_array_scalar_converter< npy_int32 >::declare();
    boost_numpy_array_scalar_converter< npy_uint64 >::declare();
    boost_numpy_array_scalar_converter< npy_int64 >::declare();

    boost_numpy_array_scalar_converter< std::complex<float> >::declare();
    boost_numpy_array_scalar_converter< std::complex<double> >::declare();
}

}/*namespace numpy*/
}/*namespace boost*/
