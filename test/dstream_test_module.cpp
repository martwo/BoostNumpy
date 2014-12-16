/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 *
 * @file test/dstream_test_module.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>
 * @brief This file implements a Python module for testing the
 *        boost::numpy::dstream library.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#include <vector>
#include <Python.h>

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

#include <boost/numpy.hpp>
#include <boost/numpy/dstream.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;
namespace ds = boost::numpy::dstream;

namespace test {

template <typename T>
static
void
unary_to_void(T)
{}

template <typename T>
static
T
unary_to_T_squared(T v)
{
    return v*v;
}

template <typename T>
static
void
binary_to_void(T, T)
{}

template <typename T>
static
T
binary_to_T_mult(T v1, T v2)
{
    return v1*v2;
}

template <typename T>
static
std::vector<T>
binary_to_vectorT(T v1, T v2)
{
    std::vector<T> vec(2);
    vec[0] = v1;
    vec[1] = v2;
    return vec;
}

struct TestClass
{
    template <typename T>
    static
    void
    static_unary_to_void(T)
    {}

    template <typename T>
    static
    void
    static_binary_to_void(T, T)
    {}

    template <typename T>
    static
    T
    static_unary_to_T_squared(T v)
    {
        return v*v;
    }

    template <typename T>
    static
    T
    static_binary_to_T_mult(T v1, T v2)
    {
        return v1*v2;
    }

    template <typename T>
    void
    unary_to_void(T)
    {}

    template <typename T>
    T
    unary_to_T_squared(T v)
    {
        return v*v;
    }

    template <typename T>
    void
    binary_to_void(T, T)
    {}

    template <typename T>
    T
    binary_to_T_mult(T v1, T v2)
    {
        return v1*v2;
    }

    template <typename T>
    std::vector<T>
    binary_to_vectorT(T v1, T v2)
    {
        std::vector<T> vec(2);
        vec[0] = v1;
        vec[1] = v2;
        return vec;
    }
};

}// namespace test

BOOST_PYTHON_MODULE(dstream_test_module)
{
    bn::initialize();

    // Unary void-return functions.
    ds::def("unary_to_void__double", &test::unary_to_void<double>, bp::arg("v"));
    ds::def("unary_to_void__explmapping__double", &test::unary_to_void<double>, bp::arg("v")
        , (ds::scalar() >> ds::none()));
    ds::def("unary_to_void__allow_threads__double", &test::unary_to_void<double>, bp::arg("v")
        , ds::allow_threads());
    ds::def("unary_to_void__min_thread_size__double", &test::unary_to_void<double>, bp::arg("v")
        , ds::min_thread_size<32>());

    // Binary void-return functions.
    ds::def("binary_to_void__double", &test::binary_to_void<double>, (bp::args("v1"),"v2"));
    ds::def("binary_to_void__explmapping__double", &test::binary_to_void<double>, (bp::args("v1"),"v2")
        , ((ds::scalar(), ds::scalar()) >> ds::none()));
    ds::def("binary_to_void__allow_threads__double", &test::binary_to_void<double>, (bp::args("v1"),"v2")
        , ds::allow_threads());
    ds::def("binary_to_void__min_thread_size__double", &test::binary_to_void<double>, (bp::args("v1"),"v2")
        , ds::min_thread_size<32>());

    // Unary non-void-return functions.
    ds::def("unary_to_T_squared__double", &test::unary_to_T_squared<double>, bp::arg("v"));
    ds::def("unary_to_T_squared__explmapping__double", &test::unary_to_T_squared<double>, bp::arg("v")
        , (ds::scalar() >> ds::scalar()));
    ds::def("unary_to_T_squared__allow_threads__double", &test::unary_to_T_squared<double>, bp::arg("v")
        , ds::allow_threads());
    ds::def("unary_to_T_squared__min_thread_size__double", &test::unary_to_T_squared<double>, bp::arg("v")
        , ds::min_thread_size<32>());

    // Binary non-void-return functions.
    ds::def("binary_to_T_mult__double", &test::binary_to_T_mult<double>, (bp::args("v1"),"v2"));
    ds::def("binary_to_T_mult__explmapping__double", &test::binary_to_T_mult<double>, (bp::args("v1"),"v2")
        , ((ds::scalar(), ds::scalar()) >> ds::scalar()));
    ds::def("binary_to_T_mult__allow_threads__double", &test::binary_to_T_mult<double>, (bp::args("v1"),"v2")
        , ds::allow_threads());
    ds::def("binary_to_T_mult__min_thread_size__double", &test::binary_to_T_mult<double>, (bp::args("v1"),"v2")
        , ds::min_thread_size<32>());
    ds::def("binary_to_vectorT__tuple__double", &test::binary_to_vectorT<double>, (bp::args("v1"),"v2")
        , ((ds::scalar(), ds::scalar()) >> (ds::scalar(), ds::scalar())));
    ds::def("binary_to_vectorT__array__double", &test::binary_to_vectorT<double>, (bp::args("v1"),"v2")
        , ((ds::scalar(), ds::scalar()) >> ds::array<2>()));

    bp::class_<test::TestClass, boost::shared_ptr<test::TestClass> >("TestClass")
        // Unary void-return methods.
        .def(ds::method("unary_to_void__double", &test::TestClass::unary_to_void<double>, bp::arg("v")))
        .def(ds::method("unary_to_void__allow_threads__double", &test::TestClass::unary_to_void<double>, bp::arg("v")
            , ds::allow_threads()))
        .def(ds::method("unary_to_void__min_thread_size__double", &test::TestClass::unary_to_void<double>, bp::arg("v")
            , ds::min_thread_size<32>()))

        // Binary void-return methods.
        .def(ds::method("binary_to_void__double", &test::TestClass::binary_to_void<double>, (bp::args("v1"),"v2")))
        .def(ds::method("binary_to_void__allow_threads__double", &test::TestClass::binary_to_void<double>, (bp::args("v1"),"v2")
            , ds::allow_threads()))
        .def(ds::method("binary_to_void__min_thread_size__double", &test::TestClass::binary_to_void<double>, (bp::args("v1"),"v2")
            , ds::min_thread_size<32>()))

        // Unary non-void-return methods.
        .def(ds::method("unary_to_T_squared__double", &test::TestClass::unary_to_T_squared<double>, bp::arg("v")))
        .def(ds::method("unary_to_T_squared__allow_threads__double", &test::TestClass::unary_to_T_squared<double>, bp::arg("v")
            , ds::allow_threads()))
        .def(ds::method("unary_to_T_squared__min_thread_size__double", &test::TestClass::unary_to_T_squared<double>, bp::arg("v")
            , ds::min_thread_size<32>()))

        // Binary non-void-return methods.
        .def(ds::method("binary_to_T_mult__double", &test::TestClass::binary_to_T_mult<double>, (bp::args("v1"),"v2")))
        .def(ds::method("binary_to_T_mult__allow_threads__double", &test::TestClass::binary_to_T_mult<double>, (bp::args("v1"),"v2")
            , ds::allow_threads()))
        .def(ds::method("binary_to_T_mult__min_thread_size__double", &test::TestClass::binary_to_T_mult<double>, (bp::args("v1"),"v2")
            , ds::min_thread_size<32>()))
        .def(ds::method("binary_to_vectorT__tuple__double", &test::TestClass::binary_to_vectorT<double>, (bp::args("v1"),"v2")
            , ((ds::scalar(), ds::scalar()) >> (ds::scalar(), ds::scalar()))))
        .def(ds::method("binary_to_vectorT__array__double", &test::TestClass::binary_to_vectorT<double>, (bp::args("v1"),"v2")
            , ((ds::scalar(), ds::scalar()) >> ds::array<2>())))

        // Note about static methods:
        //       Since the implementation of the method call of static methods
        //       is identical to the non-static methods, we need to test only
        //       one case, i.e. testing for the correct conversion from
        //       non-static to static methods.

        // Unary static methods.
        .def(ds::staticmethod("static_unary_to_void__double", &test::TestClass::static_unary_to_void<double>, bp::arg("v")))
        .def(ds::staticmethod("static_unary_to_T_squared__double", &test::TestClass::static_unary_to_T_squared<double>, bp::arg("v")))

        // Binary static methods.
        .def(ds::staticmethod("static_binary_to_void__double", &test::TestClass::static_binary_to_void<double>, (bp::args("v1"),"v2")))
        .def(ds::staticmethod("static_binary_to_T_mult__double", &test::TestClass::static_binary_to_T_mult<double>, (bp::args("v1"),"v2")))
    ;
}
