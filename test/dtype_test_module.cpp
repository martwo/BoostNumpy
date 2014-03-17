/**
 * $Id$
 *
 * Copyright (C)
 * 2014 - $Date$
 *     Martin Wolf <boostnumpy@martin-wolf.org>
 * 2010-2012
 *     Jim Bosch, Ankit Daftery
 *
 * @file test/dtype_test_module.cpp
 * @version $Revision$
 * @date $Date$
 * @author Martin Wolf <boostnumpy@martin-wolf.org>
 * @brief This file implements a Python module for data type tests of the
 *        boost::numpy::dtype class.
 *
 *        This file is distributed under the Boost Software License,
 *        Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
 *        http://www.boost.org/LICENSE_1_0.txt).
 */
#include <boost/cstdint.hpp>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace test {

template <typename T>
static
bn::dtype accept(T)
{
    return bn::dtype::get_builtin<T>();
}

}// namespace test

BOOST_PYTHON_MODULE(dtype_test_module)
{
    bn::initialize();

    // Wrap dtype equivalence test, since it isn't available in Python API.
    bp::def("equivalent", bn::dtype::equivalent);

    // integers, by number of bits
    bp::def("accept_int8", &test::accept<boost::int8_t>);
    bp::def("accept_uint8", &test::accept<boost::uint8_t>);
    bp::def("accept_int16", &test::accept<boost::int16_t>);
    bp::def("accept_uint16", &test::accept<boost::uint16_t>);
    bp::def("accept_int32", &test::accept<boost::int32_t>);
    bp::def("accept_uint32", &test::accept<boost::uint32_t>);
    bp::def("accept_int64", &test::accept<boost::int64_t>);
    bp::def("accept_uint64", &test::accept<boost::uint64_t>);

    // integers, by C name according to NumPy
    bp::def("accept_bool_", &test::accept<bool>);
    bp::def("accept_byte", &test::accept<signed char>);
    bp::def("accept_ubyte", &test::accept<unsigned char>);
    bp::def("accept_short", &test::accept<short>);
    bp::def("accept_ushort", &test::accept<unsigned short>);
    bp::def("accept_intc", &test::accept<int>);
    bp::def("accept_uintc", &test::accept<unsigned int>);

    // floats and complex
    bp::def("accept_float32", &test::accept<float>);
    bp::def("accept_float64", &test::accept<double>);
    bp::def("accept_complex64", &test::accept< std::complex<float> >);
    bp::def("accept_complex128", &test::accept< std::complex<double> >);
    if(sizeof(long double) > sizeof(double))
    {
        bp::def("accept_longdouble", &test::accept<long double>);
        bp::def("accept_clongdouble", &test::accept< std::complex<long double> >);
    }
}
