#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <boostnumpy@martin-wolf.org>
#
# This file implements tests for the boost::numpy::dstream library.
#
# This file is distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt).
#
import dstream_test_module
import unittest
import numpy as np

class TestDstream(unittest.TestCase):

    def test_unary_functions(self):
        a = np.arange(0,100000, dtype=np.float64)

        dstream_test_module.unary_to_void__double(a)
        dstream_test_module.unary_to_void__allow_threads__double(a, nthreads=3)
        dstream_test_module.unary_to_void__min_thread_size__double(a, nthreads=3)

        r = a*a

        o = dstream_test_module.unary_to_T_squared__double(a)
        self.assertTrue((o == r).all())

        o = dstream_test_module.unary_to_T_squared__explmapping__double(a)
        self.assertTrue((o == r).all())

        o = dstream_test_module.unary_to_T_squared__allow_threads__double(a, nthreads=3)
        self.assertTrue((o == r).all())

        o = dstream_test_module.unary_to_T_squared__min_thread_size__double(a, nthreads=3)
        self.assertTrue((o == r).all())

        # Test the out keyword argument.
        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.unary_to_T_squared__double(a, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.unary_to_T_squared__allow_threads__double(a, nthreads=3, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.unary_to_T_squared__min_thread_size__double(a, nthreads=3, out=o)
        self.assertTrue((o == r).all())

    def test_binary_functions(self):
        a = np.arange(0,100000, dtype=np.float64)

        dstream_test_module.binary_to_void__double(a, a)
        dstream_test_module.binary_to_void__allow_threads__double(a, a, nthreads=3)
        dstream_test_module.binary_to_void__min_thread_size__double(a, a, nthreads=3)

        r = a*a

        o = dstream_test_module.binary_to_T_mult__double(a, a)
        self.assertTrue((o == r).all())

        o = dstream_test_module.binary_to_T_mult__explmapping__double(a, a)
        self.assertTrue((o == r).all())

        o = dstream_test_module.binary_to_T_mult__allow_threads__double(a, a, nthreads=3)
        self.assertTrue((o == r).all())

        o = dstream_test_module.binary_to_T_mult__min_thread_size__double(a, a, nthreads=3)
        self.assertTrue((o == r).all())

        # Test the out keyword argument.
        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.binary_to_T_mult__double(a, a, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.binary_to_T_mult__allow_threads__double(a, a, nthreads=3, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.binary_to_T_mult__min_thread_size__double(a, a, nthreads=3, out=o)
        self.assertTrue((o == r).all())

    def test_unary_methods(self):
        testclass = dstream_test_module.TestClass()

        a = np.arange(0,100000, dtype=np.float64)

        testclass.unary_to_void__double(a)
        testclass.unary_to_void__allow_threads__double(a, nthreads=3)
        testclass.unary_to_void__min_thread_size__double(a, nthreads=3)

        r = a*a

        o = testclass.unary_to_T_squared__double(a)
        self.assertTrue((o == r).all())

        o = testclass.unary_to_T_squared__allow_threads__double(a, nthreads=3)
        self.assertTrue((o == r).all())

        o = testclass.unary_to_T_squared__min_thread_size__double(a, nthreads=3)
        self.assertTrue((o == r).all())

        # Test the out keyword argument.
        o = np.empty((100000,), dtype=np.float64)
        testclass.unary_to_T_squared__double(a, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        testclass.unary_to_T_squared__allow_threads__double(a, nthreads=3, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        testclass.unary_to_T_squared__min_thread_size__double(a, nthreads=3, out=o)
        self.assertTrue((o == r).all())

    def test_binary_methods(self):
        testclass = dstream_test_module.TestClass()

        a = np.arange(0,100000, dtype=np.float64)

        testclass.binary_to_void__double(a, a)
        testclass.binary_to_void__allow_threads__double(a, a, nthreads=3)
        testclass.binary_to_void__min_thread_size__double(a, a, nthreads=3)

        r = a*a

        o = testclass.binary_to_T_mult__double(a, a)
        self.assertTrue((o == r).all())

    def test_unary_static_methods(self):
        a = np.arange(0,100000, dtype=np.float64)

        dstream_test_module.TestClass.static_unary_to_void__double(a)

        r = a*a

        o = dstream_test_module.TestClass.static_unary_to_T_squared__double(a)
        self.assertTrue((o == r).all())

        # Test the out keyword argument.
        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.TestClass.static_unary_to_T_squared__double(a, out=o)
        self.assertTrue((o == r).all())

    def test_binary_static_methods(self):
        a = np.arange(0,100000, dtype=np.float64)

        dstream_test_module.TestClass.static_binary_to_void__double(a, a)

        r = a*a

        o = dstream_test_module.TestClass.static_binary_to_T_mult__double(a, a)
        self.assertTrue((o == r).all())

        # Test the out keyword argument.
        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.TestClass.static_binary_to_T_mult__double(a, a, out=o)
        self.assertTrue((o == r).all())

if(__name__ == "__main__"):
    unittest.main()
