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
        dstream_test_module.unary_to_void__explmapping__double(a)
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
        a1 = np.arange(0,100000, dtype=np.float64)
        a2 = np.arange(0,100000, dtype=np.float64)*3.42

        dstream_test_module.binary_to_void__double(a1, a2)
        dstream_test_module.binary_to_void__explmapping__double(a1, a2)
        dstream_test_module.binary_to_void__allow_threads__double(a1, a2, nthreads=3)
        dstream_test_module.binary_to_void__min_thread_size__double(a1, a2, nthreads=3)

        r = a1*a2

        o = dstream_test_module.binary_to_T_mult__double(a1, a2)
        self.assertTrue((o == r).all())

        o = dstream_test_module.binary_to_T_mult__explmapping__double(a1, a2)
        self.assertTrue((o == r).all())

        o = dstream_test_module.binary_to_T_mult__allow_threads__double(a1, a2, nthreads=3)
        self.assertTrue((o == r).all())

        o = dstream_test_module.binary_to_T_mult__min_thread_size__double(a1, a2, nthreads=3)
        self.assertTrue((o == r).all())

        t = dstream_test_module.binary_to_vectorT__tuple__double(a1, a2)
        self.assertTrue((t[0] == a1).all())
        self.assertTrue((t[1] == a2).all())

        # Test the out keyword argument.
        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.binary_to_T_mult__double(a1, a2, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.binary_to_T_mult__allow_threads__double(a1, a2, nthreads=3, out=o)
        self.assertTrue((o == r).all())

        o = np.empty((100000,), dtype=np.float64)
        dstream_test_module.binary_to_T_mult__min_thread_size__double(a1, a2, nthreads=3, out=o)
        self.assertTrue((o == r).all())

        t = (np.empty((100000,), dtype=np.float64),
             np.empty((100000,), dtype=np.float64))
        dstream_test_module.binary_to_vectorT__tuple__double(a1, a2, out=t)
        self.assertTrue((t[0] == a1).all())
        self.assertTrue((t[1] == a2).all())

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

        a1 = np.arange(0,100000, dtype=np.float64)
        a2 = np.arange(0,100000, dtype=np.float64)*3.42

        testclass.binary_to_void__double(a1, a2)
        testclass.binary_to_void__allow_threads__double(a1, a2, nthreads=3)
        testclass.binary_to_void__min_thread_size__double(a1, a2, nthreads=3)

        r = a1*a2

        o = testclass.binary_to_T_mult__double(a1, a2)
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
