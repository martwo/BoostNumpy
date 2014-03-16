#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <boostnumpy@martin-wolf.org>
# 2010-2012
#     Jim Bosch, Ankit Daftery
#
# This file implements tests for boost::numpy::ndarray class.
#
# This file is distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt).
#
import ndarray_test_module
import unittest
import numpy as np

class TestNdarray(unittest.TestCase):

    all_types = (
        np.bool8,
        np.int8, np.int16, np.int32,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128
    )

    def test_zeros(self):
        for dtp in self.all_types:
            v = np.zeros(60, dtype=dtp)
            dt = np.dtype(dtp)
            for shape in ((60,),(6,10),(4,3,5),(2,2,3,5)):
                a1 = ndarray_test_module.zeros(shape, dt)
                a2 = v.reshape(a1.shape)
                self.assertEqual(shape, a1.shape)
                self.assert_((a1 == a2).all())

    def test_zeros_as_matrix(self):
        for dtp in self.all_types:
            dt = np.dtype(dtp)
            shape = (6, 10)
            v = np.matrix(np.zeros(shape, dtype=dtp))
            a1 = ndarray_test_module.zeros_as_matrix(shape, dt)
            self.assertEqual(shape, a1.shape)
            self.assert_((a1 == v).all())
            self.assertEqual(type(a1), type(v))

    def test_array(self):
        a = range(0,60)[:]
        for dtp in self.all_types:
            # Skip bool type, because BOOL == VALUE gives always false.
            if(dtp == np.bool8):
                continue
            v = np.array(a, dtype=dtp)
            dt = np.dtype(dtp)
            a1 = ndarray_test_module.array(a)
            a2 = ndarray_test_module.array(a, dt)

            self.assert_((a1 == v).all())
            self.assert_((a2 == v).all())
            for shape in ((60,),(6,10),(4,3,5),(2,2,3,5)):
                a1 = a1.reshape(shape)
                a2 = a2.reshape(shape)
                self.assertEqual(shape, a1.shape)
                self.assertEqual(shape, a2.shape)

    def test_empty(self):
        for dtp in self.all_types:
            dt = np.dtype(dtp)
            for shape in ((60,),(6,10),(4,3,5),(2,2,3,5)):
                a1 = ndarray_test_module.empty(shape, dt)
                self.assertEqual(shape, a1.shape)

    def test_transpose(self):
        for dtp in self.all_types:
            dt = np.dtype(dtp)
            for shape in ((6,10),(4,3,5),(2,2,3,5)):
                a = np.empty(shape, dt)
                v = a.transpose()
                a1 = ndarray_test_module.transpose(a)
                self.assertEqual(a1.shape, v.shape)

    def test_squeeze(self):
        a = np.array([[[3,4,5]]])
        v = a.squeeze()
        a1 = ndarray_test_module.squeeze(a)
        self.assertEqual(a1.shape, v.shape)

    def test_reshape(self):
        a = np.empty((2,2))
        a1 = ndarray_test_module.reshape(a, (1,4))
        self.assertEqual(a1.shape, (1,4))

if(__name__ == "__main__"):
    unittest.main()
