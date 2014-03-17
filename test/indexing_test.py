#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <boostnumpy@martin-wolf.org>
# 2010-2012
#     Jim Bosch, Ankit Daftery
#
# This file implements tests for indexing functionalities of the
# boost::numpy::ndarray class.
#
# This file is distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt).
#
import indexing_test_module
import unittest
import numpy as np

class TestIndexing(unittest.TestCase):

    def test_single(self):
        x = np.arange(0,10)
        for i in range(0,10):
            self.assertEqual(indexing_test_module.single(x,i), i)
        for i in range(-10,0):
            self.assertEqual(indexing_test_module.single(x,i),10+i)

    def test_slice(self):
        x = np.arange(0,10)
        sl = slice(3,8)
        v = np.array([3,4,5,6,7])
        self.assertTrue((indexing_test_module.slice(x, sl) == v).all())

    def test_step_slice(self):
        x = np.arange(0,10)
        sl = slice(3,8,2)
        v = np.array([3,5,7])
        self.assertTrue((indexing_test_module.slice(x, sl) == v).all())

    def test_index_array(self):
        x = np.arange(0,10)
        v = np.array([3,4,5,6])
        self.assertTrue((indexing_test_module.index_array(x, v) == v).all())
        v = np.array([[0,1],[2,3]])
        self.assertTrue((indexing_test_module.index_array(x, v) == v).all())
        v = np.array([5,6,7,8,9])
        self.assertTrue((indexing_test_module.index_array(x, x>4) == v).all())

    def test_index_array_2d(self):
        x = np.arange(0,9).reshape(3,3)
        idxarr1 = np.array([0,1])
        idxarr2 = np.array([0,2])
        v = np.array([0,5])
        self.assertTrue((indexing_test_module.index_array_2d(x, idxarr1, idxarr2) == v).all())

    def test_index_array_slice(self):
        x = np.arange(0,9).reshape(3,3)
        idxarr = np.array([0,2])
        sl = slice(1,3)
        v = np.array([[1,2],[7,8]])
        self.assertTrue((indexing_test_module.index_array_slice(x, idxarr, sl) == v).all())

if(__name__ == "__main__"):
    unittest.main()