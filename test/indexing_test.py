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

if(__name__ == "__main__"):
    unittest.main()