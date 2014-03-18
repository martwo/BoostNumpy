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
        a = np.arange(0,9, dtype=np.float64)

        dstream_test_module.unary_to_void__double(a)

        r = a*a
        o = dstream_test_module.unary_to_T_squared__double(a)
        self.assertTrue((o == r).all())

        o = np.empty((9,), dtype=np.float64)
        dstream_test_module.unary_to_T_squared__double(a, out=o)
        self.assertTrue((o == r).all())

if(__name__ == "__main__"):
    unittest.main()
