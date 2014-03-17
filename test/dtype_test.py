#
# $Id$
#
# Copyright (C)
# 2014 - $Date$
#     Martin Wolf <boostnumpy@martin-wolf.org>
# 2010-2012
#     Jim Bosch, Ankit Daftery
#
# This file implements tests for boost::numpy::dtype class.
#
# This file is distributed under the Boost Software License,
# Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt).
#
import dtype_test_module
import unittest
import numpy as np

class TestDtype(unittest.TestCase):

    def assertEquivalent(self, a, b):
        return self.assertTrue(dtype_test_module.equivalent(a, b),
                               "%r is not equivalent to %r")

    def test_integers(self):
        for bits in (8, 16, 32, 64):
            s = getattr(np, "int%d" % bits)
            u = getattr(np, "uint%d" % bits)
            fs = getattr(dtype_test_module, "accept_int%d" % bits)
            fu = getattr(dtype_test_module, "accept_uint%d" % bits)
            self.assertEquivalent(fs(s(1)), np.dtype(s))
            self.assertEquivalent(fu(u(1)), np.dtype(u))
            # These should just use the regular Boost.Python converters.
            self.assertEquivalent(fs(True), np.dtype(s))
            self.assertEquivalent(fu(True), np.dtype(u))
            self.assertEquivalent(fs(int(1)), np.dtype(s))
            self.assertEquivalent(fu(int(1)), np.dtype(u))
            self.assertEquivalent(fs(long(1)), np.dtype(s))
            self.assertEquivalent(fu(long(1)), np.dtype(u))

        for name in ("bool_", "byte", "ubyte", "short", "ushort", "intc", "uintc"):
            t = getattr(np, name)
            ft = getattr(dtype_test_module, "accept_%s" % name)
            self.assertEquivalent(ft(t(1)), np.dtype(t))
            # These should just use the regular Boost.Python converters.
            self.assertEquivalent(ft(True), np.dtype(t))
            if(name != "bool_"):
                self.assertEquivalent(ft(int(1)), np.dtype(t))
                self.assertEquivalent(ft(long(1)), np.dtype(t))

    def test_floats(self):
        f = np.float32
        c = np.complex64
        self.assertEquivalent(dtype_test_module.accept_float32(f(np.pi)), np.dtype(f))
        self.assertEquivalent(dtype_test_module.accept_complex64(c(1+2j)), np.dtype(c))
        f = np.float64
        c = np.complex128
        self.assertEquivalent(dtype_test_module.accept_float64(f(np.pi)), np.dtype(f))
        self.assertEquivalent(dtype_test_module.accept_complex128(c(1+2j)), np.dtype(c))
        if(hasattr(np, "longdouble")):
            f = np.longdouble
            c = np.clongdouble
            self.assertEquivalent(dtype_test_module.accept_longdouble(f(np.pi)), np.dtype(f))
            self.assertEquivalent(dtype_test_module.accept_clongdouble(c(1+2j)), np.dtype(c))

if(__name__ == "__main__"):
    unittest.main()
