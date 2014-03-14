import ndarray_test_module
import unittest
import numpy

class TestNdarray(unittest.TestCase):

    def test_zeros(self):
        for dtp in (numpy.int16, numpy.int32, numpy.float32, numpy.complex128):
            v = numpy.zeros(60, dtype=dtp)
            dt = numpy.dtype(dtp)
            for shape in ((60,),(6,10),(4,3,5),(2,2,3,5)):
                a1 = ndarray_test_module.zeros(shape, dt)
                a2 = v.reshape(a1.shape)
                self.assertEqual(shape, a1.shape)
                self.assert_((a1 == a2).all())

if(__name__ == "__main__"):
    unittest.main()
