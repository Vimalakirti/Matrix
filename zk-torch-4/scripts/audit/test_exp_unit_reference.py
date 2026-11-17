"""Scalar old/new exp(0) unit regression; NOT execution of the Rust graph."""
import unittest


def scale_down(x, shift):
    return (x + ((1 << shift) // 2)) // (1 << shift)


def exp_zero(unit, output_sf):
    # At r=0, both nonconstant cubic terms vanish. TwoPow(0)=2^15;
    # the product has sf30 and the graph scales it to output_sf.
    product = (1 << 15) * unit
    return scale_down(product, 30 - output_sf)


class ExpUnit(unittest.TestCase):
    def test_old_constant_is_not_one(self):
        self.assertEqual(exp_zero(1, 10), 0)
        self.assertEqual(exp_zero(1, 15), 1)
        self.assertNotEqual(exp_zero(1, 15), 1 << 15)

    def test_repaired_zero_across_output_scales(self):
        for sf in range(1, 30):
            self.assertEqual(exp_zero(1 << 15, sf), 1 << sf)


if __name__ == '__main__':
    unittest.main()
