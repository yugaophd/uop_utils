import unittest

import numpy as np

from uop_utils import compute_current_relative_wind


class ComputeCurrentRelativeWindTests(unittest.TestCase):
    def test_missing_current_preserves_true_wind(self):
        speed = np.array([4.0, 6.0])
        direction = np.array([0.0, 90.0])
        current_east = np.array([np.nan, np.nan])
        current_north = np.array([np.nan, np.nan])

        rel_speed, rel_direction, valid = compute_current_relative_wind(
            speed, direction, current_east, current_north
        )

        np.testing.assert_allclose(rel_speed, speed)
        np.testing.assert_allclose(rel_direction, direction)
        np.testing.assert_array_equal(valid, np.array([False, False]))

    def test_valid_current_uses_vector_subtraction(self):
        speed = np.array([5.0])
        direction = np.array([90.0])
        current_east = np.array([1.0])
        current_north = np.array([0.0])

        rel_speed, rel_direction, valid = compute_current_relative_wind(
            speed, direction, current_east, current_north
        )

        np.testing.assert_allclose(rel_speed, np.array([6.0]))
        np.testing.assert_allclose(rel_direction, np.array([90.0]))
        np.testing.assert_array_equal(valid, np.array([True]))

    def test_mixed_current_does_not_introduce_artificial_zero_wind(self):
        speed = np.array([3.5, 4.2, 5.1])
        direction = np.array([45.0, 180.0, 270.0])
        current_east = np.array([0.2, np.nan, -0.5])
        current_north = np.array([0.1, np.nan, 0.3])

        rel_speed, rel_direction, valid = compute_current_relative_wind(
            speed, direction, current_east, current_north
        )

        self.assertEqual(rel_speed[1], speed[1])
        self.assertEqual(rel_direction[1], direction[1])
        self.assertGreater(rel_speed[1], 0.0)
        np.testing.assert_array_equal(valid, np.array([True, False, True]))


if __name__ == '__main__':
    unittest.main()
