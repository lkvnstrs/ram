import unittest
from itertools import product

import numpy as np

from glimpse.sensor import get_bounds, get_patch, slice_with_pad


"""
TODO

TestGetPatch
    - add more meaningful tests

TestGetBounds
    - add more meaningful tests

TestSliceWithPad
    - abstract out to only specify bounds as whether in or out and generate
      expected_pad (ex: bounds = (in, out, in, in))
    - refactor for list of height, width, bounds in/out combinations
"""


class TestGetPatch(unittest.TestCase):
    """tests the get_patch function"""

    def setUp(self):
        self.heights = [1000, 300, 500, 2053]
        self.widths = [1000, 700, 55, 2000]
        self.locs = [(0, 0), (-1, -1), (1, 1), (-1, 1), (-7, -7)]
        self.sizes = [50, 100, 20, 3]

    def test_x_y_equal_size(self):
        """tests that the side length of the patch is the same as the size"""

        param_iter = product(self.heights, self.widths, self.locs, self.sizes)
        for height, width, loc, size in param_iter:
            img = np.ones((height, width))
            patch = get_patch(img, loc, size)

            x_patch, y_patch = patch.shape
            self.assertEqual(x_patch, size)
            self.assertEqual(y_patch, size)


class TestGetBounds(unittest.TestCase):
    """tests the get_bounds function"""

    def setUp(self):
        self.heights = [1000, 300, 500, 2053]
        self.widths = [1000, 700, 55, 2000]
        self.locs = [(0, 0), (-1, -1), (1, 1), (-1, 1)]
        self.sizes = [50, 100, 20, 3]

    def test_side_length_is_size(self):
        """tests that the side length of each side is the same as size"""

        param_iter = product(self.heights, self.widths, self.locs, self.sizes)
        for height, width, loc, size in param_iter:
            x_start, x_end, y_start, y_end = get_bounds(
                height, width, loc, size)

            self.assertEqual(x_end - x_start, size)
            self.assertEqual(y_end - y_start, size)


class TestSliceWithPad(unittest.TestCase):
    """tests the slide_with_pad function"""

    def setUp(self):
        self.height = 1001
        self.width = 403

        self.arr = (np.arange(self.height * self.width)
                      .reshape(self.height, self.width))

        self.x_start_in = self.height / 2
        self.x_start_out = self.x_start_in - self.height

        self.x_end_in = self.height - 1
        self.x_end_out = self.x_end_in + self.height

        self.y_start_in = 0
        self.y_start_out = self.y_start_in - self.width

        self.y_end_in = self.width / 2
        self.y_end_out = self.y_end_in + self.width

        self.x_start_pad = abs(self.x_start_out)
        self.x_end_pad = self.x_end_out - self.height

        self.y_start_pad = abs(self.y_start_out)
        self.y_end_pad = self.y_end_out - self.width

    def _test_base(self, bounds, expected_pad):
        """base for testing slice_with_pad

        :param bounds: bounds input for slice_with_pad
        :type bounds: (int, int, int, int)

        :param expected_pad: expected result after padding slice
        :typ expected_pad: numpy.ndarray
        """

        expected_shape = self._get_expected_shape(bounds)
        actual_pad = slice_with_pad(self.arr, bounds)

        self.assertEqual(expected_pad.shape, expected_shape)
        self.assertEqual(actual_pad.shape, expected_shape)
        self.assertTrue((actual_pad == expected_pad).all())

    def test_all_in(self):
        """tests with square array and bounds all in range"""

        bounds = (
            self.x_start_in,
            self.x_end_in,
            self.y_start_in,
            self.y_end_in
        )

        expected_pad = self.arr[
            self.x_start_in:self.x_end_in,
            self.y_start_in:self.y_end_in
        ]

        self._test_base(bounds, expected_pad)

    def test_one_out(self):
        """tests with square array and all but one bound in range"""

        bounds = (
            self.x_start_out,
            self.x_end_in,
            self.y_start_in,
            self.y_end_in
        )

        expected_pad = self.arr[
            :self.x_end_in,
            self.y_start_in:self.y_end_in
        ]
        expected_pad = np.concatenate([
            np.zeros((self.x_start_pad, expected_pad.shape[1])),
            expected_pad
        ], axis=0)

        self._test_base(bounds, expected_pad)

    def test_two_out(self):
        """tests with square array and all but two bounds in range"""

        bounds = (
            self.x_start_in,
            self.x_end_out,
            self.y_start_out,
            self.y_end_in
        )

        expected_pad = self.arr[self.x_start_in:, :self.y_end_in]
        expected_pad = np.concatenate([
            expected_pad,
            np.zeros((self.x_end_pad, expected_pad.shape[1]))
        ], axis=0)

        expected_pad = np.concatenate([
            np.zeros((expected_pad.shape[0], self.y_start_pad)),
            expected_pad,
        ], axis=1)

        self._test_base(bounds, expected_pad)

    def test_three_out(self):
        """tests with square array and one bound in range"""

        bounds = (
            self.x_start_out,
            self.x_end_out,
            self.y_start_in,
            self.y_end_out
        )

        expected_pad = self.arr[:, self.y_start_in:]
        expected_pad = np.concatenate([
            np.zeros((self.x_start_pad, expected_pad.shape[1])),
            expected_pad,
            np.zeros((self.x_end_pad, expected_pad.shape[1]))
        ], axis=0)

        expected_pad = np.concatenate([
            expected_pad,
            np.zeros((expected_pad.shape[0], self.y_end_pad)),
        ], axis=1)

        self._test_base(bounds, expected_pad)

    def test_all_out(self):
        """tests with square array and all bounds out of range"""

        bounds = (
            self.x_start_out,
            self.x_end_out,
            self.y_start_out,
            self.y_end_out
        )

        expected_pad = self.arr[:, self.y_start_in:]
        expected_pad = np.concatenate([
            np.zeros((self.x_start_pad, expected_pad.shape[1])),
            expected_pad,
            np.zeros((self.x_end_pad, expected_pad.shape[1]))
        ], axis=0)

        expected_pad = np.concatenate([
            np.zeros((expected_pad.shape[0], self.y_start_pad)),
            expected_pad,
            np.zeros((expected_pad.shape[0], self.y_end_pad))
        ], axis=1)

        self._test_base(bounds, expected_pad)

    @staticmethod
    def _get_expected_shape(bounds):
        """gets the expected shape of an array sliced with bounds

        :param bounds: (x start, x end, y start, y end)
        :type bounds: (int, int, int, int)

        :returns: expected shape of sliced array
        :rtype: (int, int)
        """

        x_start, x_end, y_start, y_end = bounds
        return (x_end - x_start, y_end - y_start)
