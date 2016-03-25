import numpy as np
from scipy.misc import imread, imresize

def glimpse_sensor(img, loc, size, scale, depth):
    """gets a glimpse of img at the location loc

    non-square images are treated as zero-padded to become square
    in translating loc to img coordinates

    loc (0,0) is the center of the image and (-1, -1) is the top left

    patches extending beyond the bounds of img are filled with zeros

    :param img: 2D image to glimpse
    :type img: numpy.ndarray

    :param loc: center location of the glimpse as (x, y)
    :type loc: (int, int)

    :param size: side length of each patch
    :type size: int

    :param scale: scaling factor for the patches. must be at least 1
    :type scale: float

    :param depth: number of patches in the glimpse
    :type depth: int

    :returns: glimpse of shape (num_patches, size[0], size[1])
    :rtype: numpy.ndarray
    """

    if scale < 1.0:
        raise ValueError("scale must be at least 1")

    glimpse = np.zeros((depth, size, size))

    for i in range(depth):
        scaled_size = size * (scale ** i)
        patch = get_patch(img, loc, scaled_size)

        scaled_patch = imresize(patch, (size, size))
        glimpse[i] = scaled_patch

    return glimpse


def get_patch(img, loc, size):
    """gets a patch of img at the location loc

    non-square images are treated as zero-padded to become square
    in translating loc to img coordinates

    loc (0,0) is the center of the image and (-1, -1) is the top left

    patches extending beyond the bounds of img are filled with zeros

    :param img: 2D image to get a patch of
    :type img: numpy.ndarray

    :param loc: center location of the patch as (x, y)
    :type loc: (int, int)

    :param size: side length of the patch
    :type size: int

    :returns: glimpse of shape size
    :rtype: numpy.ndarray
    """

    height, width = img.shape[:2]
    bounds = get_bounds(height, width, loc, size)

    return slice_with_pad(img, bounds)


def get_bounds(height, width, loc, size):
    """gets the bounds of a slice of size from an array of shape centered at loc

    loc (0,0) is the center of img and (-1, -1) is the top left

    :param height: height of the array
    :type height: int

    :param width: width of the array
    :type width: int

    :param loc: center location of the patch as (x, y)
    :type loc: (int, int)

    :param size: side length of the slice
    :type size: int

    :returns: x start, x end, y start, y end
    :rtype: (int, int, int, int)
    """

    side = max(height, width)
    mid = float(side) / 2

    height_adj = float(side - height) / 2
    width_adj = float(side - width) / 2

    x, y = loc
    x_img, y_img = (x * mid) + mid, (y * mid) + mid

    size_adj = (float(size) / 2)

    x_start = int(x_img - size_adj - height_adj)
    x_end = int(x_img + size_adj - height_adj)
    y_start = int(y_img - size_adj - width_adj)
    y_end = int(y_img + size_adj - width_adj)

    if (x_end - x_start) < size:
        x_end += 1

    if (y_end - y_start) < size:
        y_end += 1

    return (x_start, x_end, y_start, y_end)


def slice_with_pad(arr, bounds):
    """slices an array replacing indices out of bounds with fill value

    :param arr: array
    :type arr: numpy.ndarray

    :param bounds: x start, x end, y start, y end
    :type bounds: (int, int, int, int)

    :returns: array slice of shape
    :rtype: numpy.ndarray
    """

    x_start, x_end, y_start, y_end = bounds

    x_len = x_end - x_start
    y_len = y_end - y_start
    ret = np.zeros((x_len, y_len))

    valid_bounds = _clip_to_valid_bounds(bounds, arr.shape)
    arr_x_start, arr_x_end, arr_y_start, arr_y_end = valid_bounds

    slc_x_start = arr_x_start - x_start
    slc_x_end = arr_x_end - x_start

    slc_y_start = arr_y_start - y_start
    slc_y_end = arr_y_end - y_start

    ret[slc_x_start:slc_x_end, slc_y_start:slc_y_end] = arr[
        arr_x_start:arr_x_end, arr_y_start:arr_y_end
    ]

    return ret


def _clip_to_valid_bounds(bounds, shape):
    """clips the bounds to within the given shape

    :param bounds: x start, x end, y start, y end
    :type bounds: (int, int, int, int)

    :param shape: shape to clip to as (x_max, y_max) (assumes min is 0)
    :type shape: (int, int)

    :returns: clipped bounds
    :rtype: (int, int, int, int)
    """

    height, width = shape
    x_start, x_end, y_start, y_end = bounds

    valid_x_start = _clip_to_bounds(x_start, 0, height)
    valid_x_end = _clip_to_bounds(x_end, 0, height)

    valid_y_start = _clip_to_bounds(y_start, 0, width)
    valid_y_end = _clip_to_bounds(y_end, 0, width)

    return valid_x_start, valid_x_end, valid_y_start, valid_y_end


def _clip_to_bounds(value, start, end):
    """clips the value to within start and end inclusive"""

    if value < start:
        return start

    if value > end:
        return end

    return value
