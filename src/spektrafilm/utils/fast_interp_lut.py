import numpy as np
import warnings
from numba import njit, prange
from scipy.ndimage import map_coordinates

@njit(cache=True)
def mitchell_weight(t, B=1/3, C=1/3):
    """
    Computes the Mitchell–Netravali kernel weight.
    """
    x = abs(t)
    if x < 1:
        return (1/6)*((12 - 9*B - 6*C)*x**3 + (-18 + 12*B + 6*C)*x**2 + (6 - 2*B))
    elif x < 2:
        return (1/6)*((-B - 6*C)*x**3 + (6*B + 30*C)*x**2 + (-12*B - 48*C)*x + (8*B + 24*C))
    else:
        return 0.0

@njit(cache=True)
def safe_index(idx, L):
    """
    Reflect an index into the valid range [0, L-1] using symmetric reflection.
    """
    if idx < 0:
        return -idx
    elif idx >= L:
        return 2*(L - 1) - idx
    else:
        return idx


@njit(cache=True)
def clamp_coordinate(coord, L):
    """
    Clamp a floating-point coordinate to the valid LUT domain [0, L-1].
    """
    if coord <= 0.0:
        return 0.0
    upper = float(L - 1)
    if coord >= upper:
        return upper
    return coord


@njit(cache=True)
def cubic_coordinate_base_fraction(coord, L):
    """
    Map a clamped coordinate onto a cubic interpolation cell in [0, L-2].
    """
    coord = clamp_coordinate(coord, L)
    if coord >= float(L - 1):
        return L - 2, 1.0

    base = int(np.floor(coord))
    return base, coord - base


@njit(cache=True)
def _constant_lut_value_3d(lut):
    out = np.empty(3, dtype=lut.dtype)
    out[0] = lut[0, 0, 0, 0]
    out[1] = lut[0, 0, 0, 1]
    out[2] = lut[0, 0, 0, 2]
    return out


@njit(cache=True)
def linear_interp_lut_at_2d(lut, x, y):
    """
    Performs bilinear interpolation at a single point in a 2D LUT.
    Used as a boundary-safe fallback for the outermost LUT cells.
    """
    L = lut.shape[0]
    channels = lut.shape[2]
    x = clamp_coordinate(x, L)
    y = clamp_coordinate(y, L)

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, L - 1)
    y1 = min(y0 + 1, L - 1)

    tx = x - x0
    ty = y - y0

    out = np.zeros(channels, dtype=lut.dtype)
    for i in range(2):
        xi = x0 if i == 0 else x1
        wx = (1.0 - tx) if i == 0 else tx
        for j in range(2):
            yj = y0 if j == 0 else y1
            wy = (1.0 - ty) if j == 0 else ty
            weight = wx * wy
            for c in range(channels):
                out[c] += weight * lut[xi, yj, c]
    return out

# ---------------------------
# 3D LUT Cubic Interpolation
# ---------------------------
@njit(cache=True)
def _cubic_interp_lut_at_3d(lut, r, g, b):
    """
    Perform cubic interpolation on a 3D LUT with reflected boundary handling.
    """
    size = lut.shape[0]
    r_base, r_frac = cubic_coordinate_base_fraction(r, size)
    g_base, g_frac = cubic_coordinate_base_fraction(g, size)
    b_base, b_frac = cubic_coordinate_base_fraction(b, size)

    wr = np.empty(4, dtype=lut.dtype)
    wg = np.empty(4, dtype=lut.dtype)
    wb = np.empty(4, dtype=lut.dtype)
    wr[0] = mitchell_weight(r_frac + 1)
    wr[1] = mitchell_weight(r_frac)
    wr[2] = mitchell_weight(r_frac - 1)
    wr[3] = mitchell_weight(r_frac - 2)
    wg[0] = mitchell_weight(g_frac + 1)
    wg[1] = mitchell_weight(g_frac)
    wg[2] = mitchell_weight(g_frac - 1)
    wg[3] = mitchell_weight(g_frac - 2)
    wb[0] = mitchell_weight(b_frac + 1)
    wb[1] = mitchell_weight(b_frac)
    wb[2] = mitchell_weight(b_frac - 1)
    wb[3] = mitchell_weight(b_frac - 2)

    out = np.zeros(3, dtype=lut.dtype)
    weight_sum = 0.0
    for i in range(4):
        ri = safe_index(r_base - 1 + i, size)
        for j in range(4):
            gj = safe_index(g_base - 1 + j, size)
            for k in range(4):
                bk = safe_index(b_base - 1 + k, size)
                weight = wr[i] * wg[j] * wb[k]
                weight_sum += weight
                out[0] += weight * lut[ri, gj, bk, 0]
                out[1] += weight * lut[ri, gj, bk, 1]
                out[2] += weight * lut[ri, gj, bk, 2]
    if weight_sum != 0.0:
        out[0] /= weight_sum
        out[1] /= weight_sum
        out[2] /= weight_sum
    return out


def cubic_interp_lut_at_3d(lut, r, g, b):
    """
    Performs cubic interpolation at a single point (r, g, b) in a 3D LUT (shape: LxLxLx3)
    using reflected boundary handling.
    """
    visible_size = lut.shape[0]
    if visible_size == 0:
        raise ValueError('3D LUT must not be empty')
    if visible_size == 1:
        return _constant_lut_value_3d(lut)
    return _cubic_interp_lut_at_3d(lut, r, g, b)

@njit(parallel=True, cache=True)
def _apply_lut_constant_3d(lut, image):
    height, width, _ = image.shape
    output = np.empty((height, width, 3), dtype=lut.dtype)
    value = _constant_lut_value_3d(lut)
    for i in prange(height):
        for j in range(width):
            output[i, j, 0] = value[0]
            output[i, j, 1] = value[1]
            output[i, j, 2] = value[2]
    return output


@njit(parallel=True, cache=True)
def _apply_lut_cubic_3d(lut, image):
    """
    Apply cubic interpolation to a 3D LUT with reflected boundary handling.
    """
    height, width, _ = image.shape
    output = np.empty((height, width, 3), dtype=lut.dtype)
    scale = lut.shape[0] - 1
    for i in prange(height):
        for j in range(width):
            r_in = image[i, j, 0] * scale
            g_in = image[i, j, 1] * scale
            b_in = image[i, j, 2] * scale
            out_val = _cubic_interp_lut_at_3d(lut, r_in, g_in, b_in)
            output[i, j, 0] = out_val[0]
            output[i, j, 1] = out_val[1]
            output[i, j, 2] = out_val[2]
    return output


def apply_lut_cubic_3d(lut, image):
    """
    Applies a 3D LUT (shape: LxLxLx3) to an image (shape: HxWx3) using cubic interpolation.
    Data is assumed to be normalized in the range [0, 1] and will be scaled to [0, L-1] for LUT indexing.
    Boundary handling uses symmetric reflection without materializing a padded LUT.
    """
    lut = np.ascontiguousarray(lut, dtype=np.float64)
    return _apply_lut_cubic_3d(lut, image)


#########################
# PCHIP 3D LUT interpolation
#########################

@njit(cache=True)
def _fill_monotone_slopes_1d(values, slopes):
    """
    Fill monotone cubic Hermite slopes for a uniformly sampled 1D signal.
    Uses a PCHIP-style limiter so monotone sample lines stay monotone.
    """
    size = values.shape[0]
    if size == 1:
        slopes[0] = 0.0
        return

    deltas = np.empty(size - 1, dtype=values.dtype)
    for i in range(size - 1):
        deltas[i] = values[i + 1] - values[i]

    if size == 2:
        slopes[0] = deltas[0]
        slopes[1] = deltas[0]
        return

    left = 0.5 * (3.0 * deltas[0] - deltas[1])
    if left * deltas[0] <= 0.0:
        left = 0.0
    elif deltas[0] * deltas[1] < 0.0 and abs(left) > abs(3.0 * deltas[0]):
        left = 3.0 * deltas[0]
    slopes[0] = left

    for i in range(1, size - 1):
        delta_prev = deltas[i - 1]
        delta_next = deltas[i]
        if delta_prev == 0.0 or delta_next == 0.0 or delta_prev * delta_next <= 0.0:
            slopes[i] = 0.0
        else:
            slopes[i] = 2.0 * delta_prev * delta_next / (delta_prev + delta_next)

    right = 0.5 * (3.0 * deltas[size - 2] - deltas[size - 3])
    if right * deltas[size - 2] <= 0.0:
        right = 0.0
    elif deltas[size - 2] * deltas[size - 3] < 0.0 and abs(right) > abs(3.0 * deltas[size - 2]):
        right = 3.0 * deltas[size - 2]
    slopes[size - 1] = right


@njit(cache=True)
def _prepare_lut_pchip_3d_impl(lut):
    size = lut.shape[0]
    slope_x = np.empty_like(lut)
    slope_y = np.empty_like(lut)
    slope_z = np.empty_like(lut)
    cell_min = np.empty((size - 1, size - 1, size - 1, 3), dtype=lut.dtype)
    cell_max = np.empty((size - 1, size - 1, size - 1, 3), dtype=lut.dtype)

    line = np.empty(size, dtype=lut.dtype)
    slopes = np.empty(size, dtype=lut.dtype)

    for j in range(size):
        for k in range(size):
            for c in range(3):
                for i in range(size):
                    line[i] = lut[i, j, k, c]
                _fill_monotone_slopes_1d(line, slopes)
                for i in range(size):
                    slope_x[i, j, k, c] = slopes[i]

    for i in range(size):
        for k in range(size):
            for c in range(3):
                for j in range(size):
                    line[j] = lut[i, j, k, c]
                _fill_monotone_slopes_1d(line, slopes)
                for j in range(size):
                    slope_y[i, j, k, c] = slopes[j]

    for i in range(size):
        for j in range(size):
            for c in range(3):
                for k in range(size):
                    line[k] = lut[i, j, k, c]
                _fill_monotone_slopes_1d(line, slopes)
                for k in range(size):
                    slope_z[i, j, k, c] = slopes[k]

    for i in range(size - 1):
        for j in range(size - 1):
            for k in range(size - 1):
                for c in range(3):
                    min_value = lut[i, j, k, c]
                    max_value = min_value
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                sample = lut[i + di, j + dj, k + dk, c]
                                if sample < min_value:
                                    min_value = sample
                                elif sample > max_value:
                                    max_value = sample
                    cell_min[i, j, k, c] = min_value
                    cell_max[i, j, k, c] = max_value

    return slope_x, slope_y, slope_z, cell_min, cell_max


def _warn_if_lut_not_monotonic_3d(
    lut,
    atol=1e-12,
    max_violation_tol=3e-3,
):
    """
    Soft-check monotonicity using first differences along each LUT dimension.
    Emits a warning when gradients along one axis contain both significant
    positive and negative steps.
    """
    axis_labels = ('r', 'g', 'b')
    channel_labels = ('out_r', 'out_g', 'out_b')

    for axis_index, axis_label in enumerate(axis_labels):
        gradients = np.diff(lut, axis=axis_index)
        for channel_index, channel_label in enumerate(channel_labels):
            channel_gradients = gradients[..., channel_index]
            max_gradient = float(np.max(channel_gradients))
            min_gradient = float(np.min(channel_gradients))

            if max_gradient <= atol or min_gradient >= -atol:
                continue

            violation = min(max_gradient, -min_gradient)
            if violation <= max_violation_tol:
                continue

            warnings.warn(
                f'3D LUT is not monotone for {channel_label} along {axis_label} axis '
                f'(min gradient={min_gradient:.3e}, '
                f'max gradient={max_gradient:.3e}, '
                f'max_violation_tol={max_violation_tol:.3e}, atol={atol}); '
                'continuing with PCHIP interpolation.',
                RuntimeWarning,
                stacklevel=3,
            )
            return False
    return True


def prepare_lut_pchip_3d(lut):
    """
    Precompute per-axis PCHIP-style monotone slopes for a 3D LUT.
    This is intended to be done once per LUT and then reused for many images.
    """
    if lut.ndim != 4 or lut.shape[3] != 3:
        raise ValueError('3D LUT must have shape LxLxLx3')

    size = lut.shape[0]
    if lut.shape[1] != size or lut.shape[2] != size:
        raise ValueError('3D LUT must have equal dimensions on all axes')
    if size == 0:
        raise ValueError('3D LUT must not be empty')

    lut = np.ascontiguousarray(lut, dtype=np.float64)
    _warn_if_lut_not_monotonic_3d(lut)
    slope_x, slope_y, slope_z, cell_min, cell_max = _prepare_lut_pchip_3d_impl(lut)
    return (
        lut,
        np.ascontiguousarray(slope_x),
        np.ascontiguousarray(slope_y),
        np.ascontiguousarray(slope_z),
        np.ascontiguousarray(cell_min),
        np.ascontiguousarray(cell_max),
    )

@njit(cache=True)
def _hermite_value(y0, y1, m0, m1, t):
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1


@njit(cache=True)
def _linear_mix(v0, v1, t):
    return v0 + t * (v1 - v0)


@njit(cache=True)
def _bilinear_mix(v00, v10, v01, v11, tx, ty):
    vx0 = _linear_mix(v00, v10, tx)
    vx1 = _linear_mix(v01, v11, tx)
    return _linear_mix(vx0, vx1, ty)


@njit(cache=True)
def _clamp_to_range(value, min_value, max_value):
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


@njit(cache=True)
def _pchip_interp_lut_at_3d_prepared(lut, slope_x, slope_y, slope_z, cell_min, cell_max, r, g, b):
    size = lut.shape[0]
    i, tr = cubic_coordinate_base_fraction(r, size)
    j, tg = cubic_coordinate_base_fraction(g, size)
    k, tb = cubic_coordinate_base_fraction(b, size)

    out = np.empty(3, dtype=lut.dtype)
    for c in range(3):
        v000 = _hermite_value(lut[i, j, k, c], lut[i + 1, j, k, c], slope_x[i, j, k, c], slope_x[i + 1, j, k, c], tr)
        v010 = _hermite_value(lut[i, j + 1, k, c], lut[i + 1, j + 1, k, c], slope_x[i, j + 1, k, c], slope_x[i + 1, j + 1, k, c], tr)
        v001 = _hermite_value(lut[i, j, k + 1, c], lut[i + 1, j, k + 1, c], slope_x[i, j, k + 1, c], slope_x[i + 1, j, k + 1, c], tr)
        v011 = _hermite_value(lut[i, j + 1, k + 1, c], lut[i + 1, j + 1, k + 1, c], slope_x[i, j + 1, k + 1, c], slope_x[i + 1, j + 1, k + 1, c], tr)

        sy00 = _linear_mix(slope_y[i, j, k, c], slope_y[i + 1, j, k, c], tr)
        sy10 = _linear_mix(slope_y[i, j + 1, k, c], slope_y[i + 1, j + 1, k, c], tr)
        sy01 = _linear_mix(slope_y[i, j, k + 1, c], slope_y[i + 1, j, k + 1, c], tr)
        sy11 = _linear_mix(slope_y[i, j + 1, k + 1, c], slope_y[i + 1, j + 1, k + 1, c], tr)

        vz0 = _hermite_value(v000, v010, sy00, sy10, tg)
        vz1 = _hermite_value(v001, v011, sy01, sy11, tg)

        sz0 = _bilinear_mix(slope_z[i, j, k, c], slope_z[i + 1, j, k, c], slope_z[i, j + 1, k, c], slope_z[i + 1, j + 1, k, c], tr, tg)
        sz1 = _bilinear_mix(slope_z[i, j, k + 1, c], slope_z[i + 1, j, k + 1, c], slope_z[i, j + 1, k + 1, c], slope_z[i + 1, j + 1, k + 1, c], tr, tg)

        interpolated = _hermite_value(vz0, vz1, sz0, sz1, tb)
        out[c] = _clamp_to_range(interpolated, cell_min[i, j, k, c], cell_max[i, j, k, c])
    return out


@njit(parallel=True, cache=True)
def _apply_lut_pchip_3d_prepared(lut, slope_x, slope_y, slope_z, cell_min, cell_max, image):
    height, width, _ = image.shape
    output = np.empty((height, width, 3), dtype=lut.dtype)
    scale = lut.shape[0] - 1
    for i in prange(height):
        for j in range(width):
            r_in = image[i, j, 0] * scale
            g_in = image[i, j, 1] * scale
            b_in = image[i, j, 2] * scale
            out_val = _pchip_interp_lut_at_3d_prepared(lut, slope_x, slope_y, slope_z, cell_min, cell_max, r_in, g_in, b_in)
            output[i, j, 0] = out_val[0]
            output[i, j, 1] = out_val[1]
            output[i, j, 2] = out_val[2]
    return output

def apply_lut_pchip_3d(lut, image):
    """
    Apply the PCHIP 3D LUT path using precomputed per-axis slopes.
    Pass the tuple returned by prepare_lut_pchip_3d().
    """
    lut, slope_x, slope_y, slope_z, cell_min, cell_max = prepare_lut_pchip_3d(lut)
    if lut.shape[0] == 1:
        return _apply_lut_constant_3d(lut, image)
    return _apply_lut_pchip_3d_prepared(lut, slope_x, slope_y, slope_z, cell_min, cell_max, image)


#########################
# Public 3D LUT API
#########################

def apply_lut_3d(lut, image, method='pchip'):
    """
    Apply a 3D LUT using the selected interpolation method.

    The default is 'pchip'. The requested method directly selects the interpolation path.
    """
    if method == 'mitchell':
        return apply_lut_cubic_3d(lut, image)
    if method == 'pchip':
        return apply_lut_pchip_3d(lut, image)
    raise ValueError("method must be 'mitchell' or 'pchip'")

# ---------------------------
# 2D LUT Cubic Interpolation (using x, y channels)
# ---------------------------
@njit(cache=True)
def _cubic_interp_lut_at_2d(lut, x, y):
    """
    Perform cubic interpolation on a 2D LUT with reflected boundary handling.
    """
    channels = lut.shape[2]
    visible_size = lut.shape[0]
    x_base, x_frac = cubic_coordinate_base_fraction(x, visible_size)
    y_base, y_frac = cubic_coordinate_base_fraction(y, visible_size)

    wx = np.empty(4, dtype=np.float64)
    wy = np.empty(4, dtype=np.float64)
    wx[0] = mitchell_weight(x_frac + 1)
    wx[1] = mitchell_weight(x_frac)
    wx[2] = mitchell_weight(x_frac - 1)
    wx[3] = mitchell_weight(x_frac - 2)
    wy[0] = mitchell_weight(y_frac + 1)
    wy[1] = mitchell_weight(y_frac)
    wy[2] = mitchell_weight(y_frac - 1)
    wy[3] = mitchell_weight(y_frac - 2)

    out = np.zeros(channels, dtype=np.float64)
    weight_sum = 0.0
    for i in range(4):
        xi = safe_index(x_base - 1 + i, visible_size)
        for j in range(4):
            yj = safe_index(y_base - 1 + j, visible_size)
            weight = wx[i] * wy[j]
            weight_sum += weight
            for c in range(channels):
                out[c] += weight * lut[xi, yj, c]
    if weight_sum != 0.0:
        for c in range(channels):
            out[c] /= weight_sum
    return out


def cubic_interp_lut_at_2d(lut, x, y):
    """
    Performs cubic interpolation at a single point (x, y) in a 2D LUT (shape: LxLxC)
    using reflected boundary handling.
    """
    visible_size = lut.shape[0]
    if visible_size < 2:
        return linear_interp_lut_at_2d(lut, x, y)
    return _cubic_interp_lut_at_2d(lut, x, y)

def apply_lut_cubic_2d(lut, image):
    """
    Applies a 2D LUT (shape: LxLxC) to an image (shape: HxWxC) using cubic interpolation.
    Here the image channels represent the (x, y) coordinates.
    """
    visible_size = lut.shape[0]
    if visible_size < 2:
        return _apply_lut_linear_2d(lut, image)
    return _apply_lut_cubic_2d(lut, image)


@njit(parallel=True, cache=True)
def _apply_lut_linear_2d(lut, image):
    height, width, _ = image.shape
    channels = lut.shape[2]
    output = np.empty((height, width, channels), dtype=np.float64)
    size = lut.shape[0]
    for i in prange(height):
        for j in range(width):
            x_in = image[i, j, 0] * (size - 1)
            y_in = image[i, j, 1] * (size - 1)
            out_val = linear_interp_lut_at_2d(lut, x_in, y_in)
            for c in range(channels):
                output[i, j, c] = out_val[c]
    return output


@njit(parallel=True, cache=True)
def _apply_lut_cubic_2d(lut, image):
    """
    Apply cubic interpolation to a 2D LUT with reflected boundary handling.
    """
    height, width, _ = image.shape
    channels = lut.shape[2]
    output = np.empty((height, width, channels), dtype=np.float64)
    scale = lut.shape[0] - 1
    for i in prange(height):
        for j in range(width):
            x_in = image[i, j, 0] * scale
            y_in = image[i, j, 1] * scale
            out_val = _cubic_interp_lut_at_2d(lut, x_in, y_in)
            for c in range(channels):
                output[i, j, c] = out_val[c]
    return output

# ---------------------------
# SciPy Reference Implementations
# ---------------------------
def apply_lut_cubic_scipy(lut, image):
    """
    Applies cubic interpolation using SciPy's map_coordinates.
    Dispatches based on the LUT dimensionality.
    For a 3D LUT (4D array) we assume channels are (r,g,b),
    and for a 2D LUT (3D array) we assume channels are (x,y,...).
    """
    if lut.ndim == 4:  # 3D LUT case
        height, width, _ = image.shape
        L = lut.shape[0]
        coords = np.empty((3, height, width), dtype=np.float64)
        coords[0] = image[:, :, 0] * (L - 1)
        coords[1] = image[:, :, 1] * (L - 1)
        coords[2] = image[:, :, 2] * (L - 1)
        output = np.empty((height, width, 3), dtype=np.float64)
        for c in range(3):
            output[:, :, c] = map_coordinates(lut[..., c], coords, order=3, mode='reflect')
        return output
    elif lut.ndim == 3:  # 2D LUT case
        height, width, _ = image.shape
        L = lut.shape[0]
        channels = lut.shape[2]
        coords = np.empty((2, height, width), dtype=np.float64)
        # Using x and y channels.
        coords[0] = image[:, :, 0] * (L - 1)
        coords[1] = image[:, :, 1] * (L - 1)
        output = np.empty((height, width, channels), dtype=np.float64)
        for c in range(channels):
            output[:, :, c] = map_coordinates(lut[..., c], coords, order=3, mode='reflect')
        return output

# ---------------------------
# Quick Local Testing Block
# ---------------------------
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    def ground_truth_3d(image):
        out = np.empty_like(image)
        out[..., 0] = image[..., 0] ** 2
        out[..., 1] = image[..., 1] ** 2
        out[..., 2] = image[..., 2] ** 2
        return out

    def ground_truth_2d(image):
        out = np.empty((image.shape[0], image.shape[1], 2), dtype=np.float64)
        out[..., 0] = image[..., 0] ** 2
        out[..., 1] = image[..., 1] ** 2
        return out

    # --- 3D LUT Example ---
    lut_size_3d = 32
    grid_3d = np.linspace(0, 1, lut_size_3d, dtype=np.float64)
    grid_r_3d, grid_g_3d, grid_b_3d = np.meshgrid(grid_3d, grid_3d, grid_3d, indexing='ij')
    # Create a 3D LUT that applies a simple non-linear transformation (r^2, g^2, b^2)
    lut_3d = np.stack((grid_r_3d**2, grid_g_3d**2, grid_b_3d**2), axis=-1)  # shape: (L, L, L, 3)

    # Create a synthetic test image (gradient image, 3 channels)
    image_height, image_width = 1024, 1024
    x_axis_3d = np.linspace(0, 1, image_width, dtype=np.float64)
    y_axis_3d = np.linspace(0, 1, image_height, dtype=np.float64)
    grid_x_3d, grid_y_3d = np.meshgrid(x_axis_3d, y_axis_3d)
    image_3d = np.stack((grid_x_3d, grid_y_3d, 0.5 * np.ones_like(grid_x_3d)), axis=-1)

    # Warm up the JIT compiler
    _ = apply_lut_3d(lut_3d, image_3d)
    _ = apply_lut_3d(lut_3d, image_3d, method='mitchell')
    _ = apply_lut_3d(lut_3d, image_3d, method='pchip')

    iterations = 10
    start_time = time.time()
    for _ in range(iterations):
        output_default_3d = apply_lut_3d(lut_3d, image_3d)
    default_time_3d = (time.time() - start_time) / iterations
    print("3D LUT - Average time per iteration (default raw LUT path): {:.6f} seconds".format(default_time_3d))

    start_time = time.time()
    for _ in range(iterations):
        output_mitchell_3d = apply_lut_3d(lut_3d, image_3d, method='mitchell')
    mitchell_time_3d = (time.time() - start_time) / iterations
    print("3D LUT - Average time per iteration (Mitchell cubic interpolation): {:.6f} seconds".format(mitchell_time_3d))

    start_time = time.time()
    for _ in range(iterations):
        output_pchip_3d = apply_lut_3d(lut_3d, image_3d, method='pchip')
    pchip_time_3d = (time.time() - start_time) / iterations
    print("3D LUT - Average time per iteration (PCHIP cubic Hermite): {:.6f} seconds".format(pchip_time_3d))

    start_time = time.time()
    for _ in range(iterations):
        output_scipy_3d = apply_lut_cubic_scipy(lut_3d, image_3d)
    scipy_time_3d = (time.time() - start_time) / iterations
    print("3D LUT - Average time per iteration (SciPy cubic interpolation): {:.6f} seconds".format(scipy_time_3d))

    output_ground_truth_3d = ground_truth_3d(image_3d)

    diff_default_scipy_3d = output_default_3d - output_scipy_3d
    rmse_default_scipy_3d = np.sqrt(np.mean(diff_default_scipy_3d**2))
    max_error_default_scipy_3d = np.max(np.abs(diff_default_scipy_3d))
    print("3D LUT - Default-path RMSE error against SciPy: {:.6e}".format(rmse_default_scipy_3d))
    print("3D LUT - Default-path max absolute error against SciPy: {:.6e}".format(max_error_default_scipy_3d))

    diff_mitchell_scipy_3d = output_mitchell_3d - output_scipy_3d
    rmse_mitchell_scipy_3d = np.sqrt(np.mean(diff_mitchell_scipy_3d**2))
    max_error_mitchell_scipy_3d = np.max(np.abs(diff_mitchell_scipy_3d))
    print("3D LUT - Mitchell RMSE error against SciPy: {:.6e}".format(rmse_mitchell_scipy_3d))
    print("3D LUT - Mitchell max absolute error against SciPy: {:.6e}".format(max_error_mitchell_scipy_3d))

    diff_default_ground_truth_3d = output_default_3d - output_ground_truth_3d
    rmse_default_ground_truth_3d = np.sqrt(np.mean(diff_default_ground_truth_3d**2))
    max_error_default_ground_truth_3d = np.max(np.abs(diff_default_ground_truth_3d))
    print("3D LUT - Default-path RMSE error against ground truth: {:.6e}".format(rmse_default_ground_truth_3d))
    print("3D LUT - Default-path max absolute error against ground truth: {:.6e}".format(max_error_default_ground_truth_3d))

    diff_pchip_ground_truth_3d = output_pchip_3d - output_ground_truth_3d
    rmse_pchip_ground_truth_3d = np.sqrt(np.mean(diff_pchip_ground_truth_3d**2))
    max_error_pchip_ground_truth_3d = np.max(np.abs(diff_pchip_ground_truth_3d))
    print("3D LUT - PCHIP RMSE error against ground truth: {:.6e}".format(rmse_pchip_ground_truth_3d))
    print("3D LUT - PCHIP max absolute error against ground truth: {:.6e}".format(max_error_pchip_ground_truth_3d))

    diff_scipy_ground_truth_3d = output_scipy_3d - output_ground_truth_3d
    rmse_scipy_ground_truth_3d = np.sqrt(np.mean(diff_scipy_ground_truth_3d**2))
    max_error_scipy_ground_truth_3d = np.max(np.abs(diff_scipy_ground_truth_3d))
    print("3D LUT - SciPy RMSE error against ground truth: {:.6e}".format(rmse_scipy_ground_truth_3d))
    print("3D LUT - SciPy max absolute error against ground truth: {:.6e}".format(max_error_scipy_ground_truth_3d))

    diff_norm_ground_truth_3d = np.sqrt(np.sum(diff_default_ground_truth_3d**2, axis=2))
    diff_norm_scipy_ground_truth_3d = np.sqrt(np.sum(diff_scipy_ground_truth_3d**2, axis=2))
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    input_im = axs[0, 0].imshow(image_3d, interpolation='nearest')
    axs[0, 0].set_title("Input Gradient Image (3D LUT)")
    axs[0, 0].axis("off")
    output_ground_truth_im = axs[0, 1].imshow(output_ground_truth_3d, interpolation='nearest')
    axs[0, 1].set_title("Output (Ground Truth, 3D LUT)")
    axs[0, 1].axis("off")
    im_numba = axs[1, 0].imshow(diff_norm_ground_truth_3d, cmap="hot", interpolation="nearest")
    axs[1, 0].set_title("Error Map (Default/PCHIP vs Ground Truth, 3D LUT)")
    axs[1, 0].axis("off")
    fig.colorbar(im_numba, ax=axs[1, 0], fraction=0.046, pad=0.04)
    im_scipy = axs[1, 1].imshow(diff_norm_scipy_ground_truth_3d, cmap="hot", interpolation="nearest")
    axs[1, 1].set_title("Error Map (SciPy vs Ground Truth, 3D LUT)")
    axs[1, 1].axis("off")
    fig.colorbar(im_scipy, ax=axs[1, 1], fraction=0.046, pad=0.04)
    fig.suptitle("3D LUT Cubic Interpolation Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 2D LUT Example (using x, y channels) ---
    # Create a 2D LUT that maps two input channels (x, y) to two output channels,
    # e.g. by applying a non-linear transform (x^2, y^2)
    lut_size_2d = 128
    grid_2d = np.linspace(0, 1, lut_size_2d, dtype=np.float64)
    lut_2d = np.empty((lut_size_2d, lut_size_2d, 2), dtype=np.float64)
    grid_x_2d, grid_y_2d = np.meshgrid(grid_2d, grid_2d, indexing='ij')
    lut_2d[..., 0] = grid_x_2d**2
    lut_2d[..., 1] = grid_y_2d**2

    # Create a synthetic test image (gradient image, 2 channels for x and y)
    image_2d = np.stack((grid_x_3d, grid_y_3d), axis=-1)

    # Warm up the JIT compiler
    _ = apply_lut_cubic_2d(lut_2d, image_2d)

    start_time = time.time()
    for _ in range(iterations):
        output_numba_2d = apply_lut_cubic_2d(lut_2d, image_2d)
    numba_time_2d = (time.time() - start_time) / iterations
    print("2D LUT - Average time per iteration (Numba cubic interpolation): {:.6f} seconds".format(numba_time_2d))

    start_time = time.time()
    for _ in range(iterations):
        output_scipy_2d = apply_lut_cubic_scipy(lut_2d, image_2d)
    scipy_time_2d = (time.time() - start_time) / iterations
    print("2D LUT - Average time per iteration (SciPy cubic interpolation): {:.6f} seconds".format(scipy_time_2d))

    output_ground_truth_2d = ground_truth_2d(image_2d)

    diff_2d = output_numba_2d - output_scipy_2d
    rmse_2d = np.sqrt(np.mean(diff_2d**2))
    max_error_2d = np.max(np.abs(diff_2d))
    print("2D LUT - RMSE error between Numba and SciPy outputs: {:.6e}".format(rmse_2d))
    print("2D LUT - Max absolute error between Numba and SciPy outputs: {:.6e}".format(max_error_2d))

    diff_ground_truth_2d = output_numba_2d - output_ground_truth_2d
    rmse_ground_truth_2d = np.sqrt(np.mean(diff_ground_truth_2d**2))
    max_error_ground_truth_2d = np.max(np.abs(diff_ground_truth_2d))
    print("2D LUT - RMSE error against ground truth: {:.6e}".format(rmse_ground_truth_2d))
    print("2D LUT - Max absolute error against ground truth: {:.6e}".format(max_error_ground_truth_2d))

    diff_scipy_ground_truth_2d = output_scipy_2d - output_ground_truth_2d
    rmse_scipy_ground_truth_2d = np.sqrt(np.mean(diff_scipy_ground_truth_2d**2))
    max_error_scipy_ground_truth_2d = np.max(np.abs(diff_scipy_ground_truth_2d))
    print("2D LUT - SciPy RMSE error against ground truth: {:.6e}".format(rmse_scipy_ground_truth_2d))
    print("2D LUT - SciPy max absolute error against ground truth: {:.6e}".format(max_error_scipy_ground_truth_2d))

    diff_norm_ground_truth_2d = np.sqrt(np.sum(diff_ground_truth_2d**2, axis=2))
    diff_norm_scipy_ground_truth_2d = np.sqrt(np.sum(diff_scipy_ground_truth_2d**2, axis=2))
    
    # For plotting, combine the two channels into one grayscale image (by taking the mean)
    image_2d_gray = np.mean(image_2d, axis=-1)
    output_ground_truth_2d_gray = np.mean(output_ground_truth_2d, axis=-1)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    input_im = axs[0, 0].imshow(image_2d_gray, interpolation='nearest', cmap='gray')
    axs[0, 0].set_title("Input Gradient Image (2D LUT, Mean)")
    axs[0, 0].axis("off")
    fig.colorbar(input_im, ax=axs[0, 0], fraction=0.046, pad=0.04)
    output_ground_truth_im = axs[0, 1].imshow(output_ground_truth_2d_gray, interpolation='nearest', cmap='gray')
    axs[0, 1].set_title("Output (Ground Truth, 2D LUT, Mean)")
    axs[0, 1].axis("off")
    fig.colorbar(output_ground_truth_im, ax=axs[0, 1], fraction=0.046, pad=0.04)
    im_numba = axs[1, 0].imshow(diff_norm_ground_truth_2d, cmap="hot", interpolation="nearest")
    axs[1, 0].set_title("Error Map (Numba vs Ground Truth, 2D LUT)")
    axs[1, 0].axis("off")
    fig.colorbar(im_numba, ax=axs[1, 0], fraction=0.046, pad=0.04)
    im_scipy = axs[1, 1].imshow(diff_norm_scipy_ground_truth_2d, cmap="hot", interpolation="nearest")
    axs[1, 1].set_title("Error Map (SciPy vs Ground Truth, 2D LUT)")
    axs[1, 1].axis("off")
    fig.colorbar(im_scipy, ax=axs[1, 1], fraction=0.046, pad=0.04)
    fig.suptitle("2D LUT Cubic Interpolation Comparison (x, y channels)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
