import numpy as np
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
def linear_interp_lut_at_3d(lut, r, g, b):
    """
    Performs trilinear interpolation at a single point in a 3D LUT.
    Used as a boundary-safe fallback for the outermost LUT cells.
    """
    L = lut.shape[0]
    r = clamp_coordinate(r, L)
    g = clamp_coordinate(g, L)
    b = clamp_coordinate(b, L)

    r0 = int(np.floor(r))
    g0 = int(np.floor(g))
    b0 = int(np.floor(b))
    r1 = min(r0 + 1, L - 1)
    g1 = min(g0 + 1, L - 1)
    b1 = min(b0 + 1, L - 1)

    tr = r - r0
    tg = g - g0
    tb = b - b0

    out = np.zeros(3, dtype=np.float64)
    for i in range(2):
        ri = r0 if i == 0 else r1
        wr = (1.0 - tr) if i == 0 else tr
        for j in range(2):
            gj = g0 if j == 0 else g1
            wg = (1.0 - tg) if j == 0 else tg
            for k in range(2):
                bk = b0 if k == 0 else b1
                wb = (1.0 - tb) if k == 0 else tb
                weight = wr * wg * wb
                out[0] += weight * lut[ri, gj, bk, 0]
                out[1] += weight * lut[ri, gj, bk, 1]
                out[2] += weight * lut[ri, gj, bk, 2]
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

    out = np.zeros(channels, dtype=np.float64)
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
def cubic_interp_lut_at_3d(lut, r, g, b):
    """
    Performs cubic interpolation at a single point (r, g, b) in a 3D LUT (shape: LxLxLxC)
    using the Mitchell–Netravali kernel.
    """
    L = lut.shape[0]
    if L < 4 or r < 1.0 or g < 1.0 or b < 1.0 or r >= (L - 2) or g >= (L - 2) or b >= (L - 2):
        return linear_interp_lut_at_3d(lut, r, g, b)

    r_base = int(np.floor(r))
    g_base = int(np.floor(g))
    b_base = int(np.floor(b))
    r_frac = r - r_base
    g_frac = g - g_base
    b_frac = b - b_base

    # Precompute kernel weights for each dimension.
    wr = np.empty(4, dtype=np.float64)
    wg = np.empty(4, dtype=np.float64)
    wb = np.empty(4, dtype=np.float64)
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

    # Accumulate weighted sum over the 4x4x4 neighborhood.
    out = np.zeros(3, dtype=np.float64)
    weight_sum = 0.0
    for i in range(4):
        ri = safe_index(r_base - 1 + i, L)
        for j in range(4):
            gj = safe_index(g_base - 1 + j, L)
            for k in range(4):
                bk = safe_index(b_base - 1 + k, L)
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

@njit(parallel=True, cache=True)
def apply_lut_cubic_3d(lut, image):
    """
    Applies a 3D LUT (shape: LxLxLx3) to an image (shape: HxWx3) using cubic interpolation.
    Data is assumed to be normalized in the range [0, 1] and will be scaled to [0, L-1] for LUT indexing.
    """
    height, width, _ = image.shape
    output = np.empty((height, width, 3), dtype=np.float64)
    L = lut.shape[0]
    for i in prange(height):
        for j in range(width):
            r_in = image[i, j, 0] * (L - 1)
            g_in = image[i, j, 1] * (L - 1)
            b_in = image[i, j, 2] * (L - 1)
            out_val = cubic_interp_lut_at_3d(lut, r_in, g_in, b_in)
            output[i, j, 0] = out_val[0]
            output[i, j, 1] = out_val[1]
            output[i, j, 2] = out_val[2]
    return output

# ---------------------------
# 2D LUT Cubic Interpolation (using x, y channels)
# ---------------------------
@njit(cache=True)
def cubic_interp_lut_at_2d(lut, x, y):
    """
    Performs cubic interpolation at a single point (x, y) in a 2D LUT (shape: LxLxC)
    using the Mitchell–Netravali kernel.
    Here, x and y are the input coordinates.
    """
    L = lut.shape[0]
    channels = lut.shape[2]
    if L < 4 or x < 1.0 or y < 1.0 or x >= (L - 2) or y >= (L - 2):
        return linear_interp_lut_at_2d(lut, x, y)

    x_base = int(np.floor(x))
    y_base = int(np.floor(y))
    x_frac = x - x_base
    y_frac = y - y_base

    # Compute kernel weights for the x and y dimensions.
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

    # Accumulate weighted sum over the 4x4 neighborhood.
    out = np.zeros(channels, dtype=np.float64)
    weight_sum = 0.0
    for i in range(4):
        xi = safe_index(x_base - 1 + i, L)
        for j in range(4):
            yj = safe_index(y_base - 1 + j, L)
            weight = wx[i] * wy[j]
            weight_sum += weight
            for c in range(channels):
                out[c] += weight * lut[xi, yj, c]
    if weight_sum != 0.0:
        for c in range(channels):
            out[c] /= weight_sum
    return out

@njit(parallel=True, cache=True)
def apply_lut_cubic_2d(lut, image):
    """
    Applies a 2D LUT (shape: LxLxC) to an image (shape: HxWxC) using cubic interpolation.
    Here the image channels represent the (x, y) coordinates.
    """
    height, width, _ = image.shape
    channels = lut.shape[2]
    output = np.empty((height, width, channels), dtype=np.float64)
    L = lut.shape[0]
    for i in prange(height):
        for j in range(width):
            x_in = image[i, j, 0] * (L - 1)
            y_in = image[i, j, 1] * (L - 1)
            out_val = cubic_interp_lut_at_2d(lut, x_in, y_in)
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

    # --- 3D LUT Example ---
    lut_size_3d = 17
    grid_3d = np.linspace(0, 1, lut_size_3d, dtype=np.float64)
    grid_r_3d, grid_g_3d, grid_b_3d = np.meshgrid(grid_3d, grid_3d, grid_3d, indexing='ij')
    # Create a 3D LUT that applies a simple non-linear transformation (r^2, g^2, b^2)
    lut_3d = np.stack((grid_r_3d**2, grid_g_3d**2, grid_b_3d**2), axis=-1)  # shape: (L, L, L, 3)

    # Create a synthetic test image (gradient image, 3 channels)
    image_height, image_width = 512, 512
    x_axis_3d = np.linspace(0, 1, image_width, dtype=np.float64)
    y_axis_3d = np.linspace(0, 1, image_height, dtype=np.float64)
    grid_x_3d, grid_y_3d = np.meshgrid(x_axis_3d, y_axis_3d)
    image_3d = np.stack((grid_x_3d, grid_y_3d, 0.5 * np.ones_like(grid_x_3d)), axis=-1)

    # Warm up the JIT compiler
    _ = apply_lut_cubic_3d(lut_3d, image_3d)

    iterations = 10
    start_time = time.time()
    for _ in range(iterations):
        output_numba_3d = apply_lut_cubic_3d(lut_3d, image_3d)
    numba_time_3d = (time.time() - start_time) / iterations
    print("3D LUT - Average time per iteration (Numba cubic interpolation): {:.6f} seconds".format(numba_time_3d))

    start_time = time.time()
    for _ in range(iterations):
        output_scipy_3d = apply_lut_cubic_scipy(lut_3d, image_3d)
    scipy_time_3d = (time.time() - start_time) / iterations
    print("3D LUT - Average time per iteration (SciPy cubic interpolation): {:.6f} seconds".format(scipy_time_3d))

    diff_3d = output_numba_3d - output_scipy_3d
    rmse_3d = np.sqrt(np.mean(diff_3d**2))
    max_error_3d = np.max(np.abs(diff_3d))
    print("3D LUT - RMSE error between Numba and SciPy outputs: {:.6e}".format(rmse_3d))
    print("3D LUT - Max absolute error between Numba and SciPy outputs: {:.6e}".format(max_error_3d))

    diff_norm_3d = np.sqrt(np.sum(diff_3d**2, axis=2))
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    input_im = axs[0, 0].imshow(image_3d, interpolation='nearest')
    axs[0, 0].set_title("Input Gradient Image (3D LUT)")
    axs[0, 0].axis("off")
    output_numba_im = axs[0, 1].imshow(output_numba_3d, interpolation='nearest')
    axs[0, 1].set_title("Output (Numba, 3D LUT)")
    axs[0, 1].axis("off")
    output_scipy_im = axs[1, 0].imshow(output_scipy_3d, interpolation='nearest')
    axs[1, 0].set_title("Output (SciPy, 3D LUT)")
    axs[1, 0].axis("off")
    im = axs[1, 1].imshow(diff_norm_3d, cmap="hot", interpolation="nearest")
    axs[1, 1].set_title("Error Map (3D LUT)")
    axs[1, 1].axis("off")
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
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

    diff_2d = output_numba_2d - output_scipy_2d
    rmse_2d = np.sqrt(np.mean(diff_2d**2))
    max_error_2d = np.max(np.abs(diff_2d))
    print("2D LUT - RMSE error between Numba and SciPy outputs: {:.6e}".format(rmse_2d))
    print("2D LUT - Max absolute error between Numba and SciPy outputs: {:.6e}".format(max_error_2d))

    diff_norm_2d = np.sqrt(np.sum(diff_2d**2, axis=2))
    
    # For plotting, combine the two channels into one grayscale image (by taking the mean)
    image_2d_gray = np.mean(image_2d, axis=-1)
    output_numba_2d_gray = np.mean(output_numba_2d, axis=-1)
    output_scipy_2d_gray = np.mean(output_scipy_2d, axis=-1)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    input_im = axs[0, 0].imshow(image_2d_gray, interpolation='nearest', cmap='gray')
    axs[0, 0].set_title("Input Gradient Image (2D LUT, Mean)")
    axs[0, 0].axis("off")
    fig.colorbar(input_im, ax=axs[0, 0], fraction=0.046, pad=0.04)
    output_numba_im = axs[0, 1].imshow(output_numba_2d_gray, interpolation='nearest', cmap='gray')
    axs[0, 1].set_title("Output (Numba, 2D LUT, Mean)")
    axs[0, 1].axis("off")
    fig.colorbar(output_numba_im, ax=axs[0, 1], fraction=0.046, pad=0.04)
    output_scipy_im = axs[1, 0].imshow(output_scipy_2d_gray, interpolation='nearest', cmap='gray')
    axs[1, 0].set_title("Output (SciPy, 2D LUT, Mean)")
    axs[1, 0].axis("off")
    fig.colorbar(output_scipy_im, ax=axs[1, 0], fraction=0.046, pad=0.04)
    im = axs[1, 1].imshow(diff_norm_2d, cmap="hot", interpolation="nearest")
    axs[1, 1].set_title("Error Map (2D LUT)")
    axs[1, 1].axis("off")
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
    fig.suptitle("2D LUT Cubic Interpolation Comparison (x, y channels)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
