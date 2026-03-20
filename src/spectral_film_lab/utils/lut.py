import numpy as np
from spectral_film_lab.utils.fast_interp_lut import apply_lut_cubic_3d, apply_lut_cubic_2d

def _as_channel_bounds(bounds):
    bounds_array = np.asarray(bounds, dtype=np.float64)
    if bounds_array.ndim == 0:
        return np.full(3, bounds_array, dtype=np.float64)
    if bounds_array.shape == (3,):
        return bounds_array
    raise ValueError('bounds must be a scalar or length-3 sequence')


def _create_lut_3d(function, xmin=(0.0, 0.0, 0.0), xmax=(1.0, 1.0, 1.0), steps=32):
    xmin = _as_channel_bounds(xmin)
    xmax = _as_channel_bounds(xmax)
    x_r = np.linspace(xmin[0], xmax[0], steps, endpoint=True)
    x_g = np.linspace(xmin[1], xmax[1], steps, endpoint=True)
    x_b = np.linspace(xmin[2], xmax[2], steps, endpoint=True)
    X = np.meshgrid(x_r, x_g, x_b, indexing='ij')
    X = np.stack(X, axis=3)
    X = np.reshape(X, (steps**2, steps, 3)) # shape as an image to be compatible with image processing
    lut = np.reshape(function(X), (steps, steps, steps, 3))
    return lut

# def _create_lut_2d(function, xmin=0, xmax=1, steps=128):
#     x = np.linspace(xmin, xmax, steps, endpoint=True)
#     X = np.meshgrid(x,x, indexing='ij')
#     X = np.stack(X, axis=3)
#     X = np.reshape(X, (steps, steps, 3)) # shape as an image to be compatible with image processing
#     lut = np.reshape(function(X), (steps, steps, 3))
#     return lut

def compute_with_lut(data, function, xmin=(0.0, 0.0, 0.0), xmax=(1.0, 1.0, 1.0), steps=32):
    # Computes the function on the data using a 3D LUT for acceleration.
    # The data is assumed to be in the range [xmin, xmax] and will be normalized for LUT indexing.
    # The lut is created on the fly in the range [xmin, xmax] with the specified number of steps.
    # Note: apply_lut_cubic_3d expects the data to be normalized to [0, 1] for proper indexing into the LUT.
    xmin = _as_channel_bounds(xmin)
    xmax = _as_channel_bounds(xmax)
    if np.any(xmax <= xmin):
        raise ValueError('xmax must be greater than xmin')
    lut = _create_lut_3d(function, xmin, xmax, steps)
    data_normalized = (data - xmin) / (xmax - xmin)
    return apply_lut_cubic_3d(lut, data_normalized), lut

def warmup_luts():
    """
    Performs a warmup for both 3D and 2D LUT JIT functions.
    This ensures that the Numba JIT compilation overhead is incurred only once.
    """
    L = 32
    grid = np.linspace(0, 1, L, dtype=np.float64)
    
    # --- Warmup 3D LUT ---
    R, G, B = np.meshgrid(grid, grid, grid, indexing='ij')
    lut_3d = np.stack((R**2, G**2, B**2), axis=-1)  # 3D LUT: shape (L,L,L,3)
    height, width = 128, 128
    x = np.linspace(0, 1, width, dtype=np.float64)
    y = np.linspace(0, 1, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    image_3d = np.stack((X, Y, 0.5 * np.ones_like(X)), axis=-1)
    _ = apply_lut_cubic_3d(lut_3d, image_3d)
    
    # --- Warmup 2D LUT ---
    # Define a 2D LUT mapping (x,y) chromaticities to RGB.
    L = 128
    grid = np.linspace(0, 1, L, dtype=np.float64)
    lut_2d = np.empty((L, L, 3), dtype=np.float64)
    X2, Y2 = np.meshgrid(grid, grid, indexing='ij')
    lut_2d[..., 0] = X2**2         # R = x^2
    lut_2d[..., 1] = Y2**2         # G = y^2
    lut_2d[..., 2] = (X2 + Y2) / 2.0  # B = (x+y)/2
    # Create a synthetic image of chromaticities (2 channels).
    image_2d = np.stack((X, Y), axis=-1)
    _ = apply_lut_cubic_2d(lut_2d, image_2d)

if __name__=='__main__':
    import matplotlib.pyplot as plt

    def run_quick_test(label, xmin=0.0, xmax=1.0):
        sample_data = np.random.uniform(xmin, xmax, size=(300, 200, 3))
        data_finterp, lut3d = compute_with_lut(sample_data, mycalculation, xmin=xmin, xmax=xmax)
        error = mycalculation(sample_data) - data_finterp
        print(f'{label} range [{xmin}, {xmax}]')
        print('  Max interpolation error:', np.max(error))
        print('  Mean interpolation error:', np.mean(np.abs(error)))
        print('  Max LUT value:', np.max(lut3d))
        print('  Min LUT value:', np.min(lut3d))
        print('  Max computed value:', np.max(data_finterp))
        print('  Min computed value:', np.min(data_finterp))
        
    def mycalculation(x):
        y = np.zeros_like(x)
        y[:,:,0] = 3*x[:,:,1] + x[:,:,0]
        y[:,:,1] = 3*x[:,:,2] + x[:,:,1]
        y[:,:,2] = 3*x[:,:,0] + x[:,:,2]
        return y

    warmup_luts()
    np.random.seed(0)
    run_quick_test('Default')
    run_quick_test('Extended', xmin=-1.0, xmax=2.0)
    plt.show()
