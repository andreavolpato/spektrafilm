import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _gaussian_kernel_1d(sigma, truncate):
    radius = int(truncate * sigma + 0.5)
    size = 2 * radius + 1
    kernel = np.empty(size, dtype=np.float64)
    total = 0.0
    for i in range(size):
        x = i - radius
        val = np.exp(-0.5 * (x / sigma) ** 2)
        kernel[i] = val
        total += val
    for i in range(size):
        kernel[i] /= total
    return kernel, radius


@njit(inline='always', cache=True)
def _reflect(i, n):
    # scipy.ndimage mode='reflect': (d c b a | a b c d | d c b a).
    if 0 <= i < n:
        return i
    if -n <= i < 0:
        return -i - 1
    if n <= i < 2 * n:
        return 2 * n - 1 - i
    period = 2 * n
    i = i % period
    if i < 0:
        i += period
    if i >= n:
        i = period - 1 - i
    return i


# ---------------------------------------------------------------------------
# Small-sigma path: fused tile-based separable FIR
#
# Strip-fused vertical+horizontal convolution. Each parallel task processes
# a horizontal strip and keeps its intermediate in a per-strip buffer that
# fits in L2, so the input image is read once and the output written once
# per strip — about half the memory traffic of the unfused two-pass version.
# ---------------------------------------------------------------------------

_STRIP_H = 16


@njit(parallel=True, cache=True, fastmath=True)
def _fir_2d_fused(image, output, kernel, radius):
    n, m = image.shape
    strip_h = _STRIP_H
    n_strips = (n + strip_h - 1) // strip_h
    for s in prange(n_strips):
        i0 = s * strip_h
        i1 = min(i0 + strip_h, n)
        sn = i1 - i0
        tmp = np.zeros((sn, m), dtype=image.dtype)
        # Vertical pass: accumulate image rows weighted by the kernel into tmp.
        for k in range(-radius, radius + 1):
            kw = kernel[k + radius]
            for li in range(sn):
                ii = _reflect(i0 + li + k, n)
                for j in range(m):
                    tmp[li, j] += image[ii, j] * kw
        # Horizontal pass: split into reflected edges + reflect-free interior.
        if 2 * radius >= m:
            for li in range(sn):
                i = i0 + li
                for j in range(m):
                    sval = 0.0
                    for k in range(-radius, radius + 1):
                        jj = _reflect(j + k, m)
                        sval += tmp[li, jj] * kernel[k + radius]
                    output[i, j] = sval
        else:
            for li in range(sn):
                i = i0 + li
                for j in range(radius):
                    sval = 0.0
                    for k in range(-radius, radius + 1):
                        jj = _reflect(j + k, m)
                        sval += tmp[li, jj] * kernel[k + radius]
                    output[i, j] = sval
                for j in range(radius, m - radius):
                    sval = 0.0
                    for k in range(-radius, radius + 1):
                        sval += tmp[li, j + k] * kernel[k + radius]
                    output[i, j] = sval
                for j in range(m - radius, m):
                    sval = 0.0
                    for k in range(-radius, radius + 1):
                        jj = _reflect(j + k, m)
                        sval += tmp[li, jj] * kernel[k + radius]
                    output[i, j] = sval


def _gaussian_filter_2d_small(image, sigma, truncate):
    if sigma <= 0.0:
        return image.copy()
    kernel, radius = _gaussian_kernel_1d(float(sigma), float(truncate))
    output = np.empty_like(image)
    _fir_2d_fused(image, output, kernel, radius)
    return output


# ---------------------------------------------------------------------------
# Large-sigma path: Young & van Vliet 2002 IIR Gaussian
#
# Recursive 3rd-order IIR with forward + backward sweep. Cost is O(1) per
# pixel regardless of sigma — for sigma=20 this replaces a 121-tap FIR
# with ~10 multiply-adds per pixel. Edge handling is sample-replication;
# max error vs the analytic Gaussian is around 1e-3.
# ---------------------------------------------------------------------------

def _yvv_coeffs(sigma):
    if sigma >= 2.5:
        q = 0.98711 * sigma - 0.96330
    else:
        q = 3.97156 - 4.14554 * np.sqrt(1.0 - 0.26891 * sigma)
    q2 = q * q
    q3 = q2 * q
    b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3
    b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3
    b2 = -(1.4281 * q2 + 1.26661 * q3)
    b3 = 0.422205 * q3
    B = 1.0 - (b1 + b2 + b3) / b0
    return B, b1 / b0, b2 / b0, b3 / b0


@njit(parallel=True, cache=True, fastmath=True)
def _iir_horizontal(image, output, B, B1, B2, B3):
    n, m = image.shape
    for i in prange(n):
        x0 = image[i, 0]
        w1 = x0
        w2 = x0
        w3 = x0
        for j in range(m):
            w = B * image[i, j] + B1 * w1 + B2 * w2 + B3 * w3
            output[i, j] = w
            w3 = w2
            w2 = w1
            w1 = w
        xn = output[i, m - 1]
        y1 = xn
        y2 = xn
        y3 = xn
        for j in range(m - 1, -1, -1):
            y = B * output[i, j] + B1 * y1 + B2 * y2 + B3 * y3
            output[i, j] = y
            y3 = y2
            y2 = y1
            y1 = y


_IIR_COL_CHUNK = 32


@njit(parallel=True, cache=True, fastmath=True)
def _iir_vertical(image, output, B, B1, B2, B3):
    # Process columns in chunks: state vectors carry the per-column
    # recurrence while the inner k-loop iterates contiguously across the
    # chunk so reads/writes touch adjacent memory each row.
    n, m = image.shape
    chunk = _IIR_COL_CHUNK
    n_chunks = (m + chunk - 1) // chunk
    for c in prange(n_chunks):
        j0 = c * chunk
        j1 = min(j0 + chunk, m)
        cs = j1 - j0
        s1 = np.empty(cs, dtype=np.float64)
        s2 = np.empty(cs, dtype=np.float64)
        s3 = np.empty(cs, dtype=np.float64)
        for k in range(cs):
            x0 = image[0, j0 + k]
            s1[k] = x0
            s2[k] = x0
            s3[k] = x0
        for i in range(n):
            for k in range(cs):
                x = image[i, j0 + k]
                w = B * x + B1 * s1[k] + B2 * s2[k] + B3 * s3[k]
                output[i, j0 + k] = w
                s3[k] = s2[k]
                s2[k] = s1[k]
                s1[k] = w
        for k in range(cs):
            xn = output[n - 1, j0 + k]
            s1[k] = xn
            s2[k] = xn
            s3[k] = xn
        for i in range(n - 1, -1, -1):
            for k in range(cs):
                x = output[i, j0 + k]
                y = B * x + B1 * s1[k] + B2 * s2[k] + B3 * s3[k]
                output[i, j0 + k] = y
                s3[k] = s2[k]
                s2[k] = s1[k]
                s1[k] = y


def _gaussian_filter_2d_large(image, sigma):
    if sigma <= 0.0:
        return image.copy()
    if sigma < 0.5:
        # IIR coefficients are unstable below 0.5 — fall back to direct FIR.
        return _gaussian_filter_2d_small(image, sigma, 3.0)
    B, B1, B2, B3 = _yvv_coeffs(float(sigma))
    tmp = np.empty_like(image)
    output = np.empty_like(image)
    _iir_horizontal(image, tmp, B, B1, B2, B3)
    _iir_vertical(tmp, output, B, B1, B2, B3)
    return output


# ---------------------------------------------------------------------------
# Dispatch + public API
# ---------------------------------------------------------------------------

# At sigma ~3 the FIR (~38 ops/pixel) and IIR (~40 ops/pixel) cross over.
# Above this, IIR is asymptotically free w.r.t. sigma; below, FIR wins on
# its tighter inner loop.
SMALL_SIGMA_MAX = 3.0


def _dispatch_2d(image, sigma, truncate):
    s = float(sigma)
    if s >= SMALL_SIGMA_MAX:
        return _gaussian_filter_2d_large(image, s)
    return _gaussian_filter_2d_small(image, s, truncate)


def _apply_per_channel(image, sigma, truncate, filter_2d):
    image = np.ascontiguousarray(image)
    if image.ndim == 2:
        return filter_2d(image, sigma, truncate)
    if image.ndim == 3:
        _, _, c = image.shape
        if np.ndim(sigma) == 0:
            sigmas = np.full(c, float(sigma), dtype=np.float64)
        else:
            sigmas = np.asarray(sigma, dtype=np.float64).ravel()
            if sigmas.shape[0] != c:
                raise ValueError(
                    "sigma length {} does not match channel count {}".format(
                        sigmas.shape[0], c
                    )
                )
        output = np.empty_like(image)
        for ch in range(c):
            ch_in = np.ascontiguousarray(image[:, :, ch])
            output[:, :, ch] = filter_2d(ch_in, sigmas[ch], truncate)
        return output
    raise ValueError("Unsupported image dimension: {}".format(image.ndim))


def fast_gaussian_filter(image, sigma, truncate=3.0):
    """Per-channel 2D Gaussian filter (mode='reflect').

    Auto-dispatches by sigma: small-sigma fused FIR for sigma < 3, recursive
    IIR (Young-van Vliet) for sigma >= 3. Pass truncate=4.0 for the
    SciPy-equivalent FIR kernel width on the small-sigma path.
    """
    return _apply_per_channel(image, sigma, truncate, _dispatch_2d)


def fast_gaussian_filter_small(image, sigma, truncate=3.0):
    """Direct fused FIR — fastest path when sigma is bounded to ~a few pixels.

    Use for unsharp-mask radii or any blur with sigma roughly <= 3 px.
    Single fused tile pass: half the memory traffic of the unfused
    separable version. Quality matches a SciPy FIR Gaussian at the same
    truncate to within fastmath rounding.
    """
    return _apply_per_channel(image, sigma, truncate, _gaussian_filter_2d_small)


def fast_gaussian_filter_large(image, sigma):
    """IIR (Young-van Vliet) Gaussian — fastest path for large sigma.

    Use for halation-style blurs or any sigma >= ~3 px. Cost is O(1) per
    pixel regardless of sigma. Max error vs analytic Gaussian ~1e-3, with
    minor edge approximation from sample-replication boundary handling.
    """
    return _apply_per_channel(image, sigma, 0.0, lambda img, s, _t: _gaussian_filter_2d_large(img, s))


def warmup_fast_gaussian_filter():
    dummy2d = np.random.rand(64, 64).astype(np.float64)
    dummy3d = np.random.rand(64, 64, 3).astype(np.float64)
    fast_gaussian_filter(dummy2d, 1.0)
    fast_gaussian_filter(dummy2d, 5.0)
    fast_gaussian_filter(dummy3d, 1.0)
    fast_gaussian_filter(dummy3d, 5.0)
    fast_gaussian_filter(dummy3d, np.array([0.5, 1.0, 1.5]))


if __name__ == '__main__':
    import time
    from scipy.ndimage import gaussian_filter

    print("Warming up...")
    warmup_fast_gaussian_filter()

    img2d = np.random.rand(6000, 4000).astype(np.float64)
    img3d = np.random.rand(6000, 4000, 3).astype(np.float64)
    iterations = 3

    def bench(fn, *args):
        # Warm + time
        fn(*args)
        t0 = time.time()
        for _ in range(iterations):
            fn(*args)
        return (time.time() - t0) / iterations

    # ---------- small sigma ----------
    sigma = 1.0
    truncate = 4.0
    fast2d = fast_gaussian_filter_small(img2d, sigma, truncate)
    ref2d = gaussian_filter(img2d, sigma, truncate=truncate, mode='reflect')
    print("Small sigma=%.1f, 2D max err vs SciPy: %.2e" % (sigma, np.abs(fast2d - ref2d).max()))

    t_fast = bench(fast_gaussian_filter_small, img2d, sigma, truncate)
    t_scipy = bench(lambda x: gaussian_filter(x, sigma, truncate=truncate, mode='reflect'), img2d)
    print("  2D  fast_small: %.4fs  scipy: %.4fs  speedup: %.2fx" % (t_fast, t_scipy, t_scipy / t_fast))

    t_fast = bench(fast_gaussian_filter_small, img3d, sigma, truncate)
    def scipy_3d_small(x):
        out = np.empty_like(x)
        for ch in range(x.shape[2]):
            out[:, :, ch] = gaussian_filter(x[:, :, ch], sigma, truncate=truncate, mode='reflect')
        return out
    t_scipy = bench(scipy_3d_small, img3d)
    print("  3D  fast_small: %.4fs  scipy: %.4fs  speedup: %.2fx" % (t_fast, t_scipy, t_scipy / t_fast))

    # ---------- large sigma ----------
    sigma = 20.0
    fast2d = fast_gaussian_filter_large(img2d, sigma)
    ref2d = gaussian_filter(img2d, sigma, truncate=4.0, mode='reflect')
    err = np.abs(fast2d - ref2d).max()
    rel = err / np.abs(ref2d).max()
    print("Large sigma=%.1f, 2D max err vs SciPy: %.2e (rel %.2e)" % (sigma, err, rel))

    t_fast = bench(fast_gaussian_filter_large, img2d, sigma)
    t_scipy = bench(lambda x: gaussian_filter(x, sigma, truncate=4.0, mode='reflect'), img2d)
    print("  2D  fast_large: %.4fs  scipy: %.4fs  speedup: %.2fx" % (t_fast, t_scipy, t_scipy / t_fast))

    t_fast = bench(fast_gaussian_filter_large, img3d, sigma)
    def scipy_3d_large(x):
        out = np.empty_like(x)
        for ch in range(x.shape[2]):
            out[:, :, ch] = gaussian_filter(x[:, :, ch], sigma, truncate=4.0, mode='reflect')
        return out
    t_scipy = bench(scipy_3d_large, img3d)
    print("  3D  fast_large: %.4fs  scipy: %.4fs  speedup: %.2fx" % (t_fast, t_scipy, t_scipy / t_fast))
