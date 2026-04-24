import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True)
def _boost_curve_kernel(
    x: np.ndarray,
    y: np.ndarray,
    inv_max_raw: float,
    a: float,
    raw_x0: float,
    boost_scale: float,
) -> None:
    h, w, c = x.shape
    for i in prange(h):
        for j in range(w):
            for ch in range(c):
                xv = x[i, j, ch]
                if xv <= raw_x0:
                    y[i, j, ch] = xv
                else:
                    dx = (xv - raw_x0) * inv_max_raw
                    b = boost_scale * (np.exp(a * dx) - a * dx - 1.0)
                    y[i, j, ch] = xv + b


def boost_highlights(
    x: np.ndarray,
    boost_ev: float = 0.0,
    boost_range: float = 0.1,
    protect_ev: float = 0.0,
    midgray: float = 0.184,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply the boost curve:

        x is interpreted in raw units, normalized internally by max(x),
        the boost curve is applied in normalized space, and the result is
        returned in the original raw units.

        y(x) = x                              for x <= raw_x0
        y(x) = x + b(x)                       for x > raw_x0
        b(x) = max_raw * k * (exp(a * dx) - a * dx - 1)
        dx = (x - raw_x0) / max_raw

    Parameters
    ----------
    x : np.ndarray
        Raw-domain image array, expected shape (..., ..., channels). Best with HxWx3.
    boost_ev : float, default 0.0, max recomended 20.0
        M, must be >= 0.
    boost_range : float, default 0.5
        A, must be in [0, 1].
    protect_ev : float, default 0.0, max recomended 3.0
        P, must be >= 0.
    midgray : float, default 0.184
        Raw-domain midgray reference, must be >= 0.
    out : np.ndarray | None
        Optional output buffer. If given, must match shape and dtype.

    Returns
    -------
    np.ndarray
        Transformed image in the same raw domain as x.
    """
    
    if boost_ev < 0:
        raise ValueError("boost_ev must be >= 0")
    if not (0.0 <= boost_range <= 1.0):
        raise ValueError("boost_range must be in [0, 1]")
    if protect_ev < 0:
        raise ValueError("protect_ev must be >= 0")
    if midgray < 0.0:
        raise ValueError("midgray must be >= 0")

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("x must be a 3D array, e.g. HxWxC")
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
        
    if out is None:
        y = np.empty_like(x)
    else:
        if out.shape != x.shape:
            raise ValueError("out must have the same shape as x")
        if out.dtype != x.dtype:
            raise ValueError("out must have the same dtype as x")
        if not out.flags["C_CONTIGUOUS"]:
            raise ValueError("out must be C-contiguous")
        y = out

    if boost_ev == 0:
        np.copyto(y, x)
        return y

    max_raw = np.max(x)
    if max_raw == 0.0:
        y.fill(0.0)
        return y

    raw_x0 = np.clip(midgray * (2.0 ** protect_ev), 0.0, max_raw)
    if raw_x0 == max_raw:
        np.copyto(y, x)
        return y

    a = 28.0 ** (1.0 - boost_range)
    x0 = raw_x0 / max_raw
    denom = np.exp(a * (1.0 - x0)) - a * (1.0 - x0) - 1.0
    if denom <= 0.0:
        raise ValueError("Invalid parameters: denominator for k is non-positive")

    k = (2.0 ** boost_ev - 1.0) / denom
    inv_max_raw = 1.0 / max_raw
    boost_scale = k * max_raw

    _boost_curve_kernel(x, y, inv_max_raw, a, raw_x0, boost_scale)
    return y


def warmup_boost_highlights() -> None:
    """Trigger Numba compilation for the highlight boost kernel."""
    sample = np.full((2, 2, 3), 1.0, dtype=np.float64)
    boost_highlights(sample, boost_ev=1.0, boost_range=0.5, protect_ev=0.0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    warmup_boost_highlights()

    plot_boost_ev = 10.0
    plot_boost_range = 0.5
    plot_protect_ev = 3.0
    plot_midgray = 0.184

    x_axis = np.geomspace(1.0e-6, 2.0 ** 10, 2048, dtype=np.float64)
    curve_input = np.repeat(x_axis[:, None, None], 3, axis=2)
    curve_output = boost_highlights(
        curve_input,
        boost_ev=plot_boost_ev,
        boost_range=plot_boost_range,
        protect_ev=plot_protect_ev,
        midgray=plot_midgray,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, curve_output[:, 0, 0], linewidth=2.0, label="boosted highlights")
    plt.plot(x_axis, x_axis, label="identity", linestyle="--", color="gray")
    plt.scatter(
        plot_midgray * (2.0 ** plot_protect_ev),
        plot_midgray * (2.0 ** plot_protect_ev),
        color="red",
        label="protected",
    )
    plt.scatter(plot_midgray, plot_midgray, color="green", label="midgray")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("input raw")
    plt.ylabel("output raw")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()