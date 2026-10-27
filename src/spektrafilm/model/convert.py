"""Convert a scanned negative back to film cmy density (the inverse scan map).

The "convert-film" front-end of the invert workflow (see WorkflowParams). A
scanned negative arrives as scene-referred linear RGB (in the pipeline input
colour space); this module recovers the film cmy densities that, under the scan
model, would have produced it — so they can be injected at the cmy_film tap and
printed. (This is *film-scan conversion*, not a colour-space conversion utility.)

The scan model is the same Beer-Lambert sandwich the scanning stage uses for
scan_film:

    D(λ)  = base(λ) + Σ_c cmy_c · dye_c(λ)
    L(λ)  = illuminant(λ) · 10^(−D(λ))
    XYZ   = ∫ L(λ) · CMF(λ) dλ
    RGB   = M · XYZ            (M: XYZ -> input colour space, linear)

so this is the exact inverse of that map (validated to ~1e-10 against the runtime
scan_film path in the c40_film_inversion study). It is inverted per pixel by a
vectorised, box-bounded damped Gauss-Newton, with an analytic log-linear seed:
because RGB ≈ A·10^(−dye·cmy), one log + one 3×3 lands inside the basin, and 2–3
Newton steps reach the exact density. There is no LUT and nothing cached — the
scan illuminant and base are tunable, so the map is re-solved on the fly.

The scan illuminant is a selectable physical input (the capture rig's light); the
base / orange mask is the creative lever. Both only enter through the product
illuminant(λ)·10^(−base(λ)), so they share one spectral axis (see the c40 n060
design note).
"""
from __future__ import annotations

import re

import colour
import numpy as np

from spektrafilm.config import STANDARD_OBSERVER_CMFS
from spektrafilm.model.develop import compute_density_spectral

LN10 = np.log(10.0)


def parse_calibration_matrix(text) -> np.ndarray:
    """Parse a 9-number string into a (3, 3) row-major matrix.

    Accepts any whitespace / comma / semicolon / bracket separators (so a value
    copied out of the GUI, a numpy repr, or a hand-typed list all work). Returns
    the 3x3 identity on anything that is not exactly nine finite numbers, so a
    malformed string is a no-op rather than an error.
    """
    try:
        nums = [float(t) for t in re.split(r"[\s,;\[\]()]+", str(text).strip()) if t]
    except (TypeError, ValueError):
        nums = []
    if len(nums) == 9 and all(np.isfinite(nums)):
        return np.asarray(nums, float).reshape(3, 3)
    return np.eye(3)


def format_calibration_matrix(matrix) -> str:
    """Format a (3, 3) matrix as the canonical 9-number calibration string."""
    m = np.asarray(matrix, float).reshape(3, 3)
    return "  ".join(" ".join(f"{v:.5f}" for v in row) for row in m)


def sample_base_rgb(image, percentile: float = 99.0) -> np.ndarray:
    """Mean RGB of the brightest pixels — the clear / unexposed film base.

    The clear base transmits the most light, so the top ``(100 - percentile)``%
    by luminance is the unexposed film (rebate / gaps between frames). Returns the
    mean RGB of that set.
    """
    rgb = np.asarray(image, float).reshape(-1, 3)
    rgb = rgb[np.all(np.isfinite(rgb), axis=1)]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    thresh = np.quantile(lum, np.clip(percentile, 0.0, 100.0) / 100.0)
    bright = rgb[lum >= thresh]
    return bright.mean(axis=0) if len(bright) else rgb.mean(axis=0)


def fit_base_params(target_rgb, channel_density, base_density, base_params,
                    scan_illuminant, output_colourspace, w_ridge=0.02):
    """Fit the film base / orange-mask tint so the cmy=0 scan reproduces a target RGB.

    Solves the base ``cyan/magenta/yellow`` tuning (and an exposure EV for the
    overall level) so the model's clear-film colour (cmy=0, base only) matches the
    measured clear-base RGB — anchoring the unexposed film to film density 0. The
    base enters exactly as the scan stage applies it (``compute_density_spectral``
    with ``is_film=True``), so the fit is consistent with the converter.

    Returns ``(fitted_base_params, exposure_ev)``.
    """
    from dataclasses import replace

    from scipy.optimize import least_squares

    target = np.asarray(target_rgb, float).reshape(3)
    cmfs = np.asarray(STANDARD_OBSERVER_CMFS[:], float)
    illum = np.asarray(scan_illuminant, float)
    dye = np.asarray(channel_density, float)
    base = np.asarray(base_density, float)
    norm = float(np.sum(illum * cmfs[:, 1]))
    illum_xy = colour.XYZ_to_xy((illum @ cmfs) / norm)
    M = colour.XYZ_to_RGB(
        np.eye(3), colourspace=output_colourspace,
        apply_cctf_encoding=False, illuminant=illum_xy,
    ).T                                                                  # (3,3)
    valid_dye = np.all(np.isfinite(dye), axis=1)

    def forward_zero(cmy3):
        params = replace(base_params, active=True,
                         cyan=float(cmy3[0]), magenta=float(cmy3[1]), yellow=float(cmy3[2]))
        offset = compute_density_spectral(
            dye, np.zeros((1, 1, 3)), base, base_density_params=params, is_film=True)[0, 0]
        valid = valid_dye & np.isfinite(offset)
        L = np.where(valid, illum, 0.0) * 10.0 ** (-np.where(valid, offset, 0.0))
        return ((L @ cmfs) / norm) @ M.T                                # (3,)

    def gain_for(f):
        denom = float(target @ target)
        return float(f @ target) / denom if denom > 0 else 1.0

    def resid(x):
        f = forward_zero(x)
        return np.concatenate([f - gain_for(f) * target, w_ridge * (x - 1.0)])

    x0 = np.array([base_params.cyan, base_params.magenta, base_params.yellow], float)
    sol = least_squares(resid, x0, method="lm")
    f = forward_zero(sol.x)
    ev = float(np.log2(max(gain_for(f), 1e-9)))
    fitted = replace(base_params, active=True,
                     cyan=float(sol.x[0]), magenta=float(sol.x[1]), yellow=float(sol.x[2]))
    return fitted, ev


class FilmConverter:
    """Vectorised inverse of the scan_film colour map: RGB (..., 3) -> cmy (..., 3).

    Parameters
    ----------
    channel_density : (Nw, 3) dye spectral densities.
    base_density : (Nw,) film base + fog spectral density.
    base_params : FilmBaseParams (the base / orange-mask tuning), applied exactly
        as ``compute_density_spectral`` does in the scan stage.
    scan_illuminant : (Nw,) spectral values of the capture rig's light.
    output_colourspace : the pipeline input colour space the scan lives in.
    cmy_max : (3,) per-channel Dmax bounding the physical cmy cube.
    exposure_ev : aligns the scan's absolute level to the model (gain = 2**ev,
        applied to the input before inversion).
    calibration : (3, 3) device-correction matrix applied to the input RGB
        (``rgb @ calibration``) before inversion, to undo the scanner/camera
        colour rendering vs the standard observer over the film dyes. ``None``
        means identity (no correction). See :func:`fit_blind_calibration`.
    dtype : working precision (float32 is exact to ~1e-4 in cmy; dE ≈ 0).
    """

    def __init__(self, channel_density, base_density, base_params, scan_illuminant,
                 output_colourspace, cmy_max, exposure_ev=0.0, calibration=None,
                 dtype=np.float32):
        self.dtype = dtype
        cmfs = np.asarray(STANDARD_OBSERVER_CMFS[:], float)                 # (Nw,3)
        illum = np.asarray(scan_illuminant, float)                          # (Nw,)
        dye = np.asarray(channel_density, float)                            # (Nw,3)
        self._norm = float(np.sum(illum * cmfs[:, 1]))
        illum_xy = colour.XYZ_to_xy((illum @ cmfs) / self._norm)

        # base offset exactly as the scan stage applies it (incl. base tuning)
        base_offset = compute_density_spectral(
            dye, np.zeros((1, 1, 3)), np.asarray(base_density, float),
            base_density_params=base_params,
            is_film=True,
        )[0, 0]                                                             # (Nw,)
        # invalid wavelengths (NaN dye/base) contribute no light, matching
        # density_to_light's NaN handling in the forward stage.
        valid = np.all(np.isfinite(dye), axis=1) & np.isfinite(base_offset)

        dye = np.where(valid[:, None], dye, 0.0)
        self.dyeT = np.ascontiguousarray(dye.T, dtype=dtype)                # (3,Nw)
        self.cmfs = np.ascontiguousarray(cmfs, dtype=dtype)                 # (Nw,3)
        self.norm = dtype(self._norm)
        self.illum_valid = np.where(valid, illum, 0.0).astype(dtype)        # (Nw,)
        self.base_safe = np.where(valid, base_offset, 0.0).astype(dtype)    # (Nw,)
        # linear XYZ -> input-space RGB operator (CAT + matrix, no offset)
        self._M = colour.XYZ_to_RGB(
            np.eye(3), colourspace=output_colourspace,
            apply_cctf_encoding=False, illuminant=illum_xy,
        ).T.astype(dtype)                                                   # (3,3)
        # Jacobian kernel W[w,c,k] = dye[w,c]·cmf[w,k] -> (Nw,9): lets dXYZ be
        # one matmul (L @ W) instead of an (N,Nw,3) tensor.
        self._W = np.ascontiguousarray(
            (dye[:, :, None] * cmfs[:, None, :]).reshape(dye.shape[0], 9), dtype=dtype)

        self.x_lo = np.zeros(3, dtype)
        self.x_hi = np.asarray(cmy_max, dtype)
        self.gain = dtype(2.0 ** exposure_ev)
        self.calibration = np.ascontiguousarray(
            np.eye(3) if calibration is None else np.asarray(calibration), dtype=dtype)  # (3,3)
        self._refresh_seed()

    # -- analytic log-linear seed ---------------------------------------------
    def _refresh_seed(self):
        self._cmy0 = 0.5 * (self.x_lo + self.x_hi)
        rgb0, J0 = self._forward_jac(self._cmy0[None, :])
        self._logrgb0 = np.log10(np.clip(rgb0[0], 1e-12, None))
        G = -(J0[0] / (np.clip(rgb0[0], 1e-12, None)[:, None] * LN10))      # (3,3)
        self._Ginv = np.linalg.pinv(G)

    def _seed(self, rgb):
        logrgb = np.log10(np.clip(rgb, 1e-12, None))
        cmy = self._cmy0[None, :] + (self._logrgb0[None, :] - logrgb) @ self._Ginv.T
        return np.clip(cmy, self.x_lo, self.x_hi).astype(self.dtype)

    # -- vectorised forward + Jacobian ----------------------------------------
    def _forward_jac(self, cmy, need_jac=True):
        with np.errstate(over="ignore", invalid="ignore"):
            D = self.base_safe + cmy @ self.dyeT                            # (N,Nw)
            L = self.illum_valid * np.exp((-LN10) * D)
            L = np.nan_to_num(L, nan=0.0, posinf=1e12, neginf=0.0)
            xyz = (L @ self.cmfs) / self.norm                              # (N,3)
            rgb = xyz @ self._M.T                                          # (N,3)
            if not need_jac:
                return rgb, None
            dxyz_ck = (L @ self._W).reshape(-1, 3, 3)                       # (N,3,3) [n,c,k]
            dxyz = (-LN10 / self.norm) * np.transpose(dxyz_ck, (0, 2, 1))   # [n,k,c]
            J = np.einsum("ik,nkc->nic", self._M, dxyz)                     # (N,3,3)
        return rgb, J

    # -- the solve ------------------------------------------------------------
    def _invert(self, y, max_iter=8, tol=1e-6, lam=1e-7):
        """Damped Gauss-Newton: target RGB y (N,3) -> cmy (N,3), in working dtype."""
        x = self._seed(y)
        eye = np.eye(3, dtype=self.dtype)[None, :, :]
        for _ in range(max_iter):
            f, J = self._forward_jac(x)
            r = f - y
            if np.nanmax(np.max(np.abs(r), axis=1)) < tol:
                break
            JtJ = np.einsum("nic,nik->nck", J, J)
            g = np.einsum("nic,ni->nc", J, r)
            dx = np.linalg.solve(JtJ + lam * eye, g[..., None])[..., 0]
            x = np.clip(x - dx, self.x_lo, self.x_hi)
        return x

    def pretransform(self, rgb):
        """Scanned RGB (..., 3) -> the gained, calibration-corrected RGB the scan
        model inverts (``rgb @ calibration · gain``).

        This is the cheap analytic half of :meth:`convert`. It is kept separate so
        it can stay *outside* a baked LUT: the device matrix and exposure are tuned
        constantly (blind-calibration / exposure EV) but only scale/rotate the
        input, so they must never invalidate a convert LUT — only the scan model's
        illuminant / base / dye do (see :meth:`invert`).
        """
        shape = rgb.shape
        y = (np.asarray(rgb, self.dtype).reshape(-1, 3) @ self.calibration) * self.gain
        return y.reshape(shape).astype(self.dtype)

    def invert(self, y, max_iter=8, tol=1e-6, lam=1e-7):
        """Gained/calibration-corrected RGB (..., 3) -> film cmy density (..., 3).

        The expensive, scan-model-dependent half of :meth:`convert`, and the bake
        target for the convert LUT (see ``SpectralLUTService.spectral_compute_convert``).
        """
        shape = y.shape
        cmy = self._invert(np.asarray(y, self.dtype).reshape(-1, 3), max_iter, tol, lam)
        return cmy.reshape(shape).astype(np.float64)

    def convert(self, rgb, max_iter=8, tol=1e-6, lam=1e-7):
        """Scanned RGB (..., 3) -> film cmy density (..., 3), exact solver (no LUT).

        Equivalent to ``invert(pretransform(rgb))``.
        """
        return self.invert(self.pretransform(rgb), max_iter, tol, lam)

    def gamut_bounds(self, grid=12, pad=0.02):
        """Per-channel ``(lo, hi)`` bounding box of the scan model's gamut.

        The forward image of the cmy cube, sampled on a ``grid``³ lattice. This is
        the range of (gained) RGB the converter can see — including the slightly
        negative channels ``XYZ -> RGB`` produces for saturated dye mixes — so a
        LUT baked over ``[lo, hi]`` never clamps an in-gamut pixel. ``pad`` adds a
        small relative margin. Depends only on the scan model (illuminant / base /
        dye / Dmax), i.e. exactly what also changes the inverse, so the LUT cache's
        validity probe and these bounds move together.
        """
        g = np.linspace(0.0, 1.0, grid)
        cube = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
        cube = (cube * self.x_hi).astype(self.dtype)
        rgb = self._forward_jac(cube, need_jac=False)[0].astype(np.float64)
        lo, hi = rgb.min(axis=0), rgb.max(axis=0)
        margin = pad * (hi - lo)
        return lo - margin, hi + margin

    # -- blind device calibration --------------------------------------------
    def fit_blind_calibration(self, rgb, n_sample=1500, w_anchor=5.0, w_ridge=0.03):
        """Fit a (3, 3) device-correction matrix from one image, no target/clicks.

        The film's dye gamut is a known region in RGB (the image of the cmy cube
        under the forward model); a real scan's colours are that region pushed
        through the unknown device matrix. We fit M so the corrected image lands
        on the gamut (zero forward residual), anchoring the clear base to the
        cmy=0 colorimetric point and ridging toward identity. Works best when the
        image FILLS the dye gamut (a vibrant frame) — see c40 s051 for the limit.

        Returns the matrix; the caller stores it (e.g. via format_calibration_matrix).
        """
        from scipy.optimize import least_squares

        pix = np.asarray(rgb, np.float64).reshape(-1, 3)
        pix = np.clip(pix[np.all(np.isfinite(pix), axis=1)], 0.0, None)
        if len(pix) > n_sample:
            pix = pix[np.linspace(0, len(pix) - 1, n_sample).astype(int)]
        gain = float(self.gain)
        lum = pix @ np.array([0.2, 0.7, 0.1])
        base_obs = pix[lum >= np.quantile(lum, 0.99)].mean(0)
        base_true = self._forward_jac(np.zeros((1, 3), self.dtype), need_jac=False)[0][0].astype(np.float64)
        eye = np.eye(3)

        def resid(x):
            M = x.reshape(3, 3)
            corr = (pix @ M) * gain
            cmy = self._invert(np.clip(corr, 0.0, None).astype(self.dtype))
            rgb_hat = self._forward_jac(cmy, need_jac=False)[0].astype(np.float64)
            return np.concatenate([
                (corr - rgb_hat).ravel(),
                w_anchor * ((base_obs @ M) * gain - base_true),
                w_ridge * (M - eye).ravel(),
            ])

        sol = least_squares(resid, eye.ravel(), method="lm", max_nfev=300)
        return sol.x.reshape(3, 3)
