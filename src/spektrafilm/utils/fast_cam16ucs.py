"""Numba kernel for CAM16-UCS output gamut compression.

A per-pixel port of the colour-science path used by
:func:`spektrafilm.utils.gamut_compression.compress_rgb_cam16ucs_chroma`
(``RGB_to_XYZ`` -> ``XYZ_to_CAM16UCS`` -> chroma knee -> ``CAM16UCS_to_XYZ``
-> ``XYZ_to_RGB``), fused into one parallel loop. The numpy version spends
most of its time allocating temporaries and in ``np.power`` / ``einsum``,
which are several times slower on Windows builds of numpy.

Only the per-pixel math is ported. Everything that depends on the viewing
conditions alone (degree of adaptation, F_L, N_bb, A_w, ...) and the RGB/XYZ
matrices is still computed by colour-science in :func:`cam16ucs_constants`,
so the kernel follows colour's conventions exactly, including ``spow``
(sign-preserving power) and ``sdiv`` in "Ignore Zero Conversion" mode
(non-finite quotients become 0).
"""

from __future__ import annotations

import math
from functools import lru_cache

import colour
import numpy as np
from colour.appearance import CAM_KWARGS_CIECAM02_sRGB
from colour.appearance.cam16 import MATRIX_16, MATRIX_INVERSE_16
from colour.appearance.ciecam02 import (
    achromatic_response_forward,
    degree_of_adaptation,
    post_adaptation_non_linear_response_compression_forward,
    viewing_conditions_dependent_parameters,
)
from colour.models.cam02_ucs import COEFFICIENTS_UCS_LUO2006
from numba import njit, prange

# Layout of the ``constants`` vector passed to the kernel.
_D_RGB = slice(0, 3)
_I_F_L, _I_N_BB, _I_N_CB, _I_N, _I_Z, _I_A_W, _I_C, _I_N_C, _I_C1, _I_C2 = range(3, 13)
_N_CONSTANTS = 13


@lru_cache(maxsize=None)
def cam16ucs_constants(xyz_w: tuple[float, float, float], L_A: float, Y_b: float) -> np.ndarray:
    """Viewing-condition constants for CAM16-UCS, computed by colour-science.

    ``xyz_w`` is the adapting white at Y=1, as passed to
    ``colour.XYZ_to_CAM16UCS``; it is scaled to the 0-100 domain the same
    way colour does in its default "reference" domain-range scale.
    """
    surround = CAM_KWARGS_CIECAM02_sRGB['surround']
    XYZ_w = np.asarray(xyz_w, dtype=float) * 100
    Y_w = XYZ_w[1]
    RGB_w = MATRIX_16 @ XYZ_w
    D = float(np.clip(degree_of_adaptation(surround.F, L_A), 0, 1))
    n, F_L, N_bb, N_cb, z = (float(v) for v in viewing_conditions_dependent_parameters(Y_b, Y_w, L_A))
    D_RGB = D * Y_w / RGB_w + 1 - D
    RGB_aw = post_adaptation_non_linear_response_compression_forward(D_RGB * RGB_w, F_L)
    A_w = float(achromatic_response_forward(RGB_aw, N_bb))
    _K_L, c_1, c_2 = COEFFICIENTS_UCS_LUO2006['CAM02-UCS'].values

    constants = np.empty(_N_CONSTANTS)
    constants[_D_RGB] = D_RGB
    constants[_I_F_L] = F_L
    constants[_I_N_BB] = N_bb
    constants[_I_N_CB] = N_cb
    constants[_I_N] = n
    constants[_I_Z] = z
    constants[_I_A_W] = A_w
    constants[_I_C] = surround.c
    constants[_I_N_C] = surround.N_c
    constants[_I_C1] = c_1
    constants[_I_C2] = c_2
    constants.setflags(write=False)
    return constants


@lru_cache(maxsize=None)
def rgb_xyz_matrices(output_color_space: str) -> tuple[np.ndarray, np.ndarray]:
    """Linear RGB<->XYZ matrices exactly as colour applies them (illuminant =
    the space's own whitepoint, no CCTF), read off by transforming the basis."""
    cs = colour.RGB_COLOURSPACES[output_color_space]
    white = np.asarray(cs.whitepoint, dtype=float)
    basis = np.eye(3)
    rgb_to_xyz = np.asarray(colour.RGB_to_XYZ(
        basis, colourspace=output_color_space, illuminant=white, apply_cctf_decoding=False,
    )).T
    xyz_to_rgb = np.asarray(colour.XYZ_to_RGB(
        basis, colourspace=output_color_space, illuminant=white, apply_cctf_encoding=False,
    )).T
    rgb_to_xyz = np.ascontiguousarray(rgb_to_xyz)
    xyz_to_rgb = np.ascontiguousarray(xyz_to_rgb)
    rgb_to_xyz.setflags(write=False)
    xyz_to_rgb.setflags(write=False)
    return rgb_to_xyz, xyz_to_rgb


@njit(inline='always', cache=True)
def _spow(a, p):
    # colour.algebra.spow: sign(a) * |a| ** p
    if a > 0.0:
        return a ** p
    if a < 0.0:
        return -((-a) ** p)
    if a == 0.0:
        return 0.0
    return a  # NaN


@njit(inline='always', cache=True, error_model='numpy')
def _sdiv(a, b):
    # colour.algebra.sdiv in "Ignore Zero Conversion" mode.
    q = a / b
    if math.isfinite(q):
        return q
    return 0.0


@njit(inline='always', cache=True)
def _reinhard_knee(d, threshold, limit, power):
    if d > threshold:
        scale = limit - threshold
        x = (d - threshold) / scale
        y = x / (1.0 + x ** power) ** (1.0 / power)
        return threshold + scale * y
    return d


@njit(inline='always', cache=True)
def _c_max_lookup(L, h, L_grid, h_grid, table):
    n_L = L_grid.shape[0]
    n_h = h_grid.shape[0]
    if L < L_grid[0]:  # np.clip, keeping NaN
        L = L_grid[0]
    elif L > L_grid[n_L - 1]:
        L = L_grid[n_L - 1]
    h_step = h_grid[1] - h_grid[0]
    h_idx = (h - h_grid[0]) / h_step
    h_floor = math.floor(h_idx)
    h_lo = int(h_floor) % n_h
    h_hi = (h_lo + 1) % n_h
    h_frac = h_idx - h_floor

    L_idx = (L - L_grid[0]) / (L_grid[n_L - 1] - L_grid[0]) * (n_L - 1)
    L_lo = min(max(int(math.floor(L_idx)), 0), n_L - 2)
    L_hi = L_lo + 1
    L_frac = L_idx - L_lo

    return (
        table[L_lo, h_lo] * (1 - L_frac) * (1 - h_frac)
        + table[L_lo, h_hi] * (1 - L_frac) * h_frac
        + table[L_hi, h_lo] * L_frac * (1 - h_frac)
        + table[L_hi, h_hi] * L_frac * h_frac
    )


@njit(parallel=True, cache=True, error_model='numpy')
def _compress_cam16ucs_kernel(
    rgb, out, rgb_to_xyz, xyz_to_rgb, m16, m16_inv, k,
    L_grid, h_grid, table,
    threshold, limit, power,
    use_lightness, l_threshold, l_limit, l_power, Jp_white,
):
    D_R, D_G, D_B = k[0], k[1], k[2]
    F_L, N_bb, N_cb, n_vc, z, A_w = k[3], k[4], k[5], k[6], k[7], k[8]
    c_sur, N_c, c_1, c_2 = k[9], k[10], k[11], k[12]
    F_L_4 = _spow(F_L, 0.25)
    n_term = _spow(1.64 - 0.29 ** n_vc, 0.73)
    deg = 180.0 / math.pi
    rad = math.pi / 180.0

    for i in prange(rgb.shape[0]):
        # RGB -> XYZ (0-100 domain).
        X = 100.0 * (rgb_to_xyz[0, 0] * rgb[i, 0] + rgb_to_xyz[0, 1] * rgb[i, 1] + rgb_to_xyz[0, 2] * rgb[i, 2])
        Y = 100.0 * (rgb_to_xyz[1, 0] * rgb[i, 0] + rgb_to_xyz[1, 1] * rgb[i, 1] + rgb_to_xyz[1, 2] * rgb[i, 2])
        Z = 100.0 * (rgb_to_xyz[2, 0] * rgb[i, 0] + rgb_to_xyz[2, 1] * rgb[i, 1] + rgb_to_xyz[2, 2] * rgb[i, 2])

        # XYZ_to_CAM16: sharpened RGB, adaptation, compression.
        Ra = 0.0
        Ga = 0.0
        Ba = 0.0
        for c in range(3):
            v = (m16[c, 0] * X + m16[c, 1] * Y + m16[c, 2] * Z) * (D_R if c == 0 else (D_G if c == 1 else D_B))
            f = _spow(F_L * abs(v) / 100.0, 0.42)
            s = 1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0)
            va = 400.0 * s * f / (27.13 + f) + 0.1
            if c == 0:
                Ra = va
            elif c == 1:
                Ga = va
            else:
                Ba = va
        a = Ra - 12.0 * Ga / 11.0 + Ba / 11.0
        b = (Ra + Ga - 2.0 * Ba) / 9.0
        h = (math.atan2(b, a) * deg) % 360.0
        e_t = 0.25 * (math.cos(2.0 + h * math.pi / 180.0) + 3.8)
        A = (2.0 * Ra + Ga + Ba / 20.0 - 0.305) * N_bb
        J = 100.0 * _spow(_sdiv(A, A_w), c_sur * z)
        t = 50000.0 / 13.0 * N_c * N_cb * _sdiv(e_t * _spow(a * a + b * b, 0.5), Ra + Ga + 21.0 * Ba / 20.0)
        C = _spow(t, 0.9) * _spow(J / 100.0, 0.5) * n_term
        M = C * F_L_4

        # JMh -> CAM16-UCS, then polar (Cp, hp) as the numpy version computes it.
        Jp = (1.0 + 100.0 * c_1) * J / (1.0 + c_1 * J)
        Mp = 1.0 / c_2 * math.log1p(c_2 * M)
        hr = h * rad
        ap = Mp * math.cos(hr)
        bp = Mp * math.sin(hr)
        if use_lightness:
            Jp = _reinhard_knee(Jp / Jp_white, l_threshold, l_limit, l_power) * Jp_white
        Cp = math.hypot(ap, bp)
        hp = math.atan2(bp, ap)

        # Chroma knee against C_max(Jp, hp).
        Cp_max = _c_max_lookup(Jp, hp, L_grid, h_grid, table)
        if not Cp_max >= 1e-9:  # np.fmax: NaN -> 1e-9
            Cp_max = 1e-9
        Cp_new = _reinhard_knee(Cp / Cp_max, threshold, limit, power) * Cp_max
        ap = Cp_new * math.cos(hp)
        bp = Cp_new * math.sin(hp)

        # CAM16-UCS -> JMh.
        J = -Jp / (c_1 * Jp - 1.0 - 100.0 * c_1)
        Mp = math.hypot(ap, bp)
        h = (math.atan2(bp, ap) * deg) % 360.0
        M = math.expm1(Mp / (1.0 / c_2)) / c_2

        # CAM16_to_XYZ.
        C = M / F_L_4
        J_prime = 2.220446049250313e-16 if J < 2.220446049250313e-16 else J  # np.maximum keeps NaN
        t = _spow(C / (math.sqrt(J_prime / 100.0) * n_term), 1.0 / 0.9)
        e_t = 0.25 * (math.cos(2.0 + h * math.pi / 180.0) + 3.8)
        A = A_w * _spow(J / 100.0, 1.0 / (c_sur * z))
        P_1 = _sdiv(50000.0 / 13.0 * N_c * N_cb * e_t, t)
        P_2 = A / N_bb + 0.305
        P_3 = 21.0 / 20.0
        hr = h * rad
        sin_hr = math.sin(hr)
        cos_hr = math.cos(hr)
        n_ab = P_2 * (2.0 + P_3) * (460.0 / 1403.0)
        # Two separate tests as in colour: with a NaN hue neither holds and
        # a = b = 0 (an achromatic result), which an if/else would miss.
        a = 0.0
        b = 0.0
        if abs(sin_hr) >= abs(cos_hr):
            b = n_ab / (_sdiv(P_1, sin_hr) + (2.0 + P_3) * (220.0 / 1403.0) * _sdiv(cos_hr, sin_hr)
                        - 27.0 / 1403.0 + P_3 * (6300.0 / 1403.0))
            a = b * _sdiv(cos_hr, sin_hr)
        elif abs(sin_hr) < abs(cos_hr):
            a = n_ab / (_sdiv(P_1, cos_hr) + (2.0 + P_3) * (220.0 / 1403.0)
                        - (27.0 / 1403.0 - P_3 * (6300.0 / 1403.0)) * _sdiv(sin_hr, cos_hr))
            b = a * _sdiv(sin_hr, cos_hr)
        if t == 0.0:
            a *= 0.0
            b *= 0.0
        Ra = (460.0 * P_2 + 451.0 * a + 288.0 * b) / 1403.0
        Ga = (460.0 * P_2 - 891.0 * a - 261.0 * b) / 1403.0
        Ba = (460.0 * P_2 - 220.0 * a - 6300.0 * b) / 1403.0

        R = 0.0
        G = 0.0
        B = 0.0
        for c in range(3):
            va = (Ra if c == 0 else (Ga if c == 1 else Ba)) - 0.1
            s = 1.0 if va > 0.0 else (-1.0 if va < 0.0 else 0.0)
            v = s * 100.0 / F_L * _spow(27.13 * abs(va) / (400.0 - abs(va)), 1.0 / 0.42)
            v /= (D_R if c == 0 else (D_G if c == 1 else D_B))
            if c == 0:
                R = v
            elif c == 1:
                G = v
            else:
                B = v
        X = (m16_inv[0, 0] * R + m16_inv[0, 1] * G + m16_inv[0, 2] * B) / 100.0
        Y = (m16_inv[1, 0] * R + m16_inv[1, 1] * G + m16_inv[1, 2] * B) / 100.0
        Z = (m16_inv[2, 0] * R + m16_inv[2, 1] * G + m16_inv[2, 2] * B) / 100.0

        for c in range(3):
            out[i, c] = xyz_to_rgb[c, 0] * X + xyz_to_rgb[c, 1] * Y + xyz_to_rgb[c, 2] * Z


def compress_rgb_cam16ucs(
    rgb: np.ndarray,
    *,
    output_color_space: str,
    xyz_w: np.ndarray,
    L_A: float,
    Y_b: float,
    c_max_table: tuple[np.ndarray, np.ndarray, np.ndarray],
    threshold: float,
    limit: float,
    power: float,
    lightness_compression: tuple[float, float, float] | None,
    Jp_white: float,
) -> np.ndarray:
    """Run the fused kernel on an ``(..., 3)`` RGB array; returns a new array."""
    rgb = np.asarray(rgb, dtype=float)
    shape = rgb.shape
    flat = np.ascontiguousarray(rgb.reshape(-1, 3))
    out = np.empty_like(flat)
    rgb_to_xyz, xyz_to_rgb = rgb_xyz_matrices(output_color_space)
    constants = cam16ucs_constants(tuple(float(v) for v in xyz_w), float(L_A), float(Y_b))
    L_grid, h_grid, table = (np.ascontiguousarray(a, dtype=float) for a in c_max_table)
    if lightness_compression is None:
        use_lightness, l_threshold, l_limit, l_power = False, 0.0, 1.0, 1.0
    else:
        use_lightness = True
        l_threshold, l_limit, l_power = (float(v) for v in lightness_compression)
    _compress_cam16ucs_kernel(
        flat, out, rgb_to_xyz, xyz_to_rgb, MATRIX_16, MATRIX_INVERSE_16, constants,
        L_grid, h_grid, table,
        float(threshold), float(limit), float(power),
        use_lightness, l_threshold, l_limit, l_power, float(Jp_white),
    )
    return out.reshape(shape)


def warmup_cam16ucs() -> None:
    """Trigger Numba compilation of the CAM16-UCS kernel (tiny dummy table)."""
    rgb = np.full((2, 3), 0.5)
    grid = np.linspace(0.0, 1.0, 2)
    compress_rgb_cam16ucs(
        rgb,
        output_color_space='sRGB',
        xyz_w=np.array([0.95047, 1.0, 1.08883]),
        L_A=64.0,
        Y_b=20.0,
        c_max_table=(grid * 100.0, grid * np.pi, np.ones((2, 2))),
        threshold=0.0,
        limit=1.0,
        power=6.0,
        lightness_compression=None,
        Jp_white=1.0,
    )
