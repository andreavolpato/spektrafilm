import numpy as np
import struct
import colour
import scipy
import functools
import importlib.resources
from opt_einsum import contract
import scipy.interpolate
import scipy.special

try:
    import tomllib                       # Python >= 3.11
except ModuleNotFoundError:              # pragma: no cover
    import tomli as tomllib              # Python < 3.11 fallback

from spektrafilm.profiles.io import Hanatos2025SensitivityAdaptation
from spektrafilm.utils.fast_interp_lut import apply_lut_cubic_2d
from spektrafilm.config import SPECTRAL_SHAPE, STANDARD_OBSERVER_CMFS
from spektrafilm.model.illuminants import standard_illuminant

_SQRT2PI = float(np.sqrt(2.0 * np.pi))
_HANATOS2025_MAX_CORRECTION_STOPS = 2.0  # matches max_correction_stops used during the fit

################################################################################
# LUT generatation of irradiance spectra for any xy chromaticity
# Thanks to hanatos for providing luts and sample code to develop this. I am grateful.

def _load_coeffs_lut(filename='hanatos_irradiance_xy_coeffs_250304.lut'):
    # load lut of coefficients for efficient computations of irradiance spectra
    # formatting
    header_fmt = '=4i'
    header_len = struct.calcsize(header_fmt)
    pixel_fmt = '=4f'
    pixel_len = struct.calcsize(pixel_fmt)

    package = importlib.resources.files('spektrafilm.data.luts.spectral_upsampling')
    resource = package / filename
    with resource.open("rb") as file:
        header = file.read(header_len)
        _, _, width, height = struct.Struct(header_fmt).unpack_from(header)
        px = [[0] * height for _ in range(width)]
        for j in range(height):
            for i in range(width):
                data = file.read(pixel_len)
                px[i][j] = struct.Struct(pixel_fmt).unpack_from(data)
    return np.array(px)

def _tri2quad(tc):
    # converts triangular coordinates into square coordinates.
    # for better sampling of the visible locus of xy chromaticities.
    # the lut is represented in triangular coordinates
    tc = np.array(tc)
    tx = tc[...,0]
    ty = tc[...,1]
    y = ty / np.fmax(1.0 - tx, 1e-10)
    x = (1.0 - tx)*(1.0 - tx)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    return np.stack((x,y), axis=-1)

def _quad2tri(xy):
    # converts square coordinates into triangular coordinates
    x = xy[...,0]
    y = xy[...,1]
    tx = 1 - np.sqrt(x)
    ty = y * np.sqrt(x)
    return np.stack((tx,ty), axis=-1)

def _fetch_coeffs(tc, lut_coeffs):
    # find the coefficients for spectral upsampling of given rgb coordinates
    # if color_space!='ITU-R BT.2020' or apply_cctf_decoding:
    #     rgb = colour.RGB_to_RGB(rgb, input_colourspace=color_space, apply_cctf_decoding=apply_cctf_decoding,
    #                                     output_colourspace='ITU-R BT.2020', apply_cctf_encoding=False)
    #     rgb = np.clip(rgb,0,1)
    # xyz = colour.RGB_to_XYZ(rgb, colourspace='ITU-R BT.2020', apply_cctf_decoding=False)
    # b = np.sum(xyz, axis=-1)
    # xy = xyz[...,0:2] / b[...,None]
    # tc = _tri2quad(xy)
    coeffs = np.zeros(np.concatenate((tc.shape[:-1],[4])))
    x = np.linspace(0,1,lut_coeffs.shape[0])
    for i in np.arange(4):
        coeffs[...,i] = scipy.interpolate.RegularGridInterpolator((x,x), lut_coeffs[:,:,i], method='cubic')(tc)
    return coeffs

def _compute_spectra_from_coeffs(coeffs, smooth_steps=1):
    wl = SPECTRAL_SHAPE.wavelengths
    wl_up = np.linspace(360,800,441) # upsampled wl for finer initial calculation 0.5 nm
    x = (coeffs[...,0,None] * wl_up + coeffs[...,1,None])*  wl_up  + coeffs[...,2,None]
    y = 1.0 / np.sqrt(x * x + 1.0)
    spectra = 0.5 * x * y +  0.5
    spectra /= coeffs[...,3][...,None]
    
    # gaussian smooth with smooth_step*sigmas and downsample
    step = np.mean(np.diff(wl))
    spectra = scipy.ndimage.gaussian_filter(spectra, step*smooth_steps, axes=-1)
    def interp_slice(a, wl, wl_up):
        return np.interp(wl, wl_up, a)
    spectra = np.apply_along_axis(interp_slice, axis=-1, wl=wl, wl_up=wl_up, arr=spectra)
    return spectra

def compute_hanatos2025_lut_spectra(lut_size=128, smooth_steps=1, lut_coeffs_filename='hanatos_irradiance_xy_coeffs_250304.lut'):
    v = np.linspace(0,1,lut_size)
    tx,ty = np.meshgrid(v,v, indexing='ij')
    tc = np.stack((tx,ty), axis=-1)
    lut_coeffs = _load_coeffs_lut(lut_coeffs_filename)
    coeffs = _fetch_coeffs(tc, lut_coeffs)
    lut_spectra = _compute_spectra_from_coeffs(coeffs, smooth_steps=smooth_steps)
    lut_spectra = np.array(lut_spectra, dtype=np.half)
    return lut_spectra

def _load_hanatos2025_spectra_lut(filename='hanatos2025_irradiance_xy_tc.npy'):
    data_path = importlib.resources.files('spektrafilm.data.luts.spectral_upsampling').joinpath(filename)
    with data_path.open('rb') as file:
        spectra_lut = np.double(np.load(file))
    return spectra_lut

def _illuminant_to_xy(illuminant_label):
    illu = standard_illuminant(illuminant_label)
    xyz = np.zeros((3))
    for i in np.arange(3):
        xyz[i] = np.sum(illu * STANDARD_OBSERVER_CMFS[:][:,i])
    xy = xyz[0:2] / np.sum(xyz)
    return xy

def _rgb_to_tc_b(rgb, color_space='ITU-R BT.2020', apply_cctf_decoding=False, reference_illuminant='D55'):
    # source_cs = colour.RGB_COLOURSPACES[color_space]
    # target_cs = source_cs.copy()
    # target_cs.whitepoint = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    # adapted_rgb = colour.RGB_to_RGB(rgb, input_colourspace=source_cs,
    #                                 output_colourspace=target_cs,
    #                                 adaptation_transform='Bradford')    
    # illu_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][reference_illuminant]
    illu_xy = _illuminant_to_xy(reference_illuminant)
    # CAT16 (Li et al. 2017) replaces the cone-primary instabilities
    # of CAT02 around blue and violet; it is the adaptation matrix
    # CIECAM16 and the spektrafilm output-side CAM16-UCS algorithm use
    # internally, so the input chromaticity projection stays in the
    # same adaptation family as the output gamut compression.
    xyz = colour.RGB_to_XYZ(rgb, colourspace=color_space,
                            apply_cctf_decoding=apply_cctf_decoding,
                            illuminant=illu_xy,
                            chromatic_adaptation_transform='CAT16')
    b = np.sum(xyz, axis=-1)
    xy = xyz[...,0:2] / np.fmax(b[...,None], 1e-10)
    # Previously: xy = np.clip(xy, 0, 1). Removed because the input gamut
    # compression baked into the tc_lut at build time (see
    # gamut_compression.remap_tc_lut_for_compression) now handles OOG
    # chromaticities correctly via the LUT itself. _tri2quad below already
    # clamps each output component to [0, 1] for the (rare) inputs that
    # also fall outside the lower xy triangle, so the LUT lookup
    # coordinates remain in-bounds.
    tc = _tri2quad(xy)
    b = np.nan_to_num(b)
    return tc, b

################################################################################
# hanatos2025 sensitivity adaptation


def _radial_mobius_warp_xy(
    xy: np.ndarray,
    center: tuple[float, float],
    alpha: float,
) -> np.ndarray:
    cx, cy = center
    dx = xy[..., 0] - cx
    dy = xy[..., 1] - cy
    d = np.sqrt(dx * dx + dy * dy)
    scale = 1.0 / (1.0 + alpha * d)
    wxy = np.zeros_like(xy)
    wxy[..., 0] = cx + dx * scale
    wxy[..., 1] = cy + dy * scale
    return wxy
    
def hanika_sigmoid(z: np.ndarray, max_val: float) -> np.ndarray:
    """Algebraic sigmoid matching Jakob & Hanika 2019, bounded to [-max_val, max_val]."""
    return z / np.sqrt(1.0 + (z / max_val) ** 2)


def poly2d_deg4(tc: np.ndarray,
                params: np.ndarray,
                center_tc: tuple[float, float] = (0.0, 0.0),) -> np.ndarray:
    x = tc[..., 0] - center_tc[0]
    y = tc[..., 1] - center_tc[1]
    x2 = x * x
    y2 = y * y
    xy_term = x * y
    x3 = x2 * x
    y3 = y2 * y
    _, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14 = params
    # note that we want center_tc to be zero correction -> not adding c0
    return (
        c1 * x + c2 * y + c3 * x2 + c4 * y2 + c5 * xy_term
        + c6 * x3 + c7 * y3 + c8 * (x2 * y) + c9 * (x * y2)
        + c10 * (x2 * x2) + c11 * (y2 * y2) + c12 * (x3 * y) + c13 * (x2 * y2) + c14 * (x * y3)
    )

def locked_logistic_rising(
    x: np.ndarray, mu: float, sigma: float, nu: float,
) -> np.ndarray:
    """Rising flex-and-sigma-locked logistic.

    Locked through (μ, ½) with maximum slope `S = 1/(σ·√(2π))`. ν > 0
    bends the tails (ν=1 ≈ erf in shape; ν<1 fattens the lower tail,
    ν>1 thins it). Inputs may broadcast freely.
    """
    nu = np.maximum(np.asarray(nu, dtype=float), 1e-5)
    S = 1.0 / (sigma * _SQRT2PI)
    k = (2.0 * nu * S) / (1.0 - 2.0 ** (-nu))
    exponent = np.clip(-k * (x - mu), -500.0, 500.0)
    nu_log2 = nu * np.log(2.0)
    log_q = np.where(
        nu_log2 > 50.0,
        nu_log2 + np.log1p(-np.exp(-nu_log2)),
        np.log(np.expm1(nu_log2)),
    )
    log_denom = np.logaddexp(0.0, log_q + exponent)
    return np.exp(-(1.0 / nu) * log_denom)


def eval_poly4_warp_log_exposure_surface(params, illuminant_xy) -> np.ndarray:
    """Evaluate a degree-4 polynomial surface after warping in xy, then mapping back to tc."""
    # Warp xy coordinate with the mobius warp, which is a radial compression towards the illuminant chromaticity.
    # This is to better sample the spectra of chromaticities close to the illuminant
    # and stabilize the surface for large chromaticity shifts.
    surface_size = HANATOS2025_SPECTRA_LUT.shape[0]
    tc_base = np.linspace(0,1,surface_size)
    tc = np.stack(np.meshgrid(tc_base, tc_base, indexing='ij'), axis=-1)
    xy = _quad2tri(tc)
    tc_center = _tri2quad(illuminant_xy)
    surface_log_exposure_correction = np.zeros((surface_size, surface_size,3))
    for i in range(3):
        poly_params = params[i, :-1]
        alpha = float(params[i, -1])
        xy_warped = _radial_mobius_warp_xy(xy, illuminant_xy, alpha=alpha)
        tc_warped = _tri2quad(xy_warped)
        raw = poly2d_deg4(tc_warped, poly_params, center_tc=tc_center)
        surface_log_exposure_correction[...,i] = hanika_sigmoid(raw, _HANATOS2025_MAX_CORRECTION_STOPS)
    return surface_log_exposure_correction

def eval_poly4_log_exposure_surface(params, illuminant_xy) -> np.ndarray:
    surface_size = HANATOS2025_SPECTRA_LUT.shape[0]
    tc_base = np.linspace(0,1,surface_size)
    tc = np.stack(np.meshgrid(tc_base, tc_base, indexing='ij'), axis=-1)
    tc_center = _tri2quad(illuminant_xy)
    surface_log_exposure_correction = np.zeros((surface_size, surface_size,3))
    for i in range(3):
        poly_params = params[i]
        raw = poly2d_deg4(tc, poly_params, center_tc=tc_center)
        surface_log_exposure_correction[...,i] = hanika_sigmoid(raw, _HANATOS2025_MAX_CORRECTION_STOPS)
    return surface_log_exposure_correction

def eval_logiflex8_spectral_bandpass(params: np.ndarray) -> np.ndarray:
    """8-param logiflex window: erf6 skeleton + (nu_uv, nu_ir) shared across channels.

    params: (c_uv_base, sigma_uv, c_ir_base, sigma_ir, c_uv_b, c_ir_r, nu_uv, nu_ir).
    """
    wavelengths = SPECTRAL_SHAPE.wavelengths
    (c_uv_base, sigma_uv, c_ir_base, sigma_ir,
     c_uv_b, c_ir_r, nu_uv, nu_ir) = params
    cuv = np.array([c_uv_base, c_uv_base, c_uv_b], dtype=float)
    cir = np.array([c_ir_r, c_ir_base, c_ir_base], dtype=float)
    w = wavelengths[:, None]
    edge_uv = locked_logistic_rising(w, cuv[None, :], sigma_uv, nu_uv)
    edge_ir = 1.0 - locked_logistic_rising(w, cir[None, :], sigma_ir, nu_ir)
    window = edge_uv * edge_ir
    return window

def eval_erf4_spectral_bandpass(params: np.ndarray) -> np.ndarray:
    """4-param erf window: common bandpass replicated across all channels.

    params: (c_uv, sigma_uv, c_ir, sigma_ir).
    """
    _SQRT2 = float(np.sqrt(2.0))
    wavelengths = SPECTRAL_SHAPE.wavelengths
    c_uv, sigma_uv, c_ir, sigma_ir = (float(v) for v in params)
    edge_uv = 0.5 * (1.0 + scipy.special.erf((wavelengths - c_uv) / (sigma_uv * _SQRT2)))
    edge_ir = 0.5 * (1.0 - scipy.special.erf((wavelengths - c_ir) / (sigma_ir * _SQRT2)))
    common = edge_uv * edge_ir
    return np.repeat(common[:, None], 3, axis=1)

def eval_spectral_bandpass_window(params: np.ndarray, model: str = 'erf4') -> np.ndarray:
    if model == 'logiflex8':
        return eval_logiflex8_spectral_bandpass(params)
    if model == 'erf4':
        return eval_erf4_spectral_bandpass(params)
    raise ValueError(f"Unknown spectral bandpass model: {model}")
    
def eval_log_exposure_correction_surface(params, illuminant_xy, model='poly4') -> np.ndarray:
    if model == 'poly4_warp_xy':
        return eval_poly4_warp_log_exposure_surface(params, illuminant_xy)
    if model == 'poly4':
        return eval_poly4_log_exposure_surface(params, illuminant_xy)
    raise ValueError(f"Unknown log exposure correction surface model: {model}")

################################################################################
# From [Mallett2019]

MALLETT2019_BASIS = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019.copy().align(SPECTRAL_SHAPE)
def rgb_to_raw_mallett2019(RGB, sensitivity,
                           color_space='sRGB', apply_cctf_decoding=True,
                           reference_illuminant='D65'):
    """
    Converts an RGB color to a raw sensor response using the method described in Mallett et al. (2019).

    Parameters
    ----------
    RGB : array_like
        RGB color values.
    illuminant : array_like
        Illuminant spectral distribution.
    sensitivity : array_like
        Camera sensor spectral sensitivities.
    color_space : str, optional
        The color space of the input RGB values. Default is 'sRGB'.
    apply_cctf_decoding : bool, optional
        Whether to apply the color component transfer function (CCTF) decoding. Default is True.

    Returns
    -------
    raw : ndarray
        Raw sensor response.
    """
    illuminant = standard_illuminant(reference_illuminant)[:]
    basis_set_with_illuminant = np.array(MALLETT2019_BASIS[:])*np.array(illuminant)[:, None]
    lrgb = colour.RGB_to_RGB(RGB, color_space, 'sRGB',
                    apply_cctf_decoding=apply_cctf_decoding,
                    apply_cctf_encoding=False)
    raw  = contract('ijk,lk,lm->ijm', lrgb, basis_set_with_illuminant, sensitivity)
    raw = np.nan_to_num(raw)
    raw = np.ascontiguousarray(raw)
    
    raw_midgray  = np.einsum('k,km->m', illuminant*0.184, sensitivity) # use 0.184 as midgray reference
    return raw / raw_midgray # per-channel -> a neutral surface maps to gray (1,1,1)


_MALLETT2019_MIDGRAY = 0.184
_MALLETT2019_SCENE_ILLUMINANT = 'D65'     # sRGB's whitepoint; the basis reproduces sRGB under it


def compute_mallett2019_reflectance_lut(lut_size=192, scene_illuminant=_MALLETT2019_SCENE_ILLUMINANT):
    """(S, S, 81) reflectance per tc chromaticity at unit brightness, from the Mallett &
    Yuksel 2019 sRGB basis — the SAME deliverable as jakob2019 / otsu2018, shipped as
    `mallett2019_reflectance_xy_tc.npy` so mallett flows through the generic reflectance
    path (relight + integrate) like every other method.

    Each tc -> xy -> XYZ_unit = [x, y, 1-x-y]; XYZ (under the scene illuminant = sRGB
    white) -> linear sRGB -> reflectance = sum_k lrgb_k * basis_k(l). Stored float16.

    NOTE: mallett is LINEAR in RGB, so the reconstruction is exactly scale-separable — the
    tc LUT + linear-b reproduces rgb_to_raw_mallett2019 EXACTLY in-gamut (no tc loss, unlike
    jakob/otsu whose nonlinear recovery costs ~5%). The only difference from the per-pixel
    path is out-of-sRGB-gamut handling: this LUT clips reflectance to [0, 1] (like jakob /
    otsu), where the per-pixel path leaves it unclipped. Weakest at saturated OOG colours.
    """
    v = np.linspace(0, 1, lut_size)
    tc = np.stack(np.meshgrid(v, v, indexing='ij'), axis=-1)        # (S,S,2)
    xy = _quad2tri(tc)
    x, y = xy[..., 0], xy[..., 1]
    XYZ_unit = np.stack([x, y, np.clip(1.0 - x - y, 0.0, None)], axis=-1)
    illu_xy = _illuminant_to_xy(scene_illuminant)
    lrgb = colour.XYZ_to_RGB(XYZ_unit, colour.RGB_COLOURSPACES['sRGB'],
                             illuminant=illu_xy, apply_cctf_encoding=False)    # linear sRGB
    reflectance = np.einsum('...k,lk->...l', lrgb, np.array(MALLETT2019_BASIS[:]))
    return np.array(np.clip(reflectance, 0.0, 1.0), dtype=np.half)

################################################################################
# Using hanatos irradiance spectra generation

HANATOS2025_SPECTRA_LUT = _load_hanatos2025_spectra_lut()
HANATOS2025_NO_ADAPTATION = Hanatos2025SensitivityAdaptation(apply_surface=False, apply_window=False)

def compute_hanatos2025_tc_lut(sensitivity, hanatos2025_adaptation, gamut_compress=None):
    """Compute the LUT of spectra for given sensitivities and hanatos2025 adaptation parameters.
    Both window and surface preserve white balance.

    Parameters
    ----------
    sensitivity, hanatos2025_adaptation :
        As before.
    gamut_compress :
        Optional ``spektrafilm.utils.gamut_compression.InputGamutCompressSpec``.
        When provided and ``gamut_compress.active``, the returned
        LUT has the input gamut compression baked into it via remap-resample
        (see n100 §3.1): a runtime lookup ``new_lut[xy]`` returns the same
        raw value the uncompressed LUT would have returned for
        ``compress(xy)``. The per-pixel hot path therefore stays
        compression-agnostic.
    """

    spectra_lut = HANATOS2025_SPECTRA_LUT

    if hanatos2025_adaptation.spectral_gaussian_blur > 0:
        spectra_lut = scipy.ndimage.gaussian_filter(spectra_lut,
                        (0, 0, hanatos2025_adaptation.spectral_gaussian_blur))

    if hanatos2025_adaptation.apply_window and hanatos2025_adaptation.window_params is not None:
        window = eval_spectral_bandpass_window(hanatos2025_adaptation.window_params)
        # White-balance preservation must reference the neutral the LUT actually EMITS — the
        # smooth-metamer reconstruction at the reference-illuminant chromaticity — NOT the bare
        # illuminant. The two share CMF chromaticity but differ spectrally; under the film's
        # (non-CMF) sensitivities, normalizing on the illuminant drifts neutrals ~2-3% off the
        # balanced gray. Referencing the reconstructed white keeps the windowed neutral exactly
        # on the unwindowed (1,1,1) gray for every reference illuminant.
        white_tc = _tri2quad(_illuminant_to_xy(hanatos2025_adaptation.reference_illuminant))
        neutral_sd = apply_lut_cubic_2d(                               # same cubic sampler as runtime
            spectra_lut, np.asarray(white_tc).reshape(1, 1, 2))[0, 0]  # (81,) reconstructed white
        normalization = (
            np.sum(sensitivity * neutral_sd[:, None] * window, axis=0)
            / np.sum(sensitivity * neutral_sd[:, None], axis=0)
        )
        window = window / normalization
        raw_lut = contract('ijl,lm->ijm', spectra_lut, sensitivity * window)
    else:
        raw_lut = contract('ijl,lm->ijm', spectra_lut, sensitivity)

    if hanatos2025_adaptation.apply_surface and hanatos2025_adaptation.surface_params is not None:
        xy_illu = _illuminant_to_xy(hanatos2025_adaptation.reference_illuminant)
        surface = eval_log_exposure_correction_surface(hanatos2025_adaptation.surface_params,
                                                       illuminant_xy=xy_illu)
        raw_lut *= 2**surface

    if gamut_compress is not None and gamut_compress.active:
        # Bake compression into the LUT at build time. The reference
        # illuminant matches what _rgb_to_tc_b uses at runtime, so the
        # compression's achromatic axis is the film's white.
        from spektrafilm.utils.gamut_compression import remap_tc_lut_for_compression
        ref_xy = _illuminant_to_xy(hanatos2025_adaptation.reference_illuminant)
        raw_lut = remap_tc_lut_for_compression(raw_lut, ref_xy, gamut_compress)

    return raw_lut

def rgb_to_raw_hanatos2025(rgb, sensitivity,
                           color_space,
                           apply_cctf_decoding,
                           reference_illuminant,
                           tc_lut=None
                           ):
    tc_raw, b = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=reference_illuminant,
    )
    if tc_lut is None: # fallback to on-the-fly computation if tc_lut not provided
        tc_lut = compute_hanatos2025_tc_lut(sensitivity, HANATOS2025_NO_ADAPTATION)
    raw = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw *= b[...,None] # scale the raw back with the scale factor
    # note that sensitivities are already normalized in balancing such that raw_midgray is 1, so no need to normalize here
    return raw

def rgb_to_smooth_spectrum(rgb, color_space, apply_cctf_decoding, reference_illuminant):
    # direct interpolation of the spectra lut, to be used only for smooth spectra close to white
    tc_w, b_w = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=reference_illuminant,
    )
    v = np.linspace(0, 1, HANATOS2025_SPECTRA_LUT.shape[0])
    spectrum_w = scipy.interpolate.RegularGridInterpolator((v,v), HANATOS2025_SPECTRA_LUT)(tc_w)
    spectrum_w *= b_w
    return spectrum_w.flatten()


################################################################################
# Generic reflectance-LUT registry — identifier-driven loading + usage
#
# jakob2019, otsu2018 (and any future method) are the SAME runtime computation: a
# (S, S, bands) reflectance LUT per tc chromaticity, recovered under a SCENE illuminant,
# RELIT by the film reference illuminant, integrated against the sensitivity, and
# self-normalized on a midgray neutral. So instead of a function per method, each LUT
# ships a `.toml` sidecar (co-located with its `.npy`) declaring identifier / kind / file
# and — for reflectance — scene_illuminant / midgray / bounded. The functions here load
# and consume ANY such LUT by identifier; the named jakob2019 / otsu2018 entry points
# below are thin wrappers. Drop a new `<id>_*.npy` + `<id>_*.toml` in the data folder and
# it is usable with no code change. (hanatos2025 is irradiance — its own path, not here.)

_LUT_DATA_PACKAGE = 'spektrafilm.data.luts.spectral_upsampling'


@functools.lru_cache(maxsize=1)
def _lut_registry():
    """{identifier: descriptor} parsed from every `*.toml` sidecar in the LUT data
    folder. Cached once per process (shipped data does not change at runtime)."""
    registry = {}
    for entry in importlib.resources.files(_LUT_DATA_PACKAGE).iterdir():
        if entry.name.endswith('.toml'):
            with entry.open('rb') as file:
                descriptor = tomllib.load(file)
            registry[descriptor['identifier']] = descriptor
    return registry


def available_lut_identifiers(kind=None):
    """Sorted identifiers of shipped LUTs, optionally filtered to a `kind`
    ('reflectance' | 'irradiance')."""
    registry = _lut_registry()
    return sorted(i for i, d in registry.items() if kind is None or d.get('kind') == kind)


def lut_descriptor(identifier):
    """Descriptor dict (parsed `.toml` sidecar) for `identifier`. Raises KeyError if
    no LUT with that identifier is shipped."""
    registry = _lut_registry()
    if identifier not in registry:
        raise KeyError(f"no spectral-upsampling LUT {identifier!r}; "
                       f"available: {sorted(registry)}")
    return registry[identifier]


_LUT_SPECTRA_CACHE = {}


def get_lut_spectra(identifier):
    """Lazy-load + cache the (S, S, bands) spectra array for `identifier` from the `.npy`
    its descriptor points at. NOT clipped to [0, 1] — methods that lift (R > 1) load
    as-is. Lazy (not an import-time global) so a fresh checkout missing the `.npy` does
    not fail at import; the array loads on first use."""
    if identifier not in _LUT_SPECTRA_CACHE:
        descriptor = lut_descriptor(identifier)
        path = importlib.resources.files(_LUT_DATA_PACKAGE).joinpath(descriptor['file'])
        with path.open('rb') as file:
            _LUT_SPECTRA_CACHE[identifier] = np.double(np.load(file))
    return _LUT_SPECTRA_CACHE[identifier]


def compute_reflectance_tc_lut(identifier, sensitivity, reference_illuminant,
                               spectra_lut=None, gamut_compress=None):
    """(S, S, 3) raw LUT for any REFLECTANCE-family method, selected by `identifier`.

    Relight the reflectance by the film reference illuminant, integrate against the film
    sensitivity, self-normalize PER CHANNEL on the descriptor's midgray neutral (so a neutral
    surface -> gray (1,1,1) raw, matching the white-balance convention the hanatos2025 window
    preserves), and optionally bake input gamut compression (build-time remap, keeping the
    per-pixel hot path compression-agnostic). The decoupled-illuminant structure (the Hanika
    point, study b50). Generalizes compute_jakob2019_tc_lut / compute_otsu2018_tc_lut.
    """
    descriptor = lut_descriptor(identifier)
    if descriptor.get('kind') != 'reflectance':
        raise ValueError(f"{identifier!r} is kind={descriptor.get('kind')!r}, not "
                         f"'reflectance' (cannot relight a non-reflectance LUT)")
    if spectra_lut is None:
        spectra_lut = get_lut_spectra(identifier)
    midgray = float(descriptor['reflectance']['midgray'])
    E_ref = standard_illuminant(reference_illuminant)[:]               # (81,)
    relit = spectra_lut * E_ref[None, None, :]                        # (S,S,81)
    raw_lut = contract('ijl,lm->ijm', relit, sensitivity)             # (S,S,3)
    # White balance: a neutral input must map to gray (1,1,1), matching the convention the
    # hanatos2025 windowed path now sits on. GREEN sets the midgray exposure anchor (flat 0.184
    # under E_ref -> green 1, method-independent so exposure is consistent across methods). R/B
    # are balanced against the METHOD'S OWN emitted neutral — its recovered reflectance at the
    # SCENE-illuminant chromaticity, i.e. exactly where a neutral lands (rgb_to_raw_reflectance
    # projects under the scene white, so the neutral sits at the D65 cell, NOT the D55 relight
    # white) — so methods whose recovered neutral is NOT perfectly flat (otsu) still resolve to
    # gray, not just the flat-neutral ones. (Was green-only, which left the film's reference-
    # illuminant cast in every neutral.)
    scene_illuminant = descriptor['reflectance']['scene_illuminant']
    raw_midgray_green = np.einsum('l,lm->m', midgray * E_ref, sensitivity)[1]
    white_tc = _tri2quad(_illuminant_to_xy(scene_illuminant))
    # sample with the SAME cubic interpolator the runtime uses (apply_lut_cubic_2d), so the
    # normalization neutral matches the sampled neutral exactly even for LUTs with sharp
    # near-white structure (arctic), not just smooth ones.
    neutral_refl = apply_lut_cubic_2d(spectra_lut, np.asarray(white_tc).reshape(1, 1, 2))[0, 0]
    raw_neutral = np.einsum('l,lm->m', neutral_refl * E_ref, sensitivity)   # its film response (3,)
    chroma = raw_neutral / raw_neutral[1]                             # neutral chroma, green = 1
    raw_lut = raw_lut / (chroma * raw_midgray_green)                  # gray neutral, green-anchored

    if gamut_compress is not None and gamut_compress.active:
        from spektrafilm.utils.gamut_compression import remap_tc_lut_for_compression
        ref_xy = _illuminant_to_xy(reference_illuminant)
        raw_lut = remap_tc_lut_for_compression(raw_lut, ref_xy, gamut_compress)

    return raw_lut


def rgb_to_raw_reflectance(identifier, rgb, sensitivity, color_space,
                           apply_cctf_decoding, reference_illuminant,
                           tc_lut=None, scene_illuminant=None):
    """Runtime path for any REFLECTANCE-family method, selected by `identifier` — the
    hanatos2025 hot path with decoupled illuminants. Chromaticity is projected under the
    descriptor's SCENE illuminant (override with `scene_illuminant`); the relit `tc_lut`
    is sampled and scaled by brightness `b`. Generalizes rgb_to_raw_jakob2019 / _otsu2018.
    """
    descriptor = lut_descriptor(identifier)
    if scene_illuminant is None:
        scene_illuminant = descriptor['reflectance']['scene_illuminant']
    tc_raw, b = _rgb_to_tc_b(
        rgb,
        color_space=color_space,
        apply_cctf_decoding=apply_cctf_decoding,
        reference_illuminant=scene_illuminant,
    )
    if tc_lut is None:  # fallback to on-the-fly computation if tc_lut not provided
        tc_lut = compute_reflectance_tc_lut(identifier, sensitivity, reference_illuminant)
    raw = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw *= b[..., None]
    return raw


def compute_tc_lut(method, sensitivity, reference_illuminant=None,
                   spectra_lut=None, gamut_compress=None, hanatos2025_adaptation=None):
    """Build the (S, S, 3) raw tc_lut for any shipped `method`, dispatched on its
    descriptor `kind`. REFLECTANCE methods relight + integrate (need reference_illuminant);
    the IRRADIANCE method integrates directly with optional sensitivity adaptation. The
    single generic builder behind the named compute_*_tc_lut wrappers / the LUT service."""
    kind = lut_descriptor(method).get('kind')
    if kind == 'reflectance':
        return compute_reflectance_tc_lut(method, sensitivity, reference_illuminant,
                                          spectra_lut=spectra_lut, gamut_compress=gamut_compress)
    if kind == 'irradiance':
        adaptation = (hanatos2025_adaptation if hanatos2025_adaptation is not None
                      else HANATOS2025_NO_ADAPTATION)
        return compute_hanatos2025_tc_lut(sensitivity, adaptation, gamut_compress=gamut_compress)
    raise ValueError(f"method {method!r} has unsupported LUT kind {kind!r}")


def rgb_to_raw(method, rgb, sensitivity, color_space, apply_cctf_decoding,
               reference_illuminant, tc_lut=None, scene_illuminant=None):
    """Convert RGB to film raw for any shipped `method`, dispatched on its descriptor
    `kind`. REFLECTANCE methods project under the scene illuminant and sample the relit
    tc_lut; the IRRADIANCE method (hanatos2025) samples its irradiance tc_lut directly.
    The single generic runtime entry behind the named rgb_to_raw_* wrappers."""
    kind = lut_descriptor(method).get('kind')
    if kind == 'reflectance':
        return rgb_to_raw_reflectance(method, rgb, sensitivity, color_space,
                                      apply_cctf_decoding, reference_illuminant,
                                      tc_lut=tc_lut, scene_illuminant=scene_illuminant)
    if kind == 'irradiance':
        return rgb_to_raw_hanatos2025(rgb, sensitivity, color_space=color_space,
                                      apply_cctf_decoding=apply_cctf_decoding,
                                      reference_illuminant=reference_illuminant, tc_lut=tc_lut)
    raise ValueError(f"method {method!r} has unsupported LUT kind {kind!r}")


################################################################################
# From [Jakob2019] — bounded surface reflectance recovery (study b50)
#
# Unlike hanatos2025 (which recovers the *irradiance* spectrum reaching the
# sensor), jakob2019 recovers a bounded surface *reflectance* R(l) in [0, 1]
# under an assumed SCENE illuminant (D65), then RELIGHTS it with the film's
# reference illuminant. It shares hanatos's exact runtime hot path
# (_rgb_to_tc_b -> apply_lut_cubic_2d(tc) * b); only the spectrum stored per tc
# chromaticity differs (bounded reflectance vs irradiance).
#
# The reflectance is the same algebraic sigmoid hanatos evaluates:
#   R(l) = 0.5 + 0.5 * u / sqrt(1 + u^2),   u = quadratic in wavelength.
# The film-independent reflectance LUT is baked ONCE (at SCENE illuminant, unit
# brightness X+Y+Z=1) and shipped as `jakob2019_reflectance_xy_tc.npy`, the
# analog of hanatos's `hanatos2025_irradiance_xy_tc.npy`. Brightness rides the
# linear `b` scale at runtime. This 2D-tc + linear-b approximation costs ~5% on
# saturated/bright colors (reflectance is not scale-separable); see study b50
# n010 for the full analysis.

_JAKOB2019_MIDGRAY = 0.184
_JAKOB2019_SCENE_ILLUMINANT = 'D65'
# shifted-Legendre basis P0,P1,P2 on the scaled wavelength t in [0,1].
# NOT the raw monomials {l^2, l, 1}: those are near-collinear on the band, which
# makes J^T J singular and the solve fail to converge. Legendre orthogonality
# conditions the batched Levenberg-Marquardt step (study b50 n010 §4a).
_JAKOB2019_T = (
    (SPECTRAL_SHAPE.wavelengths - SPECTRAL_SHAPE.wavelengths[0])
    / (SPECTRAL_SHAPE.wavelengths[-1] - SPECTRAL_SHAPE.wavelengths[0])
)
_JAKOB2019_BASIS = np.stack(
    [np.ones_like(_JAKOB2019_T), 2 * _JAKOB2019_T - 1,
     6 * _JAKOB2019_T * _JAKOB2019_T - 6 * _JAKOB2019_T + 1],
    axis=-1,
)


def _jakob2019_scene_integration(scene_illuminant):
    """E*cmfs (81,3) and white-normalizer k for a scene illuminant (R=1 -> Y=1)."""
    E = standard_illuminant(scene_illuminant)[:]
    cmfs = np.asarray(STANDARD_OBSERVER_CMFS[:])
    Ecmfs = E[:, None] * cmfs
    k = 1.0 / np.sum(E * cmfs[:, 1])
    return Ecmfs, k


def reflectance_from_jakob2019_coeffs(coeffs):
    """(..., 3) sigmoid coefficients -> (..., 81) bounded reflectance. Vectorized."""
    u = np.einsum("...j,lj->...l", coeffs, _JAKOB2019_BASIS)
    return 0.5 + 0.5 * u / np.sqrt(1.0 + u * u)


def _jakob2019_forward(coeffs, Ecmfs, k):
    u = np.einsum("...j,lj->...l", coeffs, _JAKOB2019_BASIS)
    s = 1.0 / np.sqrt(1.0 + u * u)
    R = 0.5 + 0.5 * u * s
    XYZ = k * np.einsum("...l,lm->...m", R, Ecmfs)
    dRdu = 0.5 * s ** 3  # d sigmoid / du
    J = k * np.einsum("...lj,lm->...mj", dRdu[..., None] * _JAKOB2019_BASIS, Ecmfs)
    return XYZ, J


def solve_jakob2019_coeffs(XYZ_target, scene_illuminant=_JAKOB2019_SCENE_ILLUMINANT, iters=80):
    """Batched Levenberg-Marquardt for sigmoid coefficients over a whole array.

    Returns coeffs of shape XYZ_target.shape[:-1] + (3,). In-gamut targets
    converge to ~0 residual; out-of-gamut edges settle at the closest bounded
    reflectance (the same extrapolation role hanatos's LUT plays there).
    """
    Ecmfs, k = _jakob2019_scene_integration(scene_illuminant)
    c = np.zeros(XYZ_target.shape[:-1] + (3,))
    mu = np.full(XYZ_target.shape[:-1], 1e-2)
    eye = np.eye(3)
    XYZ, J = _jakob2019_forward(c, Ecmfs, k)
    cost = np.sum((XYZ - XYZ_target) ** 2, axis=-1)
    for _ in range(iters):
        r = XYZ - XYZ_target
        JtJ = np.einsum("...mi,...mj->...ij", J, J)
        Jtr = np.einsum("...mi,...m->...i", J, r)
        # ridge floor (mu + 1e-8) keeps A positive-definite even at degenerate /
        # out-of-gamut chromaticities; without it mu can underflow to 0 and
        # np.linalg.solve raises "Singular matrix".
        A = JtJ + (mu[..., None, None] + 1e-8) * eye
        delta = np.linalg.solve(A, -Jtr[..., None])[..., 0]
        cn = c + delta
        XYZn, Jn = _jakob2019_forward(cn, Ecmfs, k)
        costn = np.sum((XYZn - XYZ_target) ** 2, axis=-1)
        better = costn < cost
        c = np.where(better[..., None], cn, c)
        XYZ = np.where(better[..., None], XYZn, XYZ)
        J = np.where(better[..., None, None], Jn, J)
        cost = np.where(better, costn, cost)
        mu = np.clip(np.where(better, mu * 0.5, mu * 3.0), 1e-7, 1e7)
    return c


def compute_jakob2019_reflectance_lut(lut_size=192, scene_illuminant=_JAKOB2019_SCENE_ILLUMINANT):
    """(S, S, 81) bounded reflectance per tc chromaticity, baked at unit brightness.

    The film-independent artifact shipped as `jakob2019_reflectance_xy_tc.npy`
    (analog of hanatos's irradiance spectra LUT). Each tc -> xy via `_quad2tri`;
    the unit-brightness tristimulus XYZ = [x, y, 1-x-y] (X+Y+Z = 1) is solved for
    its sigmoid coefficients over the whole grid at once, then evaluated to
    reflectance. Stored as float16 to match the hanatos LUT footprint.
    """
    v = np.linspace(0, 1, lut_size)
    tc = np.stack(np.meshgrid(v, v, indexing='ij'), axis=-1)        # (S,S,2)
    xy = _quad2tri(tc)                                              # (S,S,2)
    x, y = xy[..., 0], xy[..., 1]
    XYZ_unit = np.stack([x, y, np.clip(1.0 - x - y, 0.0, None)], axis=-1)
    coeffs = solve_jakob2019_coeffs(XYZ_unit, scene_illuminant)
    reflectance = reflectance_from_jakob2019_coeffs(coeffs)
    return np.array(reflectance, dtype=np.half)


def get_jakob2019_reflectance_lut():
    """Back-compat wrapper -> get_lut_spectra('jakob2019')."""
    return get_lut_spectra('jakob2019')


def compute_jakob2019_tc_lut(sensitivity, reference_illuminant,
                             spectra_lut=None, gamut_compress=None):
    """Back-compat wrapper -> compute_reflectance_tc_lut('jakob2019', ...)."""
    return compute_reflectance_tc_lut('jakob2019', sensitivity, reference_illuminant,
                                      spectra_lut=spectra_lut, gamut_compress=gamut_compress)


def rgb_to_raw_jakob2019(rgb, sensitivity, color_space, apply_cctf_decoding,
                         reference_illuminant, tc_lut=None,
                         scene_illuminant=_JAKOB2019_SCENE_ILLUMINANT):
    """Back-compat wrapper -> rgb_to_raw('jakob2019', ...)."""
    return rgb_to_raw('jakob2019', rgb, sensitivity, color_space, apply_cctf_decoding,
                      reference_illuminant, tc_lut=tc_lut, scene_illuminant=scene_illuminant)


################################################################################
# From [Otsu2018] — clustered-basis surface reflectance recovery (study b50)
#
# Second reflectance prior alongside jakob2019, sharing the SAME deliverable
# shape and hot path. Where jakob fits a smooth 3-parameter sigmoid per
# chromaticity, Otsu et al. 2018 (colour.recovery.XYZ_to_sd_Otsu2018) selects a
# leaf cluster in a trained tree and recovers reflectance = cluster_mean +
# basis @ weights. It represents multi-modal/structured reflectances a single
# sigmoid cannot, at the cost of cluster-boundary discontinuities, and its
# linear-b approximation floor is lower than jakob's on the ColorChecker (~2.3%
# vs ~5.1%; see study b50 n010 §4b).
#
# Unlike jakob, the bake is a plain per-point loop: Otsu's per-color recovery is
# ~0.55 ms (vs jakob's ~31 ms), fast enough that no batched solver is needed.

_OTSU2018_MIDGRAY = 0.184
_OTSU2018_SCENE_ILLUMINANT = 'D65'


def compute_otsu2018_reflectance_lut(lut_size=192, scene_illuminant=_OTSU2018_SCENE_ILLUMINANT):
    """(S, S, 81) bounded reflectance per tc chromaticity, baked at unit brightness.

    The film-independent artifact shipped as `otsu2018_reflectance_xy_tc.npy`.
    Each tc -> xy via `_quad2tri`; the unit-brightness tristimulus
    XYZ = [x, y, 1-x-y] is recovered per grid point with Otsu2018 (clip=True ->
    bounded reflectance), aligned to SPECTRAL_SHAPE, stored float16.
    """
    v = np.linspace(0, 1, lut_size)
    tc = np.stack(np.meshgrid(v, v, indexing='ij'), axis=-1)        # (S,S,2)
    xy = _quad2tri(tc)
    x, y = xy[..., 0], xy[..., 1]
    XYZ_unit = np.stack([x, y, np.clip(1.0 - x - y, 0.0, None)], axis=-1).reshape(-1, 3)
    illuminant_sd = standard_illuminant(scene_illuminant, return_class=True)
    wl = SPECTRAL_SHAPE.wavelengths
    reflectance = np.empty((XYZ_unit.shape[0], len(wl)))
    for i, xyz in enumerate(XYZ_unit):
        sd = colour.recovery.XYZ_to_sd_Otsu2018(
            xyz, cmfs=STANDARD_OBSERVER_CMFS, illuminant=illuminant_sd, clip=True)
        reflectance[i] = sd.align(SPECTRAL_SHAPE).values
    reflectance = reflectance.reshape(lut_size, lut_size, len(wl))
    # Otsu's clip=True bounds the recovery, but aligning its native (380,730,10)
    # shape out to SPECTRAL_SHAPE's 780 nm tail can overshoot [0,1] at OOG grid
    # edges. Clip the aligned LUT so the shipped reflectance stays physical
    # (bounded [0,1] is the property reflectance recovery guarantees over mallett).
    reflectance = np.clip(reflectance, 0.0, 1.0)
    return np.array(reflectance, dtype=np.half)


def get_otsu2018_reflectance_lut():
    """Back-compat wrapper -> get_lut_spectra('otsu2018')."""
    return get_lut_spectra('otsu2018')


def compute_otsu2018_tc_lut(sensitivity, reference_illuminant,
                            spectra_lut=None, gamut_compress=None):
    """Back-compat wrapper -> compute_reflectance_tc_lut('otsu2018', ...)."""
    return compute_reflectance_tc_lut('otsu2018', sensitivity, reference_illuminant,
                                      spectra_lut=spectra_lut, gamut_compress=gamut_compress)


def rgb_to_raw_otsu2018(rgb, sensitivity, color_space, apply_cctf_decoding,
                        reference_illuminant, tc_lut=None,
                        scene_illuminant=_OTSU2018_SCENE_ILLUMINANT):
    """Back-compat wrapper -> rgb_to_raw('otsu2018', ...)."""
    return rgb_to_raw('otsu2018', rgb, sensitivity, color_space, apply_cctf_decoding,
                      reference_illuminant, tc_lut=tc_lut, scene_illuminant=scene_illuminant)


if __name__=='__main__':
    lut_coeffs = _load_coeffs_lut()
    coeffs = _fetch_coeffs(np.array([[1,1]]) ,lut_coeffs)
    spectra = _compute_spectra_from_coeffs(coeffs)
    lut_spectra = compute_hanatos2025_lut_spectra(lut_size=128)
