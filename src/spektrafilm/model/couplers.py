import numpy as np
from opt_einsum import contract

from spektrafilm.runtime.params_schema import DirCouplersParams
from spektrafilm.utils.fast_gaussian_filter import fast_gaussian_filter, fast_exponential_filter
from spektrafilm.model.density_curves import interpolate_exposure_to_density


# ---- Langmuir saturating donor (DIR inhibitor release) ----------------------

def langmuir_donor_params(density_curves, langmuir_k_rgb, n_ch):
    """Per-channel Langmuir shape K and amplitude-matching density D_ref.

    Derived entirely in-model from the (already base-subtracted) density
    curves:

    - ``d_max[c]`` = max over exposure of ``density_curves[:, c]``. Because
      ``develop()`` removes the toe/base before couplers and the curves are a
      sum of sublayer Gaussian CDFs that plateau, this max equals the summed
      sublayer d_max — the channel's full dye/silver capacity.
    - ``D_ref[c]`` = ``d_max[c] / 2`` (d_min is taken as 0; the toe is gone).
    - ``K[c]`` = ``langmuir_k_rgb[c] * d_max[c]`` — the normalized shape knob,
      so the saturation half-scale is a fraction of the layer ceiling.

    ``langmuir_k_rgb -> inf`` recovers the linear law.
    """
    d_max = np.nanmax(density_curves, axis=0)
    d_ref = 0.5 * d_max
    k = np.asarray(langmuir_k_rgb, dtype=float)[:n_ch] * d_max
    return k, d_ref


def langmuir_donor(density_silver, langmuir_k, langmuir_d_ref):
    """Amplitude-matched Langmuir donor ``g(D) = D (K + D_ref) / (K + D)``.

    The saturating amount of inhibitor released per donor channel (last axis).
    ``g(D_ref) = D_ref``, so the calibrated operating-point inhibition is
    unchanged versus the linear model; the laws diverge only away from D_ref,
    where the Langmuir form rolls off instead of growing without bound. ``K``
    and ``D_ref`` broadcast over the last (donor channel) axis. A channel with
    ``K = inf`` stays linear (``g(D) = D``).
    """
    k = np.asarray(langmuir_k, dtype=float)
    finite = np.isfinite(k)
    k_safe = np.where(finite, k, 1.0)
    saturated = density_silver * (k_safe + langmuir_d_ref) / (k_safe + density_silver)
    return np.where(finite, saturated, density_silver)


def compute_density_curves_before_dir_couplers(density_curves, log_exposure, dir_couplers_matrix,
                                               langmuir_k, langmuir_d_ref, positive=False):
    """
    DIR couplers affect the same layer by increasing contrast.
    I suppose that in the design of a film this is taken into account, and the final film has well behaved density curves.
    In order to get final curves for gray ramps equal to the input data, the density curves before the effect of the couplers are needed.

    Args:
        density_curves (numpy.array): Characteristic density curves of the film after the application of DIR couplers
        log_exposure (numpy.array): The image as log_exposure
        dir_couplers_matrix (_type_): DIR couplers matrix computed with compute_dir_couplers_matrix()

    Returns:
        numpy.array: Corrected density curves before the effect of DIR couplers
    """
    
    if positive:
        # We are asusming that interimage effects in positve film are acting in the silver development stage
        # We are also assuming that silver density is d_max - d
        density_curves_silver = np.nanmax(density_curves, axis=0) - density_curves
    else:
        density_curves_silver = np.copy(density_curves)
    
    donor = langmuir_donor(density_curves_silver, langmuir_k, langmuir_d_ref)
    couplers_amount_curves = contract('jk, km->jm', donor, dir_couplers_matrix)
    log_exposure_0 = log_exposure[:,None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for i in np.arange(density_curves.shape[1]):
        if positive:
            density_curves_corrected[:,i] = -np.interp(log_exposure, log_exposure_0[:,i], -density_curves[:,i])
        else:
            density_curves_corrected[:,i] = np.interp(log_exposure, log_exposure_0[:,i], density_curves[:,i])
    return density_curves_corrected


def compute_dir_couplers_matrix(couplers_params: DirCouplersParams = DirCouplersParams(), n_ch: int = 3):
    """
    Compute the inhibitors matrix using a simple diffusion model across layers.

    The matrix is ``n_ch x n_ch``. The diagonal is self-inhibition (same-layer
    adjacency / edge effect, after spatial diffusion); the off-diagonal is the
    inter-image (inter-layer) effect. Inter-image is intrinsically trichromatic
    and is only populated for ``n_ch == 3``; single-emulsion BW (``n_ch == 1``)
    keeps self-inhibition only — its off-diagonal does not exist.

    Parameters:
    amount_rgb (list of float): Amounts of dir couplers for RGB channels. Default is [0.7,0.7,0.5]. Typically 0-1 range.
    layer_diffusion (float): Sigma for gaussian diffusion distance of dir couplers. Default is 1.

    Returns:
    numpy.ndarray: The computed inhibitors matrix.
    Row index is the donor/source layer that releases inhibitor.
    Column index is the receiving/affected layer whose exposure is reduced.
    """

    M_self = np.array(couplers_params.gamma_samelayer_rgb)[:n_ch]*couplers_params.inhibition_samelayer
    M_self = np.diag(M_self)
    M_inter = np.zeros((n_ch, n_ch))
    if n_ch == 3:
        # Off-diagonal terms follow the same convention: donor row, receiver column.
        M_inter[0,1] = couplers_params.gamma_interlayer_r_to_gb[0]
        M_inter[0,2] = couplers_params.gamma_interlayer_r_to_gb[1]
        M_inter[1,0] = couplers_params.gamma_interlayer_g_to_rb[0]
        M_inter[1,2] = couplers_params.gamma_interlayer_g_to_rb[1]
        M_inter[2,0] = couplers_params.gamma_interlayer_b_to_rg[0]
        M_inter[2,1] = couplers_params.gamma_interlayer_b_to_rg[1]
        M_inter *= couplers_params.inhibition_interlayer
    return M_self + M_inter

def compute_exposure_correction_dir_couplers(log_raw, density_cmy, density_max,
                                             dir_couplers_matrix,
                                             langmuir_k, langmuir_d_ref,
                                             diffusion_size_pixel,
                                             diffusion_tail_size_pixel=0.0,
                                             diffusion_exp_tail_weight=0.0,
                                             positive=False):
    """
    Apply coupler inhibitors to the raw data based on density curves and inhibitor values.
    Coupler inhibitors are released when density is formed in the emulsion layers.
    If a layer is dense, the inhibitors are released to prevent further density formation in neighboring layers.
    Also self-inhibitors in the same layer, after spatial diffusion, can prevent further density formation
    in nearby areas, adding a local contrast effect.

    Parameters:
    raw (numpy.ndarray): The raw data to which inhibitors will be applied.
    density_cmy (numpy.ndarray): The density values for each layer.
    density_max (float): The maximum density value achievable for each layer, used for normalization.
    dir_couplers_matrix (numpy.ndarray): The inhibitors matrix. Fisrt index is the input layer, second index is the output layer.
    diffusion_size_pixel (int): The size of the gaussian filter for the diffusion of the inhibitors in xy.
    diffusion_tail_size_pixel (int): The size of the exponential tail for the diffusion of the inhibitors in xy.
    diffusion_exp_tail_weight (float): The weight of the exponential tail in the diffusion of the inhibitors.
    langmuir_k (array): per-channel Langmuir half-saturation scale K (absolute density units).
    langmuir_d_ref (array): per-channel amplitude-matching density D_ref.

    Returns:
    numpy.ndarray: The modified raw exposure data after applying the effect of inhibitors.
    """
    if positive:
        density_silver = density_max - density_cmy
    else:
        density_silver = np.copy(density_cmy)
    # Saturating Langmuir inhibitor release per donor channel (replaces the old
    # supralinear high_exposure_couplers_shift term, whose nonlinearity grew the
    # wrong way at high density).
    donor = langmuir_donor(density_silver, langmuir_k, langmuir_d_ref)
    # donor[..., k] generated in donor layer k contributes to receiver m
    # through dir_couplers_matrix[k, m].
    log_raw_correction = contract('ijk, km->ijm', donor, dir_couplers_matrix)
    if diffusion_size_pixel>0:
        log_raw_correction = ( (1-diffusion_exp_tail_weight)*fast_gaussian_filter(log_raw_correction, diffusion_size_pixel)
                                       + fast_exponential_filter(log_raw_correction, diffusion_tail_size_pixel)*diffusion_exp_tail_weight)
    log_raw_corrected = log_raw - log_raw_correction
    return log_raw_corrected


def apply_density_correction_dir_couplers(
    density_cmy,
    log_raw,
    pixel_size_um,
    log_exposure,
    density_curves,
    dir_couplers,
    profile_type,
    gamma_factor=1.0,
):
    if not dir_couplers.active:
        return density_cmy

    positive = profile_type == 'positive'

    couplers_matrix = compute_dir_couplers_matrix(dir_couplers, n_ch=density_cmy.shape[-1])
    couplers_matrix *= dir_couplers.amount

    # Langmuir saturating-donor parameters, derived in-model from the density
    # curves (d_max per channel; D_ref = d_max/2). Shared by the inversion and
    # the forward correction so both apply the same saturation.
    langmuir_k, langmuir_d_ref = langmuir_donor_params(
        density_curves, dir_couplers.langmuir_k_rgb, n_ch=density_cmy.shape[-1])

    density_curves_0 = compute_density_curves_before_dir_couplers(
        density_curves,
        log_exposure,
        couplers_matrix,
        langmuir_k,
        langmuir_d_ref,
        positive=positive,
    )
    density_max = np.nanmax(density_curves, axis=0)
    # Spatial-off short-circuit: when diffusion is disabled (typical under
    # debug.lut_mode), the inner ``compute_exposure_correction_dir_couplers``
    # gates the spatial filtering on ``diffusion_size_pixel > 0`` anyway,
    # but the unconditional division below crashes when pixel_size_um is
    # None (which happens for LUT bakes that inject past preprocess). Gate
    # the conversion to pixel space on the source value being > 0 so the
    # non-spatial DIR-coupler chemistry can still run.
    if dir_couplers.diffusion_size_um > 0:
        diffusion_size_pixel = dir_couplers.diffusion_size_um / pixel_size_um
        diffusion_tail_size_pixel = dir_couplers.diffusion_tail_um / pixel_size_um
    else:
        diffusion_size_pixel = 0
        diffusion_tail_size_pixel = 0
    diffusion_tail_weight = dir_couplers.diffusion_tail_weight
    log_raw_0 = compute_exposure_correction_dir_couplers(
        log_raw,
        density_cmy,
        density_max,
        couplers_matrix,
        langmuir_k,
        langmuir_d_ref,
        diffusion_size_pixel,
        diffusion_tail_size_pixel,
        diffusion_tail_weight,
        positive=positive,
    )
    return interpolate_exposure_to_density(log_raw_0, density_curves_0, log_exposure, gamma_factor)

if __name__=='__main__':
    # # Test the raw correction coupler inhibitors
    # log_raw = np.ones((4,4,3))
    # density_cmy = np.ones((4,4,3))
    # density_max = 2.2
    # couplers_amount = [0.9,0.7,0.5]
    # diffusion_size_pixel = 2
    # log_raw = raw_correction_dir_couplers(log_raw, density_cmy, density_max, couplers_amount, diffusion_size_pixel)
    # print(log_raw)

    couplers_params = DirCouplersParams()
    M = compute_dir_couplers_matrix(couplers_params)
    print(M)