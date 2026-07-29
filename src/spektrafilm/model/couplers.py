import numpy as np
from opt_einsum import contract

from spektrafilm.model.density_curves import interpolate_exposure_to_density
from spektrafilm.runtime.params_schema import DirCouplersParams
from spektrafilm.utils.fast_gaussian_filter import (
    fast_exponential_filter,
    fast_gaussian_filter,
)

# ---- Langmuir saturating donor (DIR inhibitor release) ----------------------


def langmuir_donor_params(density_curves, langmuir_donor_k_rgb, n_ch):
    """Per-channel Langmuir shape K and amplitude-matching density D_ref.

    Derived entirely in-model from the (already base-subtracted) density
    curves:

    - ``d_max[c]`` = max over exposure of ``density_curves[:, c]``. Because
      ``develop()`` removes the toe/base before couplers and the curves are a
      sum of sublayer Gaussian CDFs that plateau, this max equals the summed
      sublayer d_max — the channel's full dye/silver capacity.
    - ``D_ref[c]`` = ``d_max[c] / 2`` (d_min is taken as 0; the toe is gone).
    - ``K[c]`` = ``langmuir_donor_k_rgb[c] * d_max[c]`` — the normalized shape knob,
      so the saturation half-scale is a fraction of the layer ceiling.

    ``langmuir_donor_k_rgb -> inf`` recovers the linear law.
    """
    d_max = np.nanmax(density_curves, axis=0)
    d_ref = 0.5 * d_max
    k = np.asarray(langmuir_donor_k_rgb, dtype=float)[:n_ch] * d_max
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


# ---- Receiver-side saturation (positive / reversal interimage) --------------


def receiver_langmuir_params(
    langmuir_d_ref, dir_couplers_matrix_unit, langmuir_receiver_k_rgb, n_ch
):
    """Per-receiver operating-point arrived inhibitor ``c_ref`` and knee ``Kr``.

    For reversal film the donor (first-dev silver release) is left linear and the
    saturation sits on the *receiver* (Shuto et al. 1997; study b30 / n110). The
    receiver Langmuir is amplitude-matched at the inhibitor arriving at the
    operating point, with the knee normalized by the arrived-inhibitor *ceiling*
    — exactly mirroring the donor side, where ``K = langmuir_donor_k_rgb * d_max``
    (knob x ceiling) and the amplitude-match point ``D_ref = d_max/2`` is half
    the ceiling. All derived in-model from the *unit* coupler matrix (before the
    ``amount`` scaling) so the knee is amount-independent:

    - ``c_ref[m]`` = ``sum_k D_ref[k] * M_unit[k, m]`` — the operating-point
      inhibitor arriving at receiver ``m`` (amplitude-match point). ``D_ref``
      (= ``langmuir_d_ref`` = ``d_max/2``) is the donor operating point; for the
      silver donor the midgray silver ``d_max - d_max/2 = d_max/2`` coincides
      with it.
    - ``c_max[m]`` = ``sum_k d_max[k] * M_unit[k, m]`` = ``2 * c_ref[m]`` — the
      arrived-inhibitor ceiling (all donors at ``d_max``), the receiver analogue
      of ``d_max``.
    - ``Kr[m]`` = ``langmuir_receiver_k_rgb[m] * c_max[m]`` — the normalized knee, a
      fraction of the ceiling, just like ``langmuir_donor_k_rgb`` on the donor side
      (``-> inf`` recovers linear).
    """
    c_ref = np.einsum(
        "k, km->m", np.asarray(langmuir_d_ref, dtype=float), dir_couplers_matrix_unit
    )
    c_max = 2.0 * c_ref  # D_ref = d_max/2, so Σ d_max·M_unit = 2·c_ref
    kr = np.asarray(langmuir_receiver_k_rgb, dtype=float)[:n_ch] * c_max
    return kr, c_ref


def receiver_langmuir(arrived_inhibitor, receiver_kr, receiver_c_ref):
    """Amplitude-matched receiver-side Langmuir ``S(c) = c (Kr + c_ref)/(Kr + c)``.

    Saturates the inhibitor ``c`` arriving at each receiver channel (last axis)
    — i.e. the receiving layer's development-rate response to inhibitor
    concentration, rather than the donor release. ``S(c_ref) = c_ref`` leaves the
    calibrated operating point unchanged; the law rolls off only as the arrived
    inhibitor exceeds ``c_ref`` (e.g. under a pushed ``amount``). ``Kr`` and
    ``c_ref`` broadcast over the last axis; a channel with ``Kr = inf`` stays
    linear.
    """
    kr = np.asarray(receiver_kr, dtype=float)
    finite = np.isfinite(kr)
    kr_safe = np.where(finite, kr, 1.0)
    saturated = (
        arrived_inhibitor * (kr_safe + receiver_c_ref) / (kr_safe + arrived_inhibitor)
    )
    return np.where(finite, saturated, arrived_inhibitor)


def compute_density_curves_before_dir_couplers(
    density_curves,
    log_exposure,
    dir_couplers_matrix,
    langmuir_k=None,
    langmuir_d_ref=None,
    receiver_kr=None,
    receiver_c_ref=None,
    positive=False,
):
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
        # We are asusming that interimage effects in positive film are acting in the silver development stage
        # We are also assuming that silver density is d_max - d
        density_curves_silver = np.nanmax(density_curves, axis=0) - density_curves
    else:
        density_curves_silver = np.copy(density_curves)

    if langmuir_k is None:
        donor = density_curves_silver  # linear donor (positive: stoichiometric silver release)
    else:
        donor = langmuir_donor(density_curves_silver, langmuir_k, langmuir_d_ref)
    couplers_amount_curves = contract("jk, km->jm", donor, dir_couplers_matrix)
    if receiver_kr is not None:
        couplers_amount_curves = receiver_langmuir(
            couplers_amount_curves, receiver_kr, receiver_c_ref
        )
    log_exposure_0 = log_exposure[:, None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for i in np.arange(density_curves.shape[1]):
        if positive:
            density_curves_corrected[:, i] = -np.interp(
                log_exposure, log_exposure_0[:, i], -density_curves[:, i]
            )
        else:
            density_curves_corrected[:, i] = np.interp(
                log_exposure, log_exposure_0[:, i], density_curves[:, i]
            )
    return density_curves_corrected


def compute_dir_couplers_matrix(
    couplers_params: DirCouplersParams = DirCouplersParams(), n_ch: int = 3
):
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

    M_self = (
        np.array(couplers_params.gamma_samelayer_rgb)[:n_ch]
        * couplers_params.inhibition_samelayer
    )
    M_self = np.diag(M_self)
    M_inter = np.zeros((n_ch, n_ch))
    if n_ch == 3:
        # Off-diagonal terms follow the same convention: donor row, receiver column.
        M_inter[0, 1] = couplers_params.gamma_interlayer_r_to_gb[0]
        M_inter[0, 2] = couplers_params.gamma_interlayer_r_to_gb[1]
        M_inter[1, 0] = couplers_params.gamma_interlayer_g_to_rb[0]
        M_inter[1, 2] = couplers_params.gamma_interlayer_g_to_rb[1]
        M_inter[2, 0] = couplers_params.gamma_interlayer_b_to_rg[0]
        M_inter[2, 1] = couplers_params.gamma_interlayer_b_to_rg[1]
        M_inter *= couplers_params.inhibition_interlayer
    return M_self + M_inter


def compute_exposure_correction_dir_couplers(
    log_raw,
    density_cmy,
    density_max,
    dir_couplers_matrix,
    langmuir_k=None,
    langmuir_d_ref=None,
    diffusion_size_pixel=0.0,
    diffusion_tail_size_pixel=0.0,
    diffusion_exp_tail_weight=0.0,
    receiver_kr=None,
    receiver_c_ref=None,
    positive=False,
):
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
    dir_couplers_matrix (numpy.ndarray): The inhibitors matrix. First index is the input layer, second index is the output layer.
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
    if langmuir_k is None:
        donor = density_silver  # linear donor (positive: stoichiometric silver release)
    else:
        donor = langmuir_donor(density_silver, langmuir_k, langmuir_d_ref)
    # donor[..., k] generated in donor layer k contributes to receiver m
    # through dir_couplers_matrix[k, m].
    log_raw_correction = contract("ijk, km->ijm", donor, dir_couplers_matrix)
    if diffusion_size_pixel > 0:
        log_raw_correction = (1 - diffusion_exp_tail_weight) * fast_gaussian_filter(
            log_raw_correction, diffusion_size_pixel
        ) + fast_exponential_filter(
            log_raw_correction, diffusion_tail_size_pixel
        ) * diffusion_exp_tail_weight
    # Receiver-side saturation: the receiving layer's response to the inhibitor
    # that actually arrives (i.e. after spatial diffusion). Positive/reversal only.
    if receiver_kr is not None:
        log_raw_correction = receiver_langmuir(
            log_raw_correction, receiver_kr, receiver_c_ref
        )
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

    positive = profile_type == "positive"
    n_ch = density_cmy.shape[-1]

    couplers_matrix = compute_dir_couplers_matrix(dir_couplers, n_ch=n_ch)

    # Langmuir parameters, derived in-model from the density curves (d_max per
    # channel; D_ref = d_max/2). Shared by the inversion and the forward
    # correction so both apply the same saturation.
    langmuir_k, langmuir_d_ref = langmuir_donor_params(
        density_curves, dir_couplers.langmuir_donor_k_rgb, n_ch=n_ch
    )

    if positive:
        # Reversal interimage (Shuto et al. 1997, n110): linear first-dev silver
        # donor + receiver-side saturation. c_ref is derived from the *unit*
        # matrix (before the amount scaling) so the receiver knee is
        # amount-independent and a pushed `amount` drives the response into
        # saturation rather than moving the knee.
        donor_k = None  # linear donor
        receiver_kr, receiver_c_ref = receiver_langmuir_params(
            langmuir_d_ref,
            couplers_matrix,
            dir_couplers.langmuir_receiver_k_rgb,
            n_ch=n_ch,
        )
    else:
        # Negative: donor-side Langmuir (coupler exhaustion), no receiver term.
        donor_k = langmuir_k
        receiver_kr = receiver_c_ref = None

    couplers_matrix = couplers_matrix * dir_couplers.amount

    density_curves_0 = compute_density_curves_before_dir_couplers(
        density_curves,
        log_exposure,
        couplers_matrix,
        donor_k,
        langmuir_d_ref,
        receiver_kr=receiver_kr,
        receiver_c_ref=receiver_c_ref,
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
        donor_k,
        langmuir_d_ref,
        diffusion_size_pixel,
        diffusion_tail_size_pixel,
        diffusion_tail_weight,
        receiver_kr=receiver_kr,
        receiver_c_ref=receiver_c_ref,
        positive=positive,
    )
    return interpolate_exposure_to_density(
        log_raw_0, density_curves_0, log_exposure, gamma_factor
    )


if __name__ == "__main__":
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
