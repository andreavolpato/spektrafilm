"""``debug.lut_mode`` must turn the pipeline into a pure per-pixel transform.

A 3D LUT can only represent a function of a single pixel's value. Any effect
that reads neighbouring pixels (halation, lens blur, diffusion filters, grain
blur, scanner unsharp, DIR-coupler inhibitor diffusion), any stochastic effect
(grain, glare), or any *image-global* effect (highlight boost normalised by the
frame max, auto-exposure metering) would bake incorrectly — so ``lut_mode`` must
deactivate them all.

These tests are a regression guard: they switch **on** every dangerous effect
before digesting, then assert the digested params have them off *and* that the
running pipeline is deterministic, neighbour-independent, and image-global
independent. If anyone removes a gate in ``digest_params``, one of these fails.
"""
from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.runtime.params_builder import digest_params, init_params
from spektrafilm.runtime.process import simulate

pytestmark = pytest.mark.integration


def _lut_mode_params(film_profile="kodak_portra_400", print_profile="kodak_portra_endura"):
    """Digested params with ``lut_mode`` on and every spatial / stochastic /
    image-global effect deliberately switched on beforehand, so the gating in
    ``digest_params`` is what must turn them back off."""
    p = init_params(film_profile=film_profile, print_profile=print_profile)
    # Tiny film format so pixel_size_um (= film_format_mm * 1000 / max_dim) is
    # small on these little test images. µm-scaled blurs (halation, camera lens
    # blur, diffusion filters, DIR-coupler diffusion) then resolve to MANY
    # pixels of spread, so the neighbour-independence test below would actually
    # catch them if a gate were removed — at 35mm on a 10px image they'd be
    # sub-pixel and slip through. Harmless when the gates work (every sigma is
    # zeroed, so pixel_size_um is never used).
    p.camera.film_format_mm = 0.05
    # Spatial (neighbour-reading) effects.
    p.film_render.halation.active = True
    p.camera.lens_blur_um = 50.0
    p.camera.diffusion_filter.active = True
    p.enlarger.diffusion_filter.active = True
    p.scanner.lens_blur = 2.0
    p.scanner.unsharp_mask = (2.0, 1.0)
    p.film_render.dir_couplers.diffusion_size_um = 30.0
    # Stochastic effects.
    p.film_render.grain.active = True
    p.print_render.glare.active = True
    # Image-global effects.
    p.film_render.halation.boost_ev = 8.0   # normalised by np.max(image)
    p.camera.auto_exposure = True            # meters the whole frame
    p.enlarger.print_exposure_compensation = True
    # Keep geometry identity so output shape == input and pixels are not
    # resampled (resampling mixes neighbours and would confound the test).
    p.io.upscale_factor = 1.0
    p.io.crop = False
    p.settings.use_enlarger_lut = False
    p.settings.use_scanner_lut = False
    p.debug.lut_mode = True
    return digest_params(p)


def _structured_image() -> np.ndarray:
    """A small high-frequency linear-RGB image with bright spikes — maximally
    sensitive to blur (neighbour mixing) and to highlight boost (bright pixels)."""
    rng = np.random.default_rng(12345)
    img = (rng.random((6, 10, 3)) ** 2 * 8.0 + 1e-3).astype(np.float64)
    img[1, 2] = (9.0, 7.0, 5.0)
    img[4, 7] = (6.0, 8.0, 9.0)
    return img


def test_lut_mode_digest_disables_every_dangerous_effect():
    """The param-level guard: digesting with lut_mode must zero/deactivate
    every spatial, stochastic, and image-global knob even when they were on."""
    p = _lut_mode_params()
    # promotion flags
    assert p.debug.deactivate_spatial_effects is True
    assert p.debug.deactivate_stochastic_effects is True
    # spatial
    assert p.film_render.halation.active is False
    assert tuple(p.film_render.halation.scatter_core_um) == (0.0, 0.0, 0.0)
    assert tuple(p.film_render.halation.scatter_tail_um) == (0.0, 0.0, 0.0)
    assert tuple(p.film_render.halation.halation_first_sigma_um) == (0.0, 0.0, 0.0)
    assert p.camera.lens_blur_um == 0.0
    assert p.camera.diffusion_filter.active is False
    assert p.enlarger.diffusion_filter.active is False
    assert p.enlarger.lens_blur == 0.0
    assert p.scanner.lens_blur == 0.0
    assert tuple(p.scanner.unsharp_mask) == (0.0, 0.0)
    assert p.film_render.dir_couplers.diffusion_size_um == 0
    assert p.film_render.grain.blur == 0.0
    assert p.film_render.grain.blur_dye_clouds_um == 0.0
    assert p.film_render.grain.mult_usm_amount == 0.0
    assert p.print_render.glare.blur == 0
    # stochastic
    assert p.film_render.grain.active is False
    assert p.print_render.glare.active is False
    # image-global
    assert p.film_render.halation.boost_ev == 0.0
    assert p.camera.auto_exposure is False
    assert p.enlarger.print_exposure_compensation is False
    assert p.scanner.white_correction is False
    assert p.scanner.black_correction is False


def test_lut_mode_pipeline_is_deterministic():
    """No stochastic effect leaks: two runs of the same image are identical."""
    params = _lut_mode_params()
    img = _structured_image()
    out1 = simulate(img.copy(), params, digest_params_first=False)
    out2 = simulate(img.copy(), params, digest_params_first=False)
    np.testing.assert_array_equal(out1, out2)


def test_lut_mode_pipeline_is_pixel_neighbor_independent():
    """No spatial effect leaks: each pixel's output depends only on its own
    input value, not its neighbours. Shuffle the pixels, run, unshuffle, and
    the result must match the unshuffled run pixel-for-pixel."""
    params = _lut_mode_params()
    img = _structured_image()
    out = simulate(img.copy(), params, digest_params_first=False).reshape(-1, 3)

    rng = np.random.default_rng(7)
    flat = img.reshape(-1, 3)
    perm = rng.permutation(flat.shape[0])
    shuffled = flat[perm].reshape(img.shape)
    out_shuffled = simulate(shuffled, params, digest_params_first=False).reshape(-1, 3)
    restored = np.empty_like(out_shuffled)
    restored[perm] = out_shuffled

    np.testing.assert_allclose(restored, out, atol=1e-6)


def test_lut_mode_pipeline_is_image_global_independent():
    """No image-global effect leaks (highlight boost, auto-exposure, etc.):
    a pixel's output is unaffected by the brightness of the rest of the frame.
    Raising one pixel to a huge value must not change any other pixel."""
    params = _lut_mode_params()
    img = _structured_image()
    out = simulate(img.copy(), params, digest_params_first=False)

    img_hot = img.copy()
    img_hot[0, 0] = (500.0, 500.0, 500.0)  # blows up np.max(image)
    out_hot = simulate(img_hot, params, digest_params_first=False)

    mask = np.ones(img.shape[:2], dtype=bool)
    mask[0, 0] = False  # exclude the pixel we changed
    np.testing.assert_allclose(out_hot[mask], out[mask], atol=1e-6)
