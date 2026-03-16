import numpy as np
import pytest
from spectral_film_lab.profiles.io import load_profile


@pytest.fixture
def small_rgb_image():
    """A small 16x16 synthetic linear RGB image with a gray gradient."""
    ramp = np.linspace(0.01, 1.0, 16)
    image = np.ones((16, 16, 3), dtype=np.float64)
    image *= ramp[None, :, None]
    return image


@pytest.fixture
def default_params():
    """Default photo_params with expensive effects disabled for fast tests."""
    from spectral_film_lab.runtime.process import photo_params
    params = photo_params()
    # Disable stochastic/spatial effects for determinism and speed
    params.debug.deactivate_spatial_effects = True
    params.debug.deactivate_stochastic_effects = True
    params.settings.use_camera_lut = False
    params.settings.use_enlarger_lut = False
    params.settings.use_scanner_lut = False
    params.io.preview_resize_factor = 1.0
    params.camera.auto_exposure = False
    params.camera.exposure_compensation_ev = 0.0
    return params


@pytest.fixture
def portra_400_profile():
    """Load the Kodak Portra 400 profile (with auto-unmixing and couplers)."""
    return load_profile('kodak_portra_400_auc')

