import colour
import numpy as np

# Constants
LOG_EXPOSURE = np.linspace(-3, 4, 256)

_OBSERVER = "CIE 1931 2 Degree Standard Observer"
_CONE_FUNDAMENTALS = "Stockman & Sharpe 2 Degree Cone Fundamentals"


def _cmfs(shape):
    return colour.MSDS_CMFS[_OBSERVER].copy().align(shape)


def _lms(shape):
    # Cone fundamentals paired to the same 2-degree observer regime used by the
    # study-side XYZ geometry and Hanatos upsampling pipeline.
    return colour.colorimetry.MSDS_CMFS_LMS[_CONE_FUNDAMENTALS].copy().align(shape)


# Working spectral grid — default 380-780 nm (visible). For NIR set_spectral_shape(380, 920, 5)
# BEFORE importing modules that read these at import (arctic2026, the model
# illuminants, ...). The 1931 CMFs are ~0 past ~780, so extending the grid leaves visible colour unchanged
# (cut a shipped LUT back to 780 for a visible-only deliverable).
SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)
STANDARD_OBSERVER_CMFS = _cmfs(SPECTRAL_SHAPE)  # default color matching functions
STANDARD_OBSERVER_LMS = _lms(SPECTRAL_SHAPE)


def set_spectral_shape(
    start: float = 380, end: float = 780, step: float = 5
) -> colour.SpectralShape:
    """Reconfigure the working spectral grid: rebuild SPECTRAL_SHAPE + the derived CMFs / cone fundamentals, and
    re-point the model illuminant grid. CALL BEFORE importing modules that bake these at import time (arctic2026,
    etc.). Returns the new SpectralShape. The 1931 CMFs vanish past ~780 nm ⇒ extending to e.g. 920 is
    backward-compatible for visible colour."""
    global SPECTRAL_SHAPE, STANDARD_OBSERVER_CMFS, STANDARD_OBSERVER_LMS
    SPECTRAL_SHAPE = colour.SpectralShape(start, end, step)
    STANDARD_OBSERVER_CMFS = _cmfs(SPECTRAL_SHAPE)
    STANDARD_OBSERVER_LMS = _lms(SPECTRAL_SHAPE)
    from spektrafilm.model import (
        illuminants as _il,
    )  # binds SPECTRAL_SHAPE at its import — re-point it here

    _il.SPECTRAL_SHAPE = SPECTRAL_SHAPE
    return SPECTRAL_SHAPE
