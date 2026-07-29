"""QA test suite, grouped by topic for navigability.

Tests live in four sibling modules:

- :mod:`.lut_fidelity` — does the cube reproduce the pipeline (5 tests)
- :mod:`.model_diagnostic` — does the pipeline produce sensible output (5 tests)
- :mod:`.gamut_compression` — input-gamut-compression diagnostics (2 tests)
- :mod:`.picture_style` — noise, gamut edge stress, slice viz (4 tests)

Shared utilities (``_save``, ``MIDGRAY_18_OKLAB_L``, the polar input-sample
helper) live in :mod:`._helpers` and are re-exported here so external
consumers (notably the rosette-alignment regression test) can keep using
``qa.tests.<name>``.
"""

from __future__ import annotations

from spektrafilm_lut_creator.qa.tests._helpers import (
    MIDGRAY_18_OKLAB_L as MIDGRAY_18_OKLAB_L,
)
from spektrafilm_lut_creator.qa.tests._helpers import _save as _save
from spektrafilm_lut_creator.qa.tests.gamut_compression import (
    input_gamut_compression_preview,
    input_gamut_compression_smoothness,
)
from spektrafilm_lut_creator.qa.tests.lut_fidelity import (
    jacobian_condition,
    monotonicity,
    off_grid_identity,
    output_gamut_compression,
    total_variation,
)
from spektrafilm_lut_creator.qa.tests.model_diagnostic import (
    characteristic_curve,
    dynamic_range_usage,
    hue_twist_oklab,
    planckian_sweep,
    spectral_locus_envelope,
)
from spektrafilm_lut_creator.qa.tests.picture_style import (
    _polar_oklch_input_samples as _polar_oklch_input_samples,
)
from spektrafilm_lut_creator.qa.tests.picture_style import (
    noise_gradient,
    noise_sensitivity,
    output_gamut_edge_stress,
    rg_plane_slices,
)

DEFAULT_TESTS = (
    off_grid_identity,
    monotonicity,
    jacobian_condition,
    total_variation,
    output_gamut_compression,
    characteristic_curve,
    dynamic_range_usage,
    planckian_sweep,
    hue_twist_oklab,
    spectral_locus_envelope,
    input_gamut_compression_preview,
    input_gamut_compression_smoothness,
    noise_sensitivity,
    noise_gradient,
    output_gamut_edge_stress,
    rg_plane_slices,
)
