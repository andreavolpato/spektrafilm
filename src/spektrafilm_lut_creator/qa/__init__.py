"""QA suite for spektrafilm LUT bundles.

Two questions this package answers:

1. **LUT fidelity** — does the cube preserve the spektrafilm pipeline
   within industry-grade tolerance? Off-grid deltaITP, monotonicity,
   Jacobian conditioning, total variation, gamut self-intersection.
2. **Model diagnostic** — does the spektrafilm model itself produce
   sensible output, or has something drifted in the physics?
   Characteristic curve, Planckian white-balance sweep, highlight
   rolloff smoothness, black toe, hue twist, spectral locus envelope.

The QA module imports from spektrafilm (linear-in / linear-out core)
and from spektrafilm_lut_creator (registry, formats, bundles); it is
never imported back into either. See studies/a40_lut_system/n080.

Entry point::

    from spektrafilm_lut_creator.qa import run
    run(spec, bundle, out_dir="qa/portra400_vlog_srgb")

Writes ``report.md``, ``figures/*.png``, and ``cache/*.npz`` under
``out_dir``. The cache holds pipeline ground-truth samples and is the
single expensive build artifact; everything else is cheap.
"""

from __future__ import annotations

from spektrafilm_lut_creator.qa.result import Result
from spektrafilm_lut_creator.qa.suite import (
    DEFAULT_SUITE,
    QAContext,
    list_tests,
    run,
)

__all__ = (
    "DEFAULT_SUITE",
    "QAContext",
    "Result",
    "list_tests",
    "run",
)
