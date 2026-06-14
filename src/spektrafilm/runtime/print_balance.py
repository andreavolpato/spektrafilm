"""Print-balance helpers: solve enlarger filter shifts so midgray prints neutral.

The neutral Y/M filters come from a static database keyed to the film's *native*
base, so once the base is tuned (e.g. by the convert-film 'Detect base' action) the
print no longer balances. ``solve_neutral_filter_shifts`` re-derives the
``m_filter_shift`` / ``y_filter_shift`` (leaving the C filter fixed) so a native
midgray negative — developed with the *current* film + base — prints neutral again.
"""
from __future__ import annotations

import copy

import numpy as np

from spektrafilm.runtime.params_builder import digest_params
from spektrafilm.runtime.pipeline import SimulationPipeline


def solve_neutral_filter_shifts(params, midgray: float = 0.184, max_nfev: int = 40) -> tuple[float, float]:
    """Return ``(m_filter_shift, y_filter_shift)`` that print a midgray neutral.

    Films a neutral ``midgray`` patch with the current film + base, prints and scans
    it, and solves the two enlarger filter shifts (C fixed) so the resulting RGB is
    neutral (R = G = B). Image-independent: it calibrates the printer for the current
    film/base/enlarger/print combination, not for any loaded scan.
    """
    from scipy.optimize import least_squares
    from spektrafilm.utils.gamut_compression import OutputGamutCompressSpec

    p = copy.deepcopy(params)
    p.workflow.route = "input > film > print > scan"
    p.settings.use_enlarger_lut = False          # recompute the enlarger each trial
    p.settings.use_scanner_lut = False
    p.io.output_cctf_encoding = False             # linear output -> clean neutrality metric
    p.io.output_gamut_compress = OutputGamutCompressSpec(algorithm="off", lightness_compression=None)
    pipeline = SimulationPipeline(digest_params(p))
    gray = np.full((2, 2, 3), float(midgray))

    def residual(x):
        pipeline.enlarger.m_filter_shift = float(x[0])
        pipeline.enlarger.y_filter_shift = float(x[1])
        rgb = pipeline.process(gray, collect="rgb_out").reshape(-1, 3).mean(0)
        g = max(float(rgb[1]), 1e-6)
        return [(rgb[0] - rgb[1]) / g, (rgb[2] - rgb[1]) / g]   # R/G and B/G -> 1

    x0 = [float(params.enlarger.m_filter_shift), float(params.enlarger.y_filter_shift)]
    sol = least_squares(residual, x0, method="lm", max_nfev=max_nfev)
    return float(sol.x[0]), float(sol.x[1])
