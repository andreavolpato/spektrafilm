"""Shared helpers used across the QA test topic modules.

Re-exported through ``spektrafilm_lut_creator.qa.tests.__init__`` so the
external rosette-alignment test in ``tests/lut_creator/qa/`` and the
topic-module callers all hit the same private utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from spektrafilm_lut_creator.qa import viz

if TYPE_CHECKING:
    from spektrafilm_lut_creator.qa.suite import QAContext


MIDGRAY_18_OKLAB_L = 0.5646225971435698
"""Oklab L of 18% linear gray.

Used as the canonical mid-gray slice for the noise-sensitivity figures
and re-exported via the package ``__init__`` so external tests can lock
to the same constant.
"""


def _save(ctx: QAContext, fig, name: str):
    """Save a figure under ``figures/<name>.png`` and close it.

    Stamps every figure with the producing spektrafilm version (bottom
    center) via :func:`viz.add_footer` so reports remain traceable
    after they leave the bundle directory.
    """
    viz.add_footer(fig, ctx.bundle.meta.provenance.spektrafilm_version)
    path = ctx.figures_dir / f"{name}.png"
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# LUT fidelity.
# ---------------------------------------------------------------------------
