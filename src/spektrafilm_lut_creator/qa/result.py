"""``Result`` — the unit of QA output.

Every QA test returns one ``Result``. The suite assembles them into a
markdown report. Keeping this dataclass small and uniform is what lets
the QA module stay minimal in code while admitting any kind of test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Result:
    """One QA test outcome.

    Attributes
    ----------
    name
        Short identifier (matches the test function name).
    summary
        Headline numbers a reader looks at first. Keys are descriptive
        (``"max_delta_itp"``, ``"p99"``, ``"violations"``). Floats are
        rendered to 4 significant figures in the report; ints are
        rendered as ints.
    scalar_field
        Optional per-sample scalar (one number per cube cell or per
        stimulus point), for downstream composition. Tests that already
        produce a useful viz from this field don't need to expose it,
        but exposing it lets a future test reuse the data.
    figure_path
        Path to the test's saved PNG (relative to the QA out_dir's
        report.md location), or ``None`` if the test has no figure.
    interpretation
        One sentence: what does it mean if this test fails? This is
        what makes the report legible to someone who didn't write it.
    reference_values
        Per-metric quality targets, keyed by the same metric name used
        in ``summary``. Rendered as a bulleted list under "Reference
        values" right after the figure in the report so readers can
        compare the headline numbers to a known-good range. The value
        is free-form text — typically a condition (``"≤ 2.0"``) plus a
        short justification (``"perceptual visibility threshold"``).
        Leave empty for tests that don't have established targets.
        Citations and prior art live in the test function's docstring,
        not on the ``Result`` — the bundle's QA report is for the
        bundle's numbers, not a literature review.
    passed
        ``True`` / ``False`` if a tolerance was checked, ``None`` if
        the test is purely informational. Used by the console log and
        any CI integrations; not surfaced in the bundle's QA report
        (the report instead exposes ``reference_values`` so readers
        can judge the numbers themselves).
    units
        Units string for ``summary`` values where relevant
        (``"deltaITP"``, ``"log10(cond)"``, ``"ΔE2000"``).
    """

    name: str
    summary: dict[str, float] = field(default_factory=dict)
    scalar_field: np.ndarray | None = None
    figure_path: Path | None = None
    interpretation: str = ""
    reference_values: dict[str, str] = field(default_factory=dict)
    passed: bool | None = None
    units: str = ""

    def short_summary(self) -> str:
        """One-line numeric digest for tables and logs."""
        if not self.summary:
            return "(no numeric summary)"
        parts = []
        for key, val in self.summary.items():
            if isinstance(val, (int, np.integer)):
                parts.append(f"{key}={int(val)}")
            elif isinstance(val, float):
                parts.append(f"{key}={val:.4g}")
            else:
                parts.append(f"{key}={val}")
        return "  ".join(parts)

    def status(self) -> str:
        """``"PASS"`` / ``"FAIL"`` / ``"INFO"`` — what the row badge shows."""
        if self.passed is None:
            return "INFO"
        return "PASS" if self.passed else "FAIL"
