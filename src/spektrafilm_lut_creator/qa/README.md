# spektrafilm_lut_creator.qa

QA suite for spektrafilm LUT bundles. Answers two questions in one pass:

1. **LUT fidelity** — does the cube preserve the spektrafilm pipeline within
   industry tolerance, both on-grid and off-grid via trilinear and tetrahedral
   interpolation (what real hosts use)?
2. **Model diagnostic** — does the spektrafilm pipeline itself produce sensible
   output? Failures here aren't LUT-bake bugs, they're model issues. The bake
   might be perfect; the model still wrong.

Design context: `studies/a40_lut_system/n080_lut_quality_and_visualization.md`.

## Use

```python
from spektrafilm_lut_creator.builders import BundleBuilder
from spektrafilm_lut_creator.bundles import BundleSpec
from spektrafilm_lut_creator.qa import run

spec = BundleSpec(...)
bundle = BundleBuilder(spec).build()
results = run(spec, bundle, out_dir="qa/my_bundle", print_index=0)
```

`run` writes:

- `qa/my_bundle/report.md` — the human-readable QA report (renders in VS Code,
  GitHub, any markdown viewer)
- `qa/my_bundle/figures/*.png` — one PNG per test
- `qa/my_bundle/cache/*.npz` — pipeline reference samples (the only expensive
  build artifact; invalidated automatically when the bundle changes)

To QA every print in a multi-print bundle, iterate over
`range(len(bundle.luts))` and call `run` per index.

## The eleven tests

### LUT fidelity

| Test | What it asks | Pass criterion |
|---|---|---|
| `off_grid_identity` | Does the LUT match the pipeline at off-grid points under trilinear + tetrahedral? | `max ΔE₀₀ ≤ 2.0`, `p99 ≤ 1.0` both methods |
| `monotonicity` | Are diagonal axes non-decreasing in their matching output channel? | Zero violations on diagonal |
| `jacobian_condition` | Local 3×3 Jacobian condition number — smoothness diagnostic | Informational |
| `total_variation` | Per-axis variation + axial-FFT high-band energy | Informational |
| `output_gamut_compression` | Face fold-backs + hull-volume compression ratio + output-gamut compression preview (OkLab + xy) | No folds, ratio in `[0.05, 1.05]` |

### Model diagnostic

| Test | What it asks | Pass criterion |
|---|---|---|
| `characteristic_curve` | System D-vs-input on the neutral ramp | Informational |
| `planckian_sweep` | Daylight illuminants → output chromaticity | `max bend angle ≤ 30°` |
| `hue_twist_oklab` | Per-saturation-band hue rotation | `max ≤ 30°` |
| `spectral_locus_envelope` | Reach of model gamut at maximum saturation | Informational |

### Picture-style diagnostics

| Test | What it asks | Pass criterion |
|---|---|---|
| `output_gamut_edge_stress` | LUT rendering of white / hue-cycle / black bands at the edges of Rec.709, Rec.2020, ACES2065-1 | Informational |
| `rg_plane_slices` | R-G cube cross-sections at evenly-spaced B-input values, rendered in sRGB | Informational |

## Layout

```text
qa/
  __init__.py     # public API
  result.py       # Result dataclass
  evaluators.py   # trilinear + tetrahedral LUT application
  reference.py    # pipeline ground-truth cache
  patterns.py     # stimulus generators (neutral ramp, Planckian, ...)
  metrics.py      # ΔE₀₀, ΔITP, Jacobian, TV, hue rotation, ...
  viz.py          # all plot functions (returns Figure, tests save)
  tests.py        # the eleven test functions
  suite.py        # QAContext + run() + markdown report emission
```

Code stays minimal on purpose — each test is one function and is explicit about
its pattern, metric, and pass criterion. The scientific weight is in the metric
implementations and the citation list inside each test function's docstring
(under a `References` heading).

## Adding a test

1. Write `def my_test(ctx: QAContext) -> Result:` in `tests.py`.
2. Add it to `DEFAULT_TESTS` at the bottom of `tests.py`.
3. If it needs a new viz, add a function to `viz.py` returning a `Figure`.
4. If it needs a new metric, add it to `metrics.py`.
5. Cite the reference standard or paper under a `References` heading in
   the test's docstring (numpydoc style). If you can't cite a source, the
   test probably isn't industry-grade yet. If the test has known quality
   targets, populate `Result.reference_values` so the report can show
   them alongside the bundle's numbers.
