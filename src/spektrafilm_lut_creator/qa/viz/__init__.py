"""QA visualization library, grouped by topic.

Topic modules:

- :mod:`._base` — palette + plot-style primitives shared by every panel
- :mod:`.lut_fidelity` — panels for the 5 LUT-fidelity tests
- :mod:`.model_diagnostic` — panels for the 5 model-diagnostic tests
- :mod:`.picture_style` — panels for the 4 picture-style tests

The gamut-compression test panels live inline in
``qa/tests/gamut_compression.py`` (no shared viz fns) and aren't re-
exported here.

Existing ``from spektrafilm_lut_creator.qa import viz; viz.X(...)`` call
sites keep working because every public name lands in this namespace.
"""

from __future__ import annotations

from spektrafilm_lut_creator.qa.viz._base import (
    BG,
    BLUE,
    DIM,
    FG,
    FOOTER_BAND_FRAC,
    FOOTER_COLOR,
    FOOTER_FS,
    GREEN,
    GRID_RGBA,
    HEADER_BAND_FRAC,
    HI,
    IDENTITY_ALPHA,
    IDENTITY_COLOR,
    PANE_EDGE_RGBA,
    PANEL_TITLE_FS,
    PANEL_TITLE_PAD,
    RED,
    SUPTITLE_FS,
    SUPTITLE_PAD,
    WARN,
    _fill_3d,
    _gamut_triangle_xy,
    _identity_line,
    _setup_2d,
    _setup_3d,
    _to_oklab,
    add_footer,
)
from spektrafilm_lut_creator.qa.viz.lut_fidelity import (
    cube_deformation,
    cube_edges,
    cube_sculpture,
    gamut_compression_3d_xy,
    jacobian_condition_3d,
    offgrid_error_scatter,
    output_histograms,
    transfer_curves,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    chromaticity_1931,
    density_transfer_curves,
    dynamic_range_curve,
    hue_twist_oklab,
    oklab_ab_slices,
    oklab_displacement,
    planckian_path,
    spectral_locus_envelope,
)
from spektrafilm_lut_creator.qa.viz.picture_style import (
    gamut_edge_stress,
    noise_gradient,
    noise_sensitivity,
    oklab_gamut_slice_outline,
    rg_plane_slices,
)
