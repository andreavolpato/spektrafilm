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

from spektrafilm_lut_creator.qa.viz._base import BG as BG
from spektrafilm_lut_creator.qa.viz._base import BLUE as BLUE
from spektrafilm_lut_creator.qa.viz._base import DIM as DIM
from spektrafilm_lut_creator.qa.viz._base import FG as FG
from spektrafilm_lut_creator.qa.viz._base import FOOTER_BAND_FRAC as FOOTER_BAND_FRAC
from spektrafilm_lut_creator.qa.viz._base import FOOTER_COLOR as FOOTER_COLOR
from spektrafilm_lut_creator.qa.viz._base import FOOTER_FS as FOOTER_FS
from spektrafilm_lut_creator.qa.viz._base import GREEN as GREEN
from spektrafilm_lut_creator.qa.viz._base import GRID_RGBA as GRID_RGBA
from spektrafilm_lut_creator.qa.viz._base import HEADER_BAND_FRAC as HEADER_BAND_FRAC
from spektrafilm_lut_creator.qa.viz._base import HI as HI
from spektrafilm_lut_creator.qa.viz._base import IDENTITY_ALPHA as IDENTITY_ALPHA
from spektrafilm_lut_creator.qa.viz._base import IDENTITY_COLOR as IDENTITY_COLOR
from spektrafilm_lut_creator.qa.viz._base import PANE_EDGE_RGBA as PANE_EDGE_RGBA
from spektrafilm_lut_creator.qa.viz._base import PANEL_TITLE_FS as PANEL_TITLE_FS
from spektrafilm_lut_creator.qa.viz._base import PANEL_TITLE_PAD as PANEL_TITLE_PAD
from spektrafilm_lut_creator.qa.viz._base import RED as RED
from spektrafilm_lut_creator.qa.viz._base import SUPTITLE_FS as SUPTITLE_FS
from spektrafilm_lut_creator.qa.viz._base import SUPTITLE_PAD as SUPTITLE_PAD
from spektrafilm_lut_creator.qa.viz._base import WARN as WARN
from spektrafilm_lut_creator.qa.viz._base import _fill_3d as _fill_3d
from spektrafilm_lut_creator.qa.viz._base import (
    _gamut_triangle_xy as _gamut_triangle_xy,
)
from spektrafilm_lut_creator.qa.viz._base import _identity_line as _identity_line
from spektrafilm_lut_creator.qa.viz._base import _setup_2d as _setup_2d
from spektrafilm_lut_creator.qa.viz._base import _setup_3d as _setup_3d
from spektrafilm_lut_creator.qa.viz._base import _to_oklab as _to_oklab
from spektrafilm_lut_creator.qa.viz._base import add_footer as add_footer
from spektrafilm_lut_creator.qa.viz.lut_fidelity import (
    cube_deformation as cube_deformation,
)
from spektrafilm_lut_creator.qa.viz.lut_fidelity import cube_edges as cube_edges
from spektrafilm_lut_creator.qa.viz.lut_fidelity import cube_sculpture as cube_sculpture
from spektrafilm_lut_creator.qa.viz.lut_fidelity import (
    gamut_compression_3d_xy as gamut_compression_3d_xy,
)
from spektrafilm_lut_creator.qa.viz.lut_fidelity import (
    jacobian_condition_3d as jacobian_condition_3d,
)
from spektrafilm_lut_creator.qa.viz.lut_fidelity import (
    offgrid_error_scatter as offgrid_error_scatter,
)
from spektrafilm_lut_creator.qa.viz.lut_fidelity import (
    output_histograms as output_histograms,
)
from spektrafilm_lut_creator.qa.viz.lut_fidelity import (
    transfer_curves as transfer_curves,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    chromaticity_1931 as chromaticity_1931,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    density_transfer_curves as density_transfer_curves,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    dynamic_range_curve as dynamic_range_curve,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    hue_twist_oklab as hue_twist_oklab,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    oklab_ab_slices as oklab_ab_slices,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    oklab_displacement as oklab_displacement,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    planckian_path as planckian_path,
)
from spektrafilm_lut_creator.qa.viz.model_diagnostic import (
    spectral_locus_envelope as spectral_locus_envelope,
)
from spektrafilm_lut_creator.qa.viz.picture_style import (
    gamut_edge_stress as gamut_edge_stress,
)
from spektrafilm_lut_creator.qa.viz.picture_style import (
    noise_gradient as noise_gradient,
)
from spektrafilm_lut_creator.qa.viz.picture_style import (
    noise_sensitivity as noise_sensitivity,
)
from spektrafilm_lut_creator.qa.viz.picture_style import (
    oklab_gamut_slice_outline as oklab_gamut_slice_outline,
)
from spektrafilm_lut_creator.qa.viz.picture_style import (
    rg_plane_slices as rg_plane_slices,
)
