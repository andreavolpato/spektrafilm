from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np

import spektrafilm_gui.virtual_photo_paper_back as virtual_photo_paper_back_module


def test_load_logo_alpha_reads_gui_asset() -> None:
    alpha = virtual_photo_paper_back_module.load_logo_alpha()

    assert alpha.dtype == np.float32
    assert alpha.ndim == 2
    assert alpha.shape[0] > 0
    assert alpha.shape[1] > 0
    assert alpha.flags.c_contiguous


def test_pyproject_includes_gui_assets_in_package_data() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / 'pyproject.toml'
    pyproject = tomllib.loads(pyproject_path.read_text(encoding='utf-8'))

    package_data = pyproject['tool']['setuptools']['package-data']

    assert 'spektrafilm_gui' in package_data
    assert 'assets/*' in package_data['spektrafilm_gui']
