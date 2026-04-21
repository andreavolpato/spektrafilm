from __future__ import annotations

import numpy as np
import pytest

import spektrafilm_profile_creator.data.loader as loader_module
from spektrafilm_profile_creator import RawProfile, RawProfileRecipe, load_raw_profile, load_stock_catalog


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    (
        'stock',
        'expected_type',
        'expected_support',
        'expected_stage',
        'expected_use',
        'expected_target_film',
        'expected_target_print',
        'expected_data_trustability',
        'expected_neutral_log_exposure_correction',
        'expected_stretch_curves',
        'expected_neutral_ramp_refinement',
    ),
    [
        ('kodak_portra_400', 'negative', 'film', 'filming', 'photo', None, 'kodak_portra_endura', 1.0, False, False, True),
        ('kodak_portra_endura', 'negative', 'paper', 'printing', 'photo', 'kodak_portra_400', None, None, False, None, True),
        ('kodak_2383', 'negative', 'film', 'printing', 'cine', 'kodak_vision3_250d', None, None, False, None, True),
        ('fujifilm_c200', 'negative', 'film', 'filming', 'photo', None, 'fujifilm_crystal_archive_typeii', 0.3, False, False, True),
        ('fujifilm_pro_400h', 'negative', 'film', 'filming', 'photo', None, 'fujifilm_crystal_archive_typeii', 0.3, False, False, True),
        ('kodak_ektachrome_100', 'positive', 'film', 'filming', 'photo', None, None, None, False, None, True),
    ],
    ids=['portra-film', 'portra-paper', 'vision-print-film', 'fuji-c200', 'fuji-pro-400h', 'ektachrome-positive'],
)
def test_load_raw_profile_reads_expected_info_and_recipe(
    stock: str,
    expected_type: str,
    expected_support: str,
    expected_stage: str,
    expected_use: str,
    expected_target_film: str | None,
    expected_target_print: str | None,
    expected_data_trustability: float | None,
    expected_neutral_log_exposure_correction: bool,
    expected_stretch_curves: bool | None,
    expected_neutral_ramp_refinement: bool,
) -> None:
    raw_profile = load_raw_profile(stock)

    assert isinstance(raw_profile, RawProfile)
    assert isinstance(raw_profile.recipe, RawProfileRecipe)
    assert raw_profile.info.stock == stock
    assert raw_profile.info.support == expected_support
    assert raw_profile.info.stage == expected_stage
    assert raw_profile.info.use == expected_use
    assert raw_profile.info.target_print == expected_target_print
    assert raw_profile.info.type == expected_type
    assert raw_profile.info.channel_model == 'color'
    assert raw_profile.recipe.target_film == expected_target_film
    assert raw_profile.recipe.neutral_log_exposure_correction is expected_neutral_log_exposure_correction
    assert raw_profile.recipe.neutral_ramp_refinement is expected_neutral_ramp_refinement

    if expected_stage == 'printing':
        assert load_stock_catalog()[stock].endswith('.print.negative')

    if expected_target_print is not None:
        assert raw_profile.recipe.dye_density_reconstruct_model == 'dmid_dmin'
        assert raw_profile.recipe.data_trustability == pytest.approx(expected_data_trustability)
        assert raw_profile.recipe.stretch_curves is expected_stretch_curves


def test_load_raw_profile_reads_reconstruct_model_from_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTraversable:
        def __truediv__(self, _name: str) -> '_FakeTraversable':
            return self

    monkeypatch.setattr(loader_module, 'load_stock_catalog', lambda: {'test_stock': 'fake.package'})
    monkeypatch.setattr(loader_module.pkg_resources, 'files', lambda _package: _FakeTraversable())
    monkeypatch.setattr(
        loader_module,
        '_load_profile_manifest',
        lambda _manifest, _stock: {
            'name': 'Test Stock',
            'profile': {
                'type': 'negative',
                'support': 'film',
                'stage': 'filming',
                'use': 'photo',
                'target_print': 'kodak_portra_endura',
                'channel_model': 'color',
            },
            'workflow': {'dye_density_reconstruct_model': 'workflow_model'},
            'recipe': {
                'dye_density_reconstruct_model': 'recipe_model',
                'target_film': 'kodak_portra_400',
                'neutral_log_exposure_correction': True,
                'neutral_ramp_refinement': True,
            },
        },
    )
    monkeypatch.setattr(
        loader_module,
        'load_stock_data',
        lambda **_kwargs: (
            np.empty((0, 3), dtype=float),
            np.empty((0, 5), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=float),
        ),
    )

    raw_profile = loader_module.load_raw_profile('test_stock')

    assert raw_profile.recipe.dye_density_reconstruct_model == 'recipe_model'
    assert raw_profile.recipe.target_film == 'kodak_portra_400'
    assert raw_profile.info.target_print == 'kodak_portra_endura'
    assert raw_profile.recipe.neutral_log_exposure_correction is True
    assert raw_profile.recipe.neutral_ramp_refinement is True


def test_load_raw_profile_manifest_reads_root_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTraversable:
        def __truediv__(self, _name: str) -> '_FakeTraversable':
            return self

    manifest_payload = {
        'name': 'Test Stock',
        'profile': {'stage': 'filming', 'use': 'photo', 'target_print': 'kodak_portra_endura'},
    }

    monkeypatch.setattr(loader_module, 'load_stock_catalog', lambda: {'test_stock': 'fake.package'})
    monkeypatch.setattr(loader_module.pkg_resources, 'files', lambda _package: _FakeTraversable())
    monkeypatch.setattr(loader_module, '_load_profile_manifest', lambda _manifest, _stock: manifest_payload)

    payload = loader_module.load_raw_profile_manifest('test_stock')

    assert payload is manifest_payload