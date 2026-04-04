from __future__ import annotations

import numpy as np
import pytest

import spektrafilm_profile_creator.data.loader as loader_module
from spektrafilm_profile_creator import RawProfile, RawProfileRecipe, load_raw_profile, load_stock_catalog


pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    (
        'stock',
        'expected_support',
        'expected_use',
        'expected_reference_channel',
        'expected_target_film',
        'expected_target_print',
        'expected_data_trustability',
        'expected_stretch_curves',
    ),
    [
        ('kodak_portra_400', 'film', 'filming', None, None, 'kodak_portra_endura', 1.0, False),
        ('kodak_portra_endura', 'paper', 'printing', None, 'kodak_portra_400', None, None, None),
        ('kodak_2383', 'film', 'printing', None, 'kodak_vision3_250d', None, None, None),
        ('fujifilm_c200', 'film', 'filming', 'green', None, None, None, None),
        ('fujifilm_pro_400h', 'film', 'filming', 'mid', None, None, None, None),
    ],
    ids=['portra-film', 'portra-paper', 'vision-print-film', 'fuji-c200', 'fuji-pro-400h'],
)
def test_load_raw_profile_reads_expected_info_and_recipe(
    stock: str,
    expected_support: str,
    expected_use: str,
    expected_reference_channel: str | None,
    expected_target_film: str | None,
    expected_target_print: str | None,
    expected_data_trustability: float | None,
    expected_stretch_curves: bool | None,
) -> None:
    raw_profile = load_raw_profile(stock)

    assert isinstance(raw_profile, RawProfile)
    assert isinstance(raw_profile.recipe, RawProfileRecipe)
    assert raw_profile.info.stock == stock
    assert raw_profile.info.support == expected_support
    assert raw_profile.info.use == expected_use
    assert raw_profile.info.type == 'negative'
    assert raw_profile.info.channel_model == 'color'
    assert raw_profile.recipe.reference_channel == expected_reference_channel
    assert raw_profile.recipe.target_film == expected_target_film

    if expected_use == 'printing':
        assert load_stock_catalog()[stock].endswith('.print.negative')

    if expected_target_print is not None:
        assert raw_profile.recipe.dye_density_reconstruct_model == 'dmid_dmin'
        assert raw_profile.recipe.target_print == expected_target_print
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
                'use': 'filming',
                'channel_model': 'color',
            },
            'workflow': {'dye_density_reconstruct_model': 'workflow_model'},
            'recipe': {'dye_density_reconstruct_model': 'recipe_model', 'target_film': 'kodak_portra_400'},
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


def test_load_raw_profile_manifest_reads_root_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTraversable:
        def __truediv__(self, _name: str) -> '_FakeTraversable':
            return self

    manifest_payload = {
        'name': 'Test Stock',
        'profile': {'use': 'filming'},
        'recipe': {'target_print': 'kodak_portra_endura'},
    }

    monkeypatch.setattr(loader_module, 'load_stock_catalog', lambda: {'test_stock': 'fake.package'})
    monkeypatch.setattr(loader_module.pkg_resources, 'files', lambda _package: _FakeTraversable())
    monkeypatch.setattr(loader_module, '_load_profile_manifest', lambda _manifest, _stock: manifest_payload)

    payload = loader_module.load_raw_profile_manifest('test_stock')

    assert payload is manifest_payload