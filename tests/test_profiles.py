import numpy as np
import pytest
import ast
import inspect

from spectral_film_lab.model import stocks
from spectral_film_lab.profile_store.io import load_profile, profile_to_dict, profile_from_dict


class TestLoadProfile:
    def test_profile_has_required_fields(self, portra_400_profile):
        p = portra_400_profile
        assert hasattr(p, 'info')
        assert hasattr(p, 'data')
        assert hasattr(p.data, 'log_sensitivity')
        assert hasattr(p.data, 'density_curves')
        assert hasattr(p.data, 'dye_density')
        assert hasattr(p.data, 'log_exposure')
        assert hasattr(p.data, 'wavelengths')

    def test_density_curves_have_three_channels(self, portra_400_profile):
        dc = portra_400_profile.data.density_curves
        assert dc.ndim == 2
        assert dc.shape[1] == 3

    @pytest.mark.parametrize(
        'stock',
        [
            'kodak_portra_400_auc',
            'fujifilm_c200_auc',
            'kodak_portra_endura_uc',
        ],
    )
    def test_profile_data_shapes_are_consistent(self, stock):
        profile = load_profile(stock)

        assert profile.data.log_exposure.ndim == 1
        assert profile.data.density_curves.ndim == 2
        assert profile.data.density_curves.shape[1] == 3
        assert profile.data.density_curves.shape[0] == profile.data.log_exposure.shape[0]

        assert profile.data.log_sensitivity.ndim == 2
        assert profile.data.log_sensitivity.shape[1] == 3

        assert profile.data.wavelengths.ndim == 1
        assert profile.data.dye_density.ndim == 2
        assert profile.data.dye_density.shape[0] == profile.data.wavelengths.shape[0]
        assert profile.data.dye_density.shape[1] >= 4

    def test_profile_namespace_round_trip_preserves_core_fields(self, portra_400_profile):
        profile_dict = profile_to_dict(portra_400_profile)
        profile_rt = profile_from_dict(profile_dict)

        assert profile_rt.info.stock == portra_400_profile.info.stock
        assert np.array(profile_rt.data.log_exposure).shape == portra_400_profile.data.log_exposure.shape
        assert np.array(profile_rt.data.density_curves).shape == portra_400_profile.data.density_curves.shape


class TestDependencyBoundaries:
    def test_stocks_module_has_no_top_level_process_import(self):
        tree = ast.parse(inspect.getsource(stocks))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != 'spectral_film_lab.runtime.process'

    def test_stocks_module_has_no_main_script_block(self):
        tree = ast.parse(inspect.getsource(stocks))
        for node in tree.body:
            assert not isinstance(node, ast.If)


