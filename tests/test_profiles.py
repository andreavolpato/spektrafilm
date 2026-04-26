import json
import numpy as np
import pytest
import ast
import inspect

from spektrafilm.model import stocks
from spektrafilm.profiles.io import Profile, _json_safe, load_profile, profile_to_dict, profile_from_dict


class TestLoadProfile:
    def test_profile_has_required_fields(self, portra_400_profile):
        p = portra_400_profile
        assert hasattr(p, 'info')
        assert hasattr(p, 'data')
        assert hasattr(p.data, 'log_sensitivity')
        assert hasattr(p.data, 'bandpass_hanatos2025')
        assert hasattr(p.data, 'density_curves')
        assert hasattr(p.data, 'channel_density')
        assert hasattr(p.data, 'base_density')
        assert hasattr(p.data, 'midscale_neutral_density')
        assert hasattr(p.data, 'log_exposure')
        assert hasattr(p.data, 'wavelengths')

    @pytest.mark.parametrize(
        'stock',
        [
            'kodak_portra_400',
            'fujifilm_c200',
            'kodak_portra_endura',
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
        assert profile.data.bandpass_hanatos2025.ndim == 2
        assert profile.data.bandpass_hanatos2025.size == 0 or profile.data.bandpass_hanatos2025.shape == profile.data.log_sensitivity.shape

        assert profile.data.wavelengths.ndim == 1
        assert profile.data.channel_density.ndim == 2
        assert profile.data.channel_density.shape[0] == profile.data.wavelengths.shape[0]
        assert profile.data.channel_density.shape[1] == 3
        assert profile.data.base_density.ndim == 1
        assert profile.data.base_density.shape[0] == profile.data.wavelengths.shape[0]
        assert profile.data.midscale_neutral_density.ndim == 1
        assert profile.data.midscale_neutral_density.shape[0] == profile.data.wavelengths.shape[0]

    def test_profile_namespace_round_trip_preserves_core_fields(self, portra_400_profile):
        profile_dict = profile_to_dict(portra_400_profile)
        profile_rt = profile_from_dict(profile_dict)

        assert profile_rt.info.stock == portra_400_profile.info.stock
        assert np.array(profile_rt.data.log_exposure).shape == portra_400_profile.data.log_exposure.shape
        assert np.array(profile_rt.data.density_curves).shape == portra_400_profile.data.density_curves.shape
        assert np.array(profile_rt.data.bandpass_hanatos2025).shape == portra_400_profile.data.bandpass_hanatos2025.shape

    def test_profile_json_payload_converts_nan_to_null(self, portra_400_profile):
        profile = portra_400_profile.clone()
        profile.data.log_sensitivity[0, 0] = np.nan

        payload = json.dumps(_json_safe(profile_to_dict(profile)), allow_nan=False)

        assert 'null' in payload

    def test_profile_json_round_trip_preserves_bandpass_hanatos2025(self, portra_400_profile):
        payload = json.dumps(_json_safe(profile_to_dict(portra_400_profile)), allow_nan=False)

        profile_rt = profile_from_dict(json.loads(payload))

        np.testing.assert_allclose(
            profile_rt.data.bandpass_hanatos2025,
            portra_400_profile.data.bandpass_hanatos2025,
        )

    def test_profile_constructor_rejects_dict_payloads(self):
        with pytest.raises(TypeError, match='ProfileInfo'):
            Profile(info={}, data={})

    def test_profile_clone_is_deep_copy(self, portra_400_profile):
        clone = portra_400_profile.clone()

        clone.data.log_exposure[0] += 1

        assert clone is not portra_400_profile
        assert clone.data is not portra_400_profile.data
        assert clone.info is not portra_400_profile.info
        assert clone.data.log_exposure[0] != portra_400_profile.data.log_exposure[0]

    def test_profile_update_helpers_replace_nested_dataclasses(self, portra_400_profile):
        original_data = portra_400_profile.data
        original_info = portra_400_profile.info
        updated_density = np.asarray(portra_400_profile.data.channel_density) * 0.5

        returned = portra_400_profile.update(
            info={'name': 'updated-name'},
            data={'channel_density': updated_density},
        )

        assert returned is portra_400_profile
        assert portra_400_profile.info is not original_info
        assert portra_400_profile.data is not original_data
        assert portra_400_profile.info.name == 'updated-name'
        np.testing.assert_allclose(portra_400_profile.data.channel_density, updated_density)


class TestDependencyBoundaries:
    def test_stocks_module_has_no_top_level_process_import(self):
        tree = ast.parse(inspect.getsource(stocks))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != 'spektrafilm.runtime.process'

    def test_stocks_module_has_no_main_script_block(self):
        tree = ast.parse(inspect.getsource(stocks))
        for node in tree.body:
            assert not isinstance(node, ast.If)



