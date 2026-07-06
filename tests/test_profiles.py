import json
import numpy as np
import pytest

from spektrafilm.data.profiles_loader import (
    Profile,
    ProfileData,
    ProfileInfo,
    _json_safe,
    _validate_profile,
    load_profile,
    profile_to_dict,
    profile_from_dict,
)


def _make_synthetic_profile(n_ch, *, channel_model='bw', n_wl=81, n_le=32):
    """Build a shape-consistent profile with `n_ch` emulsion channels.

    Values are zeros — only array shapes matter for `_validate_profile`.
    """
    data = ProfileData(
        wavelengths=np.linspace(380.0, 780.0, n_wl),
        log_sensitivity=np.zeros((n_wl, n_ch)),
        channel_density=np.zeros((n_wl, n_ch)),
        base_density=np.zeros(n_wl),
        midscale_neutral_density=np.zeros(n_wl),
        log_exposure=np.linspace(-3.0, 0.0, n_le),
        density_curves=np.zeros((n_le, n_ch)),
    )
    info = ProfileInfo(stock='synthetic', channel_model=channel_model)
    return Profile(info=info, data=data)


class TestLoadProfile:
    def test_profile_has_required_fields(self, portra_400_profile):
        p = portra_400_profile
        assert hasattr(p, 'info')
        assert hasattr(p, 'data')
        assert hasattr(p.data, 'log_sensitivity')
        assert hasattr(p.data, 'hanatos2025_adaptation_window_params')
        assert hasattr(p.data, 'hanatos2025_adaptation_surface_params')
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
        # Adaptation params are optional: None when absent/empty, else shaped.
        window = profile.data.hanatos2025_adaptation_window_params
        surface = profile.data.hanatos2025_adaptation_surface_params
        assert window is None or window.ndim == 1
        assert surface is None or (surface.ndim == 2 and surface.shape[0] == 3)

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
        assert (
            np.array(profile_rt.data.hanatos2025_adaptation_window_params).shape
            == portra_400_profile.data.hanatos2025_adaptation_window_params.shape
        )
        assert (
            np.array(profile_rt.data.hanatos2025_adaptation_surface_params).shape
            == portra_400_profile.data.hanatos2025_adaptation_surface_params.shape
        )

    def test_profile_json_payload_converts_nan_to_null(self, portra_400_profile):
        profile = portra_400_profile.clone()
        profile.data.log_sensitivity[0, 0] = np.nan

        payload = json.dumps(_json_safe(profile_to_dict(profile)), allow_nan=False)

        assert 'null' in payload

    def test_profile_json_round_trip_preserves_hanatos2025_adaptation(self, portra_400_profile):
        payload = json.dumps(_json_safe(profile_to_dict(portra_400_profile)), allow_nan=False)

        profile_rt = profile_from_dict(json.loads(payload))

        np.testing.assert_allclose(
            profile_rt.data.hanatos2025_adaptation_window_params,
            portra_400_profile.data.hanatos2025_adaptation_window_params,
        )
        np.testing.assert_allclose(
            profile_rt.data.hanatos2025_adaptation_surface_params,
            portra_400_profile.data.hanatos2025_adaptation_surface_params,
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


class TestChannelCount:
    """Channel count is declared by channel_model: 1 for bw, 3 for color."""

    def test_bw_reports_one_channel(self):
        profile = _make_synthetic_profile(1, channel_model='bw')
        assert profile.info.n_channels == 1

    def test_color_reports_three_channels(self):
        profile = _make_synthetic_profile(3, channel_model='color')
        assert profile.info.n_channels == 3

    def test_n_channels_ignores_density_curve_columns(self):
        # A BW dev-time family adds density-curve columns; channel count stays 1.
        profile = _make_synthetic_profile(1, channel_model='bw')
        n_le = profile.data.log_exposure.shape[0]
        profile.data.density_curves = np.zeros((n_le, 4))  # 4 development times
        assert profile.info.n_channels == 1

    def test_bw_single_channel_validates(self):
        profile = _make_synthetic_profile(1, channel_model='bw')
        _validate_profile(profile, 'synthetic')  # must not raise

    def test_color_three_channel_validates(self):
        profile = _make_synthetic_profile(3, channel_model='color')
        _validate_profile(profile, 'synthetic')  # must not raise

    def test_bw_density_curves_may_have_more_columns_than_channels(self):
        # BW dev-time family: one density-curve column per development time,
        # decoupled from the single-emulsion channel count (n_ch=1).
        profile = _make_synthetic_profile(1, channel_model='bw')
        n_le = profile.data.log_exposure.shape[0]
        profile.data.density_curves = np.zeros((n_le, 4))
        _validate_profile(profile, 'synthetic')  # must not raise

    def test_color_density_curves_must_match_channel_count(self):
        # Color has no dev-time family, so density_curves columns must equal n_ch.
        profile = _make_synthetic_profile(3, channel_model='color')
        n_le = profile.data.log_exposure.shape[0]
        profile.data.density_curves = np.zeros((n_le, 4))
        with pytest.raises(ValueError, match='synthetic'):
            _validate_profile(profile, 'synthetic')

    def test_bw_with_three_channel_sensitivity_is_rejected(self):
        # bw declares n_ch=1, so 3-column spectral arrays must not validate.
        profile = _make_synthetic_profile(3, channel_model='bw')
        with pytest.raises(ValueError, match='synthetic'):
            _validate_profile(profile, 'synthetic')

    def test_color_with_one_channel_sensitivity_is_rejected(self):
        # color declares n_ch=3, so 1-column spectral arrays must not validate.
        profile = _make_synthetic_profile(1, channel_model='color')
        with pytest.raises(ValueError, match='synthetic'):
            _validate_profile(profile, 'synthetic')

    def test_sensitivity_density_disagreement_is_rejected(self):
        # log_sensitivity and channel_density must both match the declared count.
        profile = _make_synthetic_profile(3, channel_model='color')
        profile.data.channel_density = np.zeros((profile.data.wavelengths.shape[0], 1))
        with pytest.raises(ValueError, match='synthetic'):
            _validate_profile(profile, 'synthetic')


class TestFlexibleProfileData:
    """The data model adapts: absent/empty -> None, unknown keys tolerated."""

    def test_omitted_optional_fields_default_to_none(self):
        profile = profile_from_dict({'info': {'channel_model': 'bw'},
                                     'data': {'wavelengths': [380.0, 385.0]}})
        assert profile.data.density_curves_layers is None
        assert profile.data.density_curves_model is None
        assert profile.data.midscale_neutral_density is None
        assert profile.data.hanatos2025_adaptation_window_params is None
        assert profile.data.hanatos2025_adaptation_surface_params is None

    def test_empty_array_becomes_none(self):
        # "no data is no data": an explicit [] is treated the same as absent.
        profile = profile_from_dict({'data': {'midscale_neutral_density': []}})
        assert profile.data.midscale_neutral_density is None

    def test_present_field_is_coerced_to_float_array(self):
        profile = profile_from_dict({'data': {'wavelengths': [380, 385, 390]}})
        assert isinstance(profile.data.wavelengths, np.ndarray)
        assert profile.data.wavelengths.dtype == float

    def test_unknown_data_field_is_tolerated_with_warning(self):
        payload = {'info': {'channel_model': 'bw'},
                   'data': {'wavelengths': [380.0], 'experimental_curve': [1.0, 2.0]}}
        with pytest.warns(RuntimeWarning, match='experimental_curve'):
            profile = profile_from_dict(payload)
        assert not hasattr(profile.data, 'experimental_curve')
        np.testing.assert_array_equal(profile.data.wavelengths, [380.0])



