from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from spektrafilm_profile_creator.diagnostics.messages import (
    get_diagnostic_profile_snapshots,
    log_event,
)
from spektrafilm_profile_creator.neutral_print_filters import NeutralPrintFilterFitResult
from spektrafilm_profile_creator.core.profile_transforms import apply_scale_shift_stretch_density_curves
from spektrafilm_profile_creator.refinement import (
    DensityCurvesCorrection,
    refine_negative_film,
    refine_negative_print,
    refine_positive_film,
)



import spektrafilm_profile_creator.refinement.negative_film as negative_film_module
import spektrafilm_profile_creator.refinement.negative_print as negative_print_module
import spektrafilm_profile_creator.refinement.positive_film as positive_film_module
import spektrafilm_profile_creator.refinement.common as refinement_common_module

from tests.profiles_creator.helpers import make_test_profile


def test_log_event_stores_profile_snapshot_as_deep_copy() -> None:
    profile = make_test_profile()

    log_event('diagnostic_event', profile, residual=np.array([0.1, 0.2, 0.3]))

    snapshots = get_diagnostic_profile_snapshots()

    assert list(snapshots) == ['diagnostic_event']
    entry = snapshots['diagnostic_event'][0]
    assert entry['sequence'] == 1
    assert entry['stock'] == profile.info.stock
    assert entry['output'].startswith('[profile_creator / diagnostic_test_stock] diagnostic_event')
    assert 'residual' in entry['output']
    np.testing.assert_allclose(entry['profile'].data.density_curves, profile.data.density_curves)

    profile.data.density_curves[0, 0] = 99.0
    snapshots['diagnostic_event'][0]['profile'].info.stock = 'mutated'

    refreshed = get_diagnostic_profile_snapshots()
    assert refreshed['diagnostic_event'][0]['profile'].info.stock == 'diagnostic_test_stock'
    assert refreshed['diagnostic_event'][0]['profile'].data.density_curves[0, 0] == pytest.approx(0.1)


def test_refine_negative_film_stores_corrected_profile_snapshot(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodak_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=True,
        ),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
        ),
    )

    class FakeRuntimeSimulationSession:
        def __init__(self, *, output_film_density_cmy: bool = False):
            self.output_film_density_cmy = output_film_density_cmy

        @classmethod
        def create(cls, *_args, **kwargs):
            return cls(output_film_density_cmy=kwargs.get('output_film_density_cmy', False))

        def refresh(self, *_args, **_kwargs):
            return None

        def render(self, _image, _profile, **_kwargs):
            if self.output_film_density_cmy:
                return np.array([[[0.75, 0.75, 0.75]]], dtype=np.float64)
            return np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)

        def gray_ramp(self, _profile, _ev_ramp, **_kwargs):
            return (
                np.array([
                    [0.18, 0.18, 0.18],
                    [0.184, 0.184, 0.184],
                    [0.19, 0.19, 0.19],
                    [0.20, 0.20, 0.20],
                    [0.21, 0.21, 0.21],
                    [0.22, 0.22, 0.22],
                    [0.23, 0.23, 0.23],
                ], dtype=np.float64),
                np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64),
            )

    monkeypatch.setattr(negative_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(
        negative_film_module,
        'RuntimeSimulationSession',
        FakeRuntimeSimulationSession,
    )
    monkeypatch.setattr(
        negative_film_module,
        'replace_fitted_density_curves',
        lambda profile: profile,
    )
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_filters',
        lambda *args, **kwargs: NeutralPrintFilterFitResult(
            c_filter=20.0,
            m_filter=40.0,
            y_filter=30.0,
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        negative_film_module.scipy.optimize,
        'least_squares',
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
    )
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_ramp',
        lambda *args, **kwargs: DensityCurvesCorrection(
            scale=(1.1, 0.9, 1.05),
            shift=(0.1, 0.0, -0.1),
            stretch=(1.0, 1.0, 1.0),
        ),
    )

    result: Any = refine_negative_film(source_profile, target_print='kodak_portra_endura')

    snapshots = get_diagnostic_profile_snapshots()
    entry = snapshots['refine_negative_film'][0]

    assert params.enlarger.y_filter_neutral == 30.0
    assert params.enlarger.m_filter_neutral == 40.0
    assert params.enlarger.c_filter_neutral == 20.0
    assert entry['stock'] == 'kodak_test_stock'
    assert entry['profile'] is not result
    np.testing.assert_allclose(entry['profile'].data.density_curves, result.data.density_curves)


def test_refine_positive_film_logs_anchor_and_neutral_maes(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodachrome_test_stock')
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(scan_film=False),
        settings=SimpleNamespace(rgb_to_raw_method=''),
    )

    class FakeRuntimeSimulationSession:
        @classmethod
        def create(cls, *_args, **_kwargs):
            return cls()

        def refresh(self, *_args, **_kwargs):
            return None

        def gray_ramp(self, _profile, ev_ramp, **_kwargs):
            if tuple(ev_ramp) == (0,):
                return (
                    np.array([[0.186, 0.183, 0.184]], dtype=np.float64),
                    np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64),
                )
            return (
                np.array([
                    [0.150, 0.160, 0.140],
                    [0.170, 0.180, 0.160],
                    [0.170, 0.180, 0.160],
                    [0.184, 0.182, 0.186],
                    [0.200, 0.210, 0.190],
                ], dtype=np.float64),
                np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64),
            )

    monkeypatch.setattr(positive_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(positive_film_module, 'replace_fitted_density_curves', lambda profile: profile)
    monkeypatch.setattr(
        positive_film_module,
        'RuntimeSimulationSession',
        FakeRuntimeSimulationSession,
    )
    monkeypatch.setattr(
        positive_film_module.scipy.optimize,
        'least_squares',
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0.05, -0.02, 0.01], dtype=np.float64)),
    )
    monkeypatch.setattr(
        positive_film_module,
        'fit_neutral_ramp',
        lambda *args, **kwargs: DensityCurvesCorrection(),
    )
    metric_values = iter([
        {
            'midgray_rgb': np.array([0.180, 0.186, 0.182], dtype=np.float64),
            'midgray_mae': 0.004,
            'ramp_neutrality_mae': 0.010,
        },
        {
            'midgray_rgb': np.array([0.184, 0.183, 0.185], dtype=np.float64),
            'midgray_mae': 0.001,
            'ramp_neutrality_mae': 0.006,
        },
        {
            'midgray_rgb': np.array([0.184, 0.184, 0.184], dtype=np.float64),
            'midgray_mae': 0.0,
            'ramp_neutrality_mae': 0.002,
        },
    ])
    monkeypatch.setattr(
        positive_film_module,
        '_positive_film_metrics',
        lambda *args, **kwargs: next(metric_values),
    )

    result: Any = refine_positive_film(source_profile)

    snapshots = get_diagnostic_profile_snapshots()
    anchor_entry = snapshots['fit_positive_film_gray_anchor'][0]
    refine_entry = snapshots['refine_positive_film'][0]

    assert anchor_entry['stock'] == 'kodachrome_test_stock'
    assert 'anchor_shift_correction' in anchor_entry['output']
    assert 'before_midgray_mae' in anchor_entry['output']
    assert 'anchored_midgray_mae' in anchor_entry['output']
    assert 'before_ramp_neutrality_mae' in anchor_entry['output']
    assert 'anchored_ramp_neutrality_mae' in anchor_entry['output']
    assert refine_entry['profile'] is not result
    assert 'before_midgray_mae' in refine_entry['output']
    assert 'anchored_midgray_mae' in refine_entry['output']
    assert 'ending_midgray_mae' in refine_entry['output']
    assert 'before_ramp_neutrality_mae' in refine_entry['output']
    assert 'anchored_ramp_neutrality_mae' in refine_entry['output']
    assert 'ending_ramp_neutrality_mae' in refine_entry['output']
    assert '0.004' in refine_entry['output']
    assert '0.001' in refine_entry['output']
    assert '0.002' in refine_entry['output']


def test_refine_negative_print_logs_anchor_and_neutral_maes(monkeypatch) -> None:
    source_profile = make_test_profile(stock='print_test_stock')
    target_film_profile = make_test_profile(stock='film_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=target_film_profile.clone(),
        print=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=False),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=False,
        ),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
            normalize_print_exposure=False,
        ),
    )

    class FakeRuntimeSimulationSession:
        @classmethod
        def create(cls, *_args, **_kwargs):
            return cls()

        def refresh(self, *_args, **_kwargs):
            return None

        def render(self, *_args, **kwargs):
            print_exposure = float(kwargs.get('print_exposure', 1.0))
            return np.array([[[0.184 + 0.001 * print_exposure, 0.184, 0.184]]], dtype=np.float64)

    monkeypatch.setattr(
        negative_print_module,
        '_build_negative_print_runtime_params',
        lambda *args, **kwargs: (params, target_film_profile),
    )
    monkeypatch.setattr(
        negative_print_module,
        'RuntimeSimulationSession',
        FakeRuntimeSimulationSession,
    )
    monkeypatch.setattr(
        negative_print_module,
        'fit_gray_anchor',
        lambda *args, **kwargs: DensityCurvesCorrection(shift=(0.05, -0.02, 0.01)),
    )
    monkeypatch.setattr(
        negative_print_module,
        'fit_neutral_ramp',
        lambda *args, **kwargs: DensityCurvesCorrection(),
    )
    monkeypatch.setattr(
        negative_print_module,
        '_convert_exposure_ev_ramp_out_to_print_exposure_ramp_in',
        lambda *args, **kwargs: np.array([0.5, 0.8, 1.0, 1.2, 1.5], dtype=np.float64),
    )
    metric_values = iter([
        {
            'midgray_rgb': np.array([0.180, 0.186, 0.182], dtype=np.float64),
            'midgray_mae': 0.004,
            'ramp_neutrality_mae': 0.010,
        },
        {
            'midgray_rgb': np.array([0.184, 0.183, 0.185], dtype=np.float64),
            'midgray_mae': 0.001,
            'ramp_neutrality_mae': 0.006,
        },
        {
            'midgray_rgb': np.array([0.184, 0.184, 0.184], dtype=np.float64),
            'midgray_mae': 0.0,
            'ramp_neutrality_mae': 0.002,
        },
    ])
    monkeypatch.setattr(
        negative_print_module,
        '_negative_print_metrics',
        lambda *args, **kwargs: next(metric_values),
    )

    result: Any = refine_negative_print(
        source_profile,
        target_film='kodak_vision3_250d',
        exposure_ev_ramp=(-1, -0.5, 0, 0.5, 1),
    )

    snapshots = get_diagnostic_profile_snapshots()
    anchor_entry = snapshots['fit_negative_print_gray_anchor'][0]
    refine_entry = snapshots['refine_negative_print'][0]

    assert anchor_entry['stock'] == 'print_test_stock'
    assert 'anchor_shift_correction' in anchor_entry['output']
    assert 'before_midgray_mae' in anchor_entry['output']
    assert 'anchored_midgray_mae' in anchor_entry['output']
    assert 'before_ramp_neutrality_mae' in anchor_entry['output']
    assert 'anchored_ramp_neutrality_mae' in anchor_entry['output']
    assert refine_entry['profile'] is not result
    assert 'before_midgray_mae' in refine_entry['output']
    assert 'anchored_midgray_mae' in refine_entry['output']
    assert 'ending_midgray_mae' in refine_entry['output']
    assert 'before_ramp_neutrality_mae' in refine_entry['output']
    assert 'anchored_ramp_neutrality_mae' in refine_entry['output']
    assert 'ending_ramp_neutrality_mae' in refine_entry['output']
    assert '0.004' in refine_entry['output']
    assert '0.001' in refine_entry['output']
    assert '0.002' in refine_entry['output']


def test_build_negative_film_runtime_params_disables_print_exposure_normalization(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodak_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=True,
        ),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
            normalize_print_exposure=True,
        ),
    )
    captured_kwargs = {}

    monkeypatch.setattr(negative_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(negative_film_module, 'replace_fitted_density_curves', lambda profile: profile)

    def fake_fit_neutral_filters(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return NeutralPrintFilterFitResult(
            c_filter=20.0,
            m_filter=40.0,
            y_filter=30.0,
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        )

    monkeypatch.setattr(negative_film_module, 'fit_neutral_filters', fake_fit_neutral_filters)

    build_negative_film_runtime_params = getattr(
        negative_film_module,
        '_build_negative_film_runtime_params',
    )

    build_negative_film_runtime_params(
        source_profile,
        'kodak_portra_endura',
    )

    assert captured_kwargs['normalize_print_exposure'] is False


def test_build_negative_film_runtime_params_preserves_fitted_filters_through_runtime_digest(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodak_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=True,
            use_enlarger_lut=False,
            use_scanner_lut=False,
        ),
        debug=SimpleNamespace(
            deactivate_spatial_effects=False,
            deactivate_stochastic_effects=False,
            debug_mode='off',
            output_film_log_raw=False,
            output_film_density_cmy=False,
            output_print_density_cmy=False,
            inject_film_density_cmy=False,
        ),
        print_render=SimpleNamespace(glare=SimpleNamespace(active=True)),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
            normalize_print_exposure=False,
        ),
    )

    monkeypatch.setattr(negative_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(negative_film_module, 'replace_fitted_density_curves', lambda profile: profile)
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_filters',
        lambda *args, **kwargs: NeutralPrintFilterFitResult(
            c_filter=20.0,
            m_filter=40.0,
            y_filter=30.0,
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        ),
    )

    def fake_digest_params(runtime_params):
        if runtime_params.settings.neutral_print_filters_from_database:
            runtime_params.enlarger.c_filter_neutral = 0.0
            runtime_params.enlarger.m_filter_neutral = 48.0
            runtime_params.enlarger.y_filter_neutral = 59.0
        return runtime_params

    monkeypatch.setattr(refinement_common_module, 'digest_params', fake_digest_params)

    captured_runtime_params: list[Any] = []

    class FakeSimulator:
        def __init__(self, runtime_params):
            captured_runtime_params.append(runtime_params)

    monkeypatch.setattr(refinement_common_module, 'Simulator', FakeSimulator)

    build_negative_film_runtime_params = negative_film_module.__dict__[
        '_build_negative_film_runtime_params'
    ]

    built_params, _, _ = build_negative_film_runtime_params(
        source_profile,
        'kodak_portra_endura',
    )
    refinement_common_module.RuntimeSimulationSession.create(built_params)
    digested_params = captured_runtime_params[0]

    assert built_params.settings.neutral_print_filters_from_database is False
    assert digested_params.enlarger.c_filter_neutral == pytest.approx(20.0)
    assert digested_params.enlarger.m_filter_neutral == pytest.approx(40.0)
    assert digested_params.enlarger.y_filter_neutral == pytest.approx(30.0)


def test_build_negative_film_runtime_params_keeps_runtime_print_exposure_normalization_disabled(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodak_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=True,
        ),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
            normalize_print_exposure=False,
        ),
    )

    monkeypatch.setattr(negative_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(negative_film_module, 'replace_fitted_density_curves', lambda profile: profile)
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_filters',
        lambda *args, **kwargs: NeutralPrintFilterFitResult(
            c_filter=20.0,
            m_filter=40.0,
            y_filter=30.0,
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        ),
    )

    build_negative_film_runtime_params = negative_film_module.__dict__[
        '_build_negative_film_runtime_params'
    ]

    built_params, _, _ = build_negative_film_runtime_params(
        source_profile,
        'kodak_portra_endura',
    )

    assert built_params.enlarger.normalize_print_exposure is False


def test_refine_negative_film_returns_corrected_fitted_working_profile(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodak_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=True,
        ),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
            normalize_print_exposure=False,
        ),
    )

    class FakeRuntimeSimulationSession:
        def __init__(self, *, output_film_density_cmy: bool = False):
            self.output_film_density_cmy = output_film_density_cmy

        @classmethod
        def create(cls, *_args, **kwargs):
            return cls(output_film_density_cmy=kwargs.get('output_film_density_cmy', False))

        def refresh(self, *_args, **_kwargs):
            return None

        def render(self, _image, _profile, **_kwargs):
            if self.output_film_density_cmy:
                return np.array([[[0.75, 0.75, 0.75]]], dtype=np.float64)
            return np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)

        def gray_ramp(self, _profile, _ev_ramp, **_kwargs):
            return (
                np.ones((5, 3), dtype=np.float64) * 0.184,
                np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64),
            )

    def replace_with_shifted_curves(profile: Any) -> Any:
        return profile.update_data(density_curves=np.asarray(profile.data.density_curves, dtype=np.float64) + 1.0)

    monkeypatch.setattr(negative_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(negative_film_module, 'RuntimeSimulationSession', FakeRuntimeSimulationSession)
    monkeypatch.setattr(negative_film_module, 'replace_fitted_density_curves', replace_with_shifted_curves)
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_filters',
        lambda *args, **kwargs: NeutralPrintFilterFitResult(
            c_filter=20.0,
            m_filter=40.0,
            y_filter=30.0,
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        negative_film_module.scipy.optimize,
        'least_squares',
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
    )
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_ramp',
        lambda *args, **kwargs: DensityCurvesCorrection(
            scale=(2.0, 1.0, 1.0),
            shift=(0.0, 0.0, 0.0),
            stretch=(1.0, 1.0, 1.0),
        ),
    )

    result: Any = refine_negative_film(source_profile, target_print='kodak_portra_endura')

    fitted_working_profile = replace_with_shifted_curves(source_profile)
    expected = apply_scale_shift_stretch_density_curves(
        fitted_working_profile,
        density_scale=(2.0, 1.0, 1.0),
        log_exposure_shift=(0.0, 0.0, 0.0),
        log_exposure_stretch=(1.0, 1.0, 1.0),
    )

    np.testing.assert_allclose(result.data.density_curves, expected.data.density_curves)


def test_refine_negative_film_keeps_anchor_print_exposure_in_runtime_params(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodak_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=True,
        ),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
            normalize_print_exposure=False,
        ),
    )

    class FakeRuntimeSimulationSession:
        def __init__(self, *, output_film_density_cmy: bool = False):
            self.output_film_density_cmy = output_film_density_cmy

        @classmethod
        def create(cls, *_args, **kwargs):
            return cls(output_film_density_cmy=kwargs.get('output_film_density_cmy', False))

        def refresh(self, *_args, **_kwargs):
            return None

        def render(self, _image, _profile, **_kwargs):
            if self.output_film_density_cmy:
                return np.array([[[0.75, 0.75, 0.75]]], dtype=np.float64)
            return np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)

        def gray_ramp(self, _profile, _ev_ramp, **_kwargs):
            return (
                np.ones((5, 3), dtype=np.float64) * 0.184,
                np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64),
            )

    monkeypatch.setattr(negative_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(negative_film_module, 'RuntimeSimulationSession', FakeRuntimeSimulationSession)
    monkeypatch.setattr(negative_film_module, 'replace_fitted_density_curves', lambda profile: profile)
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_filters',
        lambda *args, **kwargs: NeutralPrintFilterFitResult(
            c_filter=20.0,
            m_filter=40.0,
            y_filter=30.0,
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        negative_film_module.scipy.optimize,
        'least_squares',
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0.0, 0.0, 0.0, 0.82], dtype=np.float64)),
    )
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_ramp',
        lambda *args, **kwargs: DensityCurvesCorrection(),
    )

    refine_negative_film(source_profile, target_print='kodak_portra_endura')

    assert params.enlarger.print_exposure == pytest.approx(0.82)


def test_refine_negative_film_keeps_runtime_print_exposure_normalization_disabled(monkeypatch) -> None:
    source_profile = make_test_profile(stock='kodak_test_stock').update_info(
        fitted_cmy_midscale_neutral_density=np.array([0.75, 0.75, 0.75], dtype=np.float64),
    )
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(
            rgb_to_raw_method='',
            neutral_print_filters_from_database=True,
        ),
        enlarger=SimpleNamespace(
            c_filter_neutral=0.0,
            y_filter_neutral=0.0,
            m_filter_neutral=0.0,
            print_exposure=1.0,
            print_exposure_compensation=False,
            normalize_print_exposure=False,
        ),
    )
    captured_normalize_print_exposure_values: list[bool] = []

    class FakeRuntimeSimulationSession:
        def __init__(self, *, output_film_density_cmy: bool = False):
            self.output_film_density_cmy = output_film_density_cmy

        @classmethod
        def create(cls, runtime_params, **kwargs):
            captured_normalize_print_exposure_values.append(
                bool(runtime_params.enlarger.normalize_print_exposure)
            )
            return cls(output_film_density_cmy=kwargs.get('output_film_density_cmy', False))

        def refresh(self, runtime_params):
            captured_normalize_print_exposure_values.append(
                bool(runtime_params.enlarger.normalize_print_exposure)
            )

        def render(self, _image, _profile, **_kwargs):
            if self.output_film_density_cmy:
                return np.array([[[0.75, 0.75, 0.75]]], dtype=np.float64)
            return np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)

        def gray_ramp(self, _profile, _ev_ramp, **_kwargs):
            return (
                np.ones((5, 3), dtype=np.float64) * 0.184,
                np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64),
            )

    monkeypatch.setattr(negative_film_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(negative_film_module, 'RuntimeSimulationSession', FakeRuntimeSimulationSession)
    monkeypatch.setattr(negative_film_module, 'replace_fitted_density_curves', lambda profile: profile)
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_filters',
        lambda *args, **kwargs: NeutralPrintFilterFitResult(
            c_filter=20.0,
            m_filter=40.0,
            y_filter=30.0,
            print_exposure=1.0,
            residual=np.zeros(3, dtype=np.float64),
        ),
    )
    monkeypatch.setattr(
        negative_film_module.scipy.optimize,
        'least_squares',
        lambda *args, **kwargs: SimpleNamespace(x=np.array([0.0, 0.0, 0.0, 0.82], dtype=np.float64)),
    )
    monkeypatch.setattr(
        negative_film_module,
        'fit_neutral_ramp',
        lambda *args, **kwargs: DensityCurvesCorrection(),
    )

    refine_negative_film(source_profile, target_print='kodak_portra_endura')

    assert captured_normalize_print_exposure_values
    assert all(value is False for value in captured_normalize_print_exposure_values)


def test_log_event_promotes_explicit_stock_to_prefix(capsys) -> None:
    log_event('fit_neutral_filters', stock='kodak_gold_200')

    captured = capsys.readouterr()

    assert captured.out == '[profile_creator / kodak_gold_200] fit_neutral_filters\n'