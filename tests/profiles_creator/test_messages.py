from types import SimpleNamespace

import numpy as np
import pytest

from spektrafilm_profile_creator.diagnostics.messages import (
    get_diagnostic_profile_snapshots,
    log_event,
)
from spektrafilm_profile_creator.refinement import DensityCurvesCorrection, refine_negative_film
import spektrafilm_profile_creator.refinement as refinement_module

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
    source_profile = make_test_profile(stock='kodak_test_stock')
    params = SimpleNamespace(
        film=source_profile.clone(),
        io=SimpleNamespace(full_image=False),
        camera=SimpleNamespace(auto_exposure=True),
        settings=SimpleNamespace(rgb_to_raw_method=''),
        enlarger=SimpleNamespace(y_filter_neutral=0.0, m_filter_neutral=0.0),
    )

    monkeypatch.setattr(refinement_module, '_build_runtime_params', lambda *args, **kwargs: params)
    monkeypatch.setattr(refinement_module, 'fit_neutral_print_filters', lambda current_params, stock=None: (30.0, 40.0, None))
    monkeypatch.setattr(
        refinement_module,
        'fit_gray_anchor',
        lambda *args, **kwargs: DensityCurvesCorrection(),
    )
    monkeypatch.setattr(
        refinement_module,
        'fit_neutral_ramp',
        lambda *args, **kwargs: DensityCurvesCorrection(
            scale=(1.1, 0.9, 1.05),
            shift=(0.1, 0.0, -0.1),
            stretch=(1.0, 1.0, 1.0),
        ),
    )

    result = refine_negative_film(source_profile, target_print='kodak_portra_endura')

    snapshots = get_diagnostic_profile_snapshots()
    entry = snapshots['refine_negative_film'][0]

    assert entry['stock'] == 'kodak_test_stock'
    assert entry['profile'] is not result
    np.testing.assert_allclose(entry['profile'].data.density_curves, result.data.density_curves)


def test_log_event_promotes_explicit_stock_to_prefix(capsys) -> None:
    log_event('fit_neutral_print_filters', stock='kodak_gold_200')

    captured = capsys.readouterr()

    assert captured.out == '[profile_creator / kodak_gold_200] fit_neutral_print_filters\n'