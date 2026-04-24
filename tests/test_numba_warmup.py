import numpy as np

from spektrafilm.utils import numba_boost_hightlights as boost_module
from spektrafilm.utils import numba_warmup


def test_warmup_boost_highlights_uses_small_float64_sample(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_boost_highlights(x: np.ndarray, **kwargs: object) -> np.ndarray:
        calls['x'] = x.copy()
        calls['kwargs'] = kwargs
        return x

    monkeypatch.setattr(boost_module, 'boost_highlights', fake_boost_highlights)

    boost_module.warmup_boost_highlights()

    sample = calls['x']
    assert isinstance(sample, np.ndarray)
    assert sample.shape == (2, 2, 3)
    assert sample.dtype == np.float64
    np.testing.assert_allclose(sample, 1.0)
    assert calls['kwargs'] == {'boost_ev': 1.0, 'boost_range': 0.5, 'protect_ev': 0.0}


def test_global_warmup_includes_boost_highlights(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(numba_warmup, 'warmup_fast_stats', lambda: calls.append('fast_stats'))
    monkeypatch.setattr(numba_warmup, 'warmup_luts', lambda: calls.append('luts'))
    monkeypatch.setattr(numba_warmup, 'warmup_fast_interp', lambda: calls.append('fast_interp'))
    monkeypatch.setattr(numba_warmup, 'warmup_boost_highlights', lambda: calls.append('boost_highlights'))

    numba_warmup.warmup()

    assert calls == ['fast_stats', 'luts', 'fast_interp', 'boost_highlights']


def test_boost_highlights_only_boosts_values_above_x0() -> None:
    x = np.array([[[0.1], [0.184], [4.0]]], dtype=np.float64)

    y = boost_module.boost_highlights(
        x,
        boost_ev=1.0,
        boost_range=0.5,
        protect_ev=0.0,
        midgray=0.184,
    )

    np.testing.assert_allclose(y[0, 0, 0], x[0, 0, 0])
    np.testing.assert_allclose(y[0, 1, 0], x[0, 1, 0])
    assert y[0, 2, 0] > x[0, 2, 0]


def test_boost_highlights_preserves_normalized_behavior_under_scaling() -> None:
    raw = np.array([[[0.1], [0.184], [4.0], [8.0]]], dtype=np.float64)
    scale = 17.0

    y_raw = boost_module.boost_highlights(
        raw,
        boost_ev=1.5,
        boost_range=0.5,
        protect_ev=0.5,
        midgray=0.184,
    )
    y_scaled = boost_module.boost_highlights(
        raw * scale,
        boost_ev=1.5,
        boost_range=0.5,
        protect_ev=0.5,
        midgray=0.184 * scale,
    )

    np.testing.assert_allclose(y_scaled, y_raw * scale, rtol=1e-12, atol=1e-12)


def test_boost_highlights_identity_paths_return_output_buffer() -> None:
    x = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float64)
    out = np.empty_like(x)

    y_zero_boost = boost_module.boost_highlights(x, boost_ev=0.0, out=out)
    assert y_zero_boost is out
    np.testing.assert_allclose(out, x)

    y_clipped_x0 = boost_module.boost_highlights(
        x,
        boost_ev=1.0,
        protect_ev=4.0,
        midgray=1.0,
        out=out,
    )

    assert y_clipped_x0 is out
    np.testing.assert_allclose(out, x)