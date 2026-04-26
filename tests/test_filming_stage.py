from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import spektrafilm.runtime.stages.filming as filming_module


def test_rgb_to_film_raw_applies_hanatos_bandpass_to_sensitivity(monkeypatch) -> None:
    captured: dict[str, np.ndarray] = {}
    bandpass = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=float)

    def fake_rgb_to_raw_hanatos2025(
        rgb,
        sensitivity,
        *,
        color_space=None,
        apply_cctf_decoding=None,
        reference_illuminant=None,
        tc_lut=None,
    ):
        del color_space, apply_cctf_decoding, reference_illuminant, tc_lut
        captured['sensitivity'] = np.asarray(sensitivity, dtype=float)
        return np.ones(rgb.shape, dtype=float)

    monkeypatch.setattr(filming_module, 'rgb_to_raw_hanatos2025', fake_rgb_to_raw_hanatos2025)

    stage = object.__new__(filming_module.FilmingStage)
    setattr(stage, '_film', SimpleNamespace(
        info=SimpleNamespace(reference_illuminant='D55'),
        data=SimpleNamespace(
            log_sensitivity=np.zeros((2, 3), dtype=float),
            bandpass_hanatos2025=bandpass,
        ),
    ))
    setattr(stage, '_camera', SimpleNamespace(filter_uv=(0.0, 0.0, 0.0), filter_ir=(0.0, 0.0, 0.0)))
    setattr(stage, '_settings', SimpleNamespace(rgb_to_raw_method='hanatos2025', bandpass_hanatos2025=True))
    setattr(stage, '_lut_service', SimpleNamespace(get_filming_tc_lut=lambda sensitivity: None))

    rgb = np.ones((1, 1, 3), dtype=float)

    getattr(stage, '_rgb_to_film_raw')(rgb)

    np.testing.assert_allclose(captured['sensitivity'], bandpass)