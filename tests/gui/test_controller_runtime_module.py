from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spektrafilm_gui import controller_runtime as runtime_module


class FakeSignal:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    def emit(self, value) -> None:
        self.emitted.append(value)


def test_display_profile_name_falls_back_to_filename_stem() -> None:
    display_profile = SimpleNamespace(filename='C:/profiles/monitor.icc')
    imagecms_module = SimpleNamespace(getProfileName=lambda profile: '', PyCMSError=RuntimeError)

    assert runtime_module.display_profile_name(display_profile, imagecms_module=imagecms_module) == 'monitor'


def test_execute_simulation_request_uses_padding_and_preview_builder() -> None:
    request = runtime_module.SimulationRequest(
        mode_label='Preview',
        image=np.full((2, 2, 3), 0.25, dtype=np.float32),
        params=object(),
        output_color_space='ACES2065-1',
        use_display_transform=True,
        white_padding_fraction=0.5,
    )
    captured: dict[str, object] = {}

    result = runtime_module.execute_simulation_request(
        request,
        simulate_fn=lambda image, params: np.full((4, 4, 3), 0.5, dtype=np.float32),
        prepare_output_display_image_fn=lambda image, **kwargs: _capture_preview_result(captured, image, **kwargs),
        padding_pixels_for_image_fn=lambda image, fraction: 2,
    )

    np.testing.assert_allclose(captured['display_args']['image'], np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert captured['display_args']['padding_pixels'] == 2
    assert result.mode_label == 'Preview'
    np.testing.assert_allclose(result.float_image, np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert result.status_message == 'Display transform: active'


def test_simulation_worker_emits_failure_message() -> None:
    request = runtime_module.SimulationRequest(
        mode_label='Preview',
        image=np.zeros((1, 1, 3), dtype=np.float32),
        params=object(),
        output_color_space='sRGB',
        use_display_transform=False,
        white_padding_fraction=0.0,
    )
    worker = runtime_module.SimulationWorker(
        request,
        execute_request=lambda request: (_ for _ in ()).throw(ValueError('bad simulation')),
    )
    worker.signals = SimpleNamespace(finished=FakeSignal(), failed=FakeSignal())

    worker.run()

    assert worker.signals.finished.emitted == []
    assert worker.signals.failed.emitted == ['ValueError: bad simulation']


def test_apply_white_padding_preserves_float_fill_value() -> None:
    image = np.full((1, 1, 3), 0.25, dtype=np.float32)

    padded = runtime_module.apply_white_padding(image, 1)

    assert padded.shape == (3, 3, 3)
    np.testing.assert_allclose(padded[1, 1], np.array([0.25, 0.25, 0.25], dtype=np.float32))
    np.testing.assert_allclose(padded[0, 0], np.array([1.0, 1.0, 1.0], dtype=np.float32))


def _capture_preview_result(captured: dict[str, object], image: np.ndarray, **kwargs):
    captured['display_args'] = {'image': image.copy(), **kwargs}
    return np.full((6, 6, 3), 99, dtype=np.uint8), 'Display transform: active'