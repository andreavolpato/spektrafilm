from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spektrafilm.utils import raw_file_processor


def _assert_valid_rgb(image: np.ndarray) -> None:
    assert image.dtype == np.float32
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.size > 0
    assert np.all(np.isfinite(image))


def test_load_and_process_raw_file_smoke_without_external_raw(monkeypatch) -> None:
    raw_image = np.array(
        [
            [[8192, 8192, 8192], [12288, 8192, 6144]],
            [[16384, 12288, 8192], [8192, 12288, 16384]],
        ],
        dtype=np.uint16,
    )

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **_kwargs):
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )

    daylight = raw_file_processor.load_and_process_raw_file('synthetic.nef', white_balance='daylight')
    custom_daylight = raw_file_processor.load_and_process_raw_file(
        'synthetic.nef',
        white_balance='custom',
        temperature=6504.0,
        tint=1.0,
    )
    custom_tungsten = raw_file_processor.load_and_process_raw_file(
        'synthetic.nef',
        white_balance='custom',
        temperature=3200.0,
        tint=1.0,
    )
    custom_tungsten_tinted = raw_file_processor.load_and_process_raw_file(
        'synthetic.nef',
        white_balance='custom',
        temperature=3200.0,
        tint=0.85,
    )

    for image in (daylight, custom_daylight, custom_tungsten, custom_tungsten_tinted):
        _assert_valid_rgb(image)
        assert image.shape == daylight.shape

    # 6504 K should be the daylight identity point for the simplified WB path.
    np.testing.assert_allclose(custom_daylight, daylight, atol=1e-6)

    # A significantly warmer temperature setting should change the result.
    assert not np.allclose(custom_tungsten, daylight, atol=1e-4)

    # Tint is implemented as a direct green scaling, so it must reduce mean green.
    green_mean = float(np.mean(custom_tungsten[..., 1]))
    tinted_green_mean = float(np.mean(custom_tungsten_tinted[..., 1]))
    assert tinted_green_mean < green_mean
