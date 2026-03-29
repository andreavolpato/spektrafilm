from types import SimpleNamespace

import numpy as np

from spektrafilm.utils import raw_file_processor


def test_process_raw_file_daylight_uses_linear_output_and_colour_conversion(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    captured = {}

    class Reader:
        daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        rgb_xyz_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            captured['postprocess'] = kwargs
            return raw_image

    def imread(path):
        captured['path'] = path
        return Reader()

    def fake_rgb_to_rgb(image, **kwargs):
        captured['image'] = image
        captured['colour'] = kwargs
        return image + 0.25

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=imread,
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )
    monkeypatch.setattr(raw_file_processor.colour, 'RGB_to_RGB', fake_rgb_to_rgb)

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        white_balance='daylight',
        output_colorspace='sRGB',
        output_cctf_encoding=False,
    )

    expected_linear = raw_image.astype(np.float32) / 65535.0
    np.testing.assert_allclose(captured['image'], expected_linear)
    np.testing.assert_allclose(result, expected_linear + 0.25)
    assert captured['path'] == 'example.nef'
    assert captured['postprocess']['output_color'] == 'ACES'
    assert captured['postprocess']['output_bps'] == 16
    assert captured['postprocess']['no_auto_bright'] is True
    assert captured['postprocess']['gamma'] == (1, 1)
    assert captured['postprocess']['user_wb'] == Reader.daylight_whitebalance
    assert captured['colour']['input_colourspace'] == raw_file_processor.colour.RGB_COLOURSPACES['ACES2065-1']
    assert captured['colour']['output_colourspace'] == raw_file_processor.colour.RGB_COLOURSPACES['sRGB']
    assert captured['colour']['apply_cctf_decoding'] is False
    assert captured['colour']['apply_cctf_encoding'] is False


def test_process_raw_file_as_shot_uses_camera_white_balance(monkeypatch):
    raw_image = np.full((1, 1, 3), 32768, dtype=np.uint16)
    captured = {}

    class Reader:
        daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        rgb_xyz_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            captured['postprocess'] = kwargs
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )

    result = raw_file_processor.load_and_process_raw_file('example.nef', white_balance='as_shot')

    np.testing.assert_allclose(result, raw_image.astype(np.float32) / 65535.0)
    assert captured['postprocess']['use_camera_wb'] is True
    assert 'user_wb' not in captured['postprocess']


def test_process_raw_file_tungsten_and_custom_derive_user_wb(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    captured = []
    matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float64,
    )

    class Reader:
        daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        rgb_xyz_matrix = matrix

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            captured.append(kwargs)
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )
    monkeypatch.setattr(raw_file_processor.colour, 'CCT_to_xy', lambda temperature, method='Kang 2002': np.array([temperature, 0.0]))
    monkeypatch.setattr(raw_file_processor.colour, 'xy_to_XYZ', lambda xy: np.array([0.25, 0.5, 1.0]))

    raw_file_processor.load_and_process_raw_file('example.nef', white_balance='tungsten')
    raw_file_processor.load_and_process_raw_file(
        'example.nef',
        white_balance='custom',
        temperature=3200.0,
        tint=0.85,
    )

    assert captured[0]['user_wb'] == [3.0, 1.0, 0.375, 1.0]
    assert captured[1]['user_wb'] == [3.0, 0.85, 0.375, 0.85]
