from types import SimpleNamespace

import numpy as np

from spektrafilm.utils import raw_file_processor


def _stub_raw_reader(monkeypatch, raw_image: np.ndarray) -> None:
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


def _stub_exif(monkeypatch, **overrides) -> None:
    values = {
        'make': 'Canon',
        'model': 'Canon EOS 5D Mark IV',
        'lens_make': 'Canon',
        'lens_model': 'Canon EF 50mm f/1.8 STM',
        'focal_length': 50.0,
        'f_number': 2.8,
    }
    values.update(overrides)
    exif_metadata = raw_file_processor.ExifData(**values)
    monkeypatch.setattr(raw_file_processor, '_read_exif_metadata', lambda path: exif_metadata)


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
    assert 'user_wb' not in captured['postprocess']
    assert 'use_camera_wb' not in captured['postprocess']
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


def test_process_raw_file_tungsten_and_custom_apply_colour_science_adjustment(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    postprocess_calls = []
    adaptation_calls = []

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            postprocess_calls.append(kwargs)
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )
    monkeypatch.setattr(raw_file_processor.colour, 'CCT_to_xy', lambda temperature, method='CIE Illuminant D Series': np.array([temperature, 0.0]))
    monkeypatch.setattr(raw_file_processor.colour, 'xy_to_XYZ', lambda xy: np.array([xy[0] / 1000.0, 1.0, xy[0] / 2000.0]))
    monkeypatch.setattr(raw_file_processor.colour, 'RGB_to_XYZ', lambda image, **kwargs: image)

    def fake_chromatic_adaptation(xyz, source_white_xyz, target_white_xyz, **kwargs):
        adaptation_calls.append(
            {
                'source': source_white_xyz,
                'target': target_white_xyz,
                'kwargs': kwargs,
            }
        )
        return xyz + np.float32(0.25)

    monkeypatch.setattr(raw_file_processor.colour, 'chromatic_adaptation', fake_chromatic_adaptation)
    monkeypatch.setattr(raw_file_processor.colour, 'XYZ_to_RGB', lambda xyz, **kwargs: xyz)

    tungsten = raw_file_processor.load_and_process_raw_file('example.nef', white_balance='tungsten')
    custom = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        white_balance='custom',
        temperature=3200.0,
        tint=0.85,
    )

    expected_linear = raw_image.astype(np.float32) / 65535.0
    np.testing.assert_allclose(tungsten, expected_linear + 0.25)
    np.testing.assert_allclose(custom, (expected_linear + 0.25) * np.array([1.0, 0.85, 1.0], dtype=np.float32))
    assert all('user_wb' not in call for call in postprocess_calls)
    assert all('use_camera_wb' not in call for call in postprocess_calls)
    assert len(adaptation_calls) == 2
    np.testing.assert_allclose(adaptation_calls[0]['source'], [2.85, 1.0, 1.425])
    np.testing.assert_allclose(adaptation_calls[0]['target'], [6.504, 1.0, 3.252])
    np.testing.assert_allclose(adaptation_calls[1]['source'], [3.2, 1.0, 1.6])
    np.testing.assert_allclose(adaptation_calls[1]['target'], [6.504, 1.0, 3.252])
    assert adaptation_calls[0]['kwargs']['method'] == 'Von Kries'
    assert adaptation_calls[1]['kwargs']['method'] == 'Von Kries'


def test_process_raw_file_custom_daylight_uses_daylight_base_without_adjustment(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    captured = {}

    class Reader:
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
    monkeypatch.setattr(raw_file_processor.colour, 'chromatic_adaptation', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('unexpected chromatic adaptation')))

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        white_balance='custom',
        temperature=6504.0,
        tint=1.0,
    )

    np.testing.assert_allclose(result, raw_image.astype(np.float32) / 65535.0)
    assert 'user_wb' not in captured['postprocess']
    assert 'use_camera_wb' not in captured['postprocess']


def test_process_raw_file_custom_does_not_require_raw_white_balance_metadata(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    captured = {'postprocess': []}

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **kwargs):
            captured['postprocess'].append(kwargs)
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )
    monkeypatch.setattr(raw_file_processor.colour, 'CCT_to_xy', lambda temperature, method='CIE Illuminant D Series': np.array([temperature, 0.0]))
    monkeypatch.setattr(raw_file_processor.colour, 'xy_to_XYZ', lambda xy: np.array([xy[0] / 1000.0, 1.0, xy[0] / 2000.0]))
    monkeypatch.setattr(raw_file_processor.colour, 'RGB_to_XYZ', lambda image, **kwargs: image)
    monkeypatch.setattr(raw_file_processor.colour, 'chromatic_adaptation', lambda xyz, source_white_xyz, target_white_xyz, **kwargs: xyz + np.float32(0.25))
    monkeypatch.setattr(raw_file_processor.colour, 'XYZ_to_RGB', lambda xyz, **kwargs: xyz)

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        white_balance='custom',
        temperature=3200.0,
        tint=1.0,
    )

    expected_linear = raw_image.astype(np.float32) / 65535.0
    assert captured['postprocess'] == [
        {
            'output_color': 'ACES',
            'output_bps': 16,
            'no_auto_bright': True,
            'gamma': (1, 1),
        }
    ]
    np.testing.assert_allclose(result, expected_linear + 0.25)


def test_process_raw_file_skips_lens_correction_when_lens_model_missing(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    lens_info_out: dict[str, str] = {}

    def fail_database_creation():
        raise AssertionError('lensfun database should not be queried without a lens model')

    _stub_raw_reader(monkeypatch, raw_image)
    _stub_exif(monkeypatch, lens_model='')
    monkeypatch.setattr(raw_file_processor.lensfunpy, 'Database', fail_database_creation)

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        lens_correction=True,
        lens_info_out=lens_info_out,
    )

    np.testing.assert_allclose(result, raw_image.astype(np.float32) / 65535.0)
    assert lens_info_out == {}


def test_process_raw_file_leaves_lens_info_empty_when_no_correction_applied(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    lens_info_out: dict[str, str] = {}
    _stub_raw_reader(monkeypatch, raw_image)
    _stub_exif(monkeypatch)
    monkeypatch.setattr(
        raw_file_processor,
        '_apply_lens_correction',
        lambda rgb, exif_metadata: (rgb, ''),
    )

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        lens_correction=True,
        lens_info_out=lens_info_out,
    )

    np.testing.assert_allclose(result, raw_image.astype(np.float32) / 65535.0)
    assert lens_info_out == {}


def test_process_raw_file_populates_lens_info_when_correction_applied(monkeypatch):
    raw_image = np.full((1, 1, 3), 16384, dtype=np.uint16)
    lens_info_out: dict[str, str] = {}
    _stub_raw_reader(monkeypatch, raw_image)
    _stub_exif(monkeypatch)
    monkeypatch.setattr(
        raw_file_processor,
        '_apply_lens_correction',
        lambda rgb, exif_metadata: (rgb, 'Canon EF 50mm f/1.8 STM @ 50.0mm f/2.8'),
    )

    result = raw_file_processor.load_and_process_raw_file(
        'example.nef',
        lens_correction=True,
        lens_info_out=lens_info_out,
    )

    np.testing.assert_allclose(result, raw_image.astype(np.float32) / 65535.0)
    assert lens_info_out == {'summary': 'Canon EF 50mm f/1.8 STM @ 50.0mm f/2.8'}
