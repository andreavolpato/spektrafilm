from types import SimpleNamespace
import copy

import numpy as np
import pytest

from spektrafilm import AgXPhoto, Simulator, photo_params, simulate
from spektrafilm.model.stocks import FilmStocks, PrintPapers
from spektrafilm.runtime import pipeline as pipeline_module
from spektrafilm.runtime import process as process_module


pytestmark = pytest.mark.integration


class TestRuntimeApi:
    def test_simulate_matches_simulator_process(self, small_rgb_image, default_params):
        new_result = simulate(small_rgb_image, default_params)
        direct_result = Simulator(default_params).process(small_rgb_image)

        np.testing.assert_allclose(new_result, direct_result, atol=1e-12)

    def test_update_params_delegates_to_pipeline_without_public_state(self, monkeypatch):
        class FakePipeline:
            def __init__(self, params):
                self.label = params.label
                self.timings = {'label': params.label}

            def process(self, image):
                return f'processed-{self.label}-{image}'

            def update(self, params):
                self.label = params.label
                self.timings = {'label': params.label}

        monkeypatch.setattr(process_module, 'SimulationPipeline', FakePipeline)
        initial_params = SimpleNamespace(label='initial')
        updated_params = SimpleNamespace(label='updated')

        simulator = process_module.Simulator(initial_params)
        assert not hasattr(simulator, 'camera')
        assert not hasattr(simulator, 'timings')
        assert not hasattr(simulator, 'update')

        simulator.update_params(updated_params)

        assert simulator.process('frame') == 'processed-updated-frame'

    def test_soft_update_delegates_to_pipeline(self, monkeypatch):
        captured_kwargs = {}

        class FakePipeline:
            def __init__(self, params):
                self.label = params.label
                self.timings = {'label': params.label}

            def process(self, image):
                return image

            def soft_update(self, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(process_module, 'SimulationPipeline', FakePipeline)
        simulator = process_module.Simulator(SimpleNamespace(label='initial'))

        simulator.soft_update(print_exposure=1.5, exposure_compensation_ev=-0.25)

        assert captured_kwargs == {
            'print_exposure': 1.5,
            'exposure_compensation_ev': -0.25,
        }

    def test_soft_update_keeps_print_exposure_compensation_consistent_with_rebuild(self, default_params):
        params = copy.deepcopy(default_params)
        params.camera.auto_exposure = False
        params.enlarger.normalize_print_exposure = True
        params.enlarger.print_exposure_compensation = True

        image = np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)
        simulator = process_module.Simulator(copy.deepcopy(params))

        for exposure_compensation_ev in (-2.0, -1.0, 0.0, 1.0, 2.0):
            simulator.soft_update(exposure_compensation_ev=exposure_compensation_ev)
            soft_updated = simulator.process(image)

            rebuilt_params = copy.deepcopy(params)
            rebuilt_params.camera.exposure_compensation_ev = exposure_compensation_ev
            rebuilt = process_module.Simulator(rebuilt_params).process(image)

            np.testing.assert_allclose(soft_updated, rebuilt, atol=1e-12)

    def test_simulate_prints_pipeline_timings(self, monkeypatch, capsys):
        class FakePipeline:
            def __init__(self, params):
                del params
                self.timings = {'previous': 1.0}
                self._last_elapsed_time = None

            def process(self, image):
                self.timings.clear()
                start = pipeline_module.perf_counter()
                try:
                    self.timings['FilmingStage.expose'] = 0.012345
                    self.timings['ScanningStage.scan'] = 0.0004567
                    return image
                finally:
                    self._last_elapsed_time = pipeline_module.perf_counter() - start

            def get_timings(self):
                return self.timings

            def get_total_elapsed_time(self):
                return self._last_elapsed_time

            def format_timings(self):
                return pipeline_module.format_timings(
                    self.get_timings(),
                    total_elapsed_time=self.get_total_elapsed_time(),
                )

            def print_timings(self):
                print(self.format_timings())

        monkeypatch.setattr(process_module, 'SimulationPipeline', FakePipeline)
        ticks = iter((10.0, 10.1234))
        monkeypatch.setattr(pipeline_module, 'perf_counter', lambda: next(ticks))

        params = SimpleNamespace(label='timed')

        result = process_module.simulate('frame', params, digest_params_first=False, print_timings=True)

        assert result == 'frame'
        assert capsys.readouterr().out.strip() == (
            "Simulation timings\n"
            "  Total                 123 ms  100.0%\n"
            "  -------------------  -------  ------\n"
            "  FilmingStage.expose  \033[31m12.3 ms\033[0m  \033[31m 10.0%\033[0m\n"
            "  ScanningStage.scan   \033[31m 457 us\033[0m  \033[31m  0.4%\033[0m"
        )

    def test_art_extlut_compatibility_path_runs(self):
        """reference this https://github.com/artraweditor/ART/blob/master/tools/extlut/spektrafilm_mklut.py"""
        def make_art_params():
            params = photo_params(
                FilmStocks.kodak_portra_400.value,
                PrintPapers.kodak_portra_endura.value,
            )
            params.camera.auto_exposure = False
            params.camera.auto_exposure_method = 'median'
            params.camera.exposure_compensation_ev = 0.0
            params.debug.deactivate_spatial_effects = True
            params.debug.deactivate_stochastic_effects = True
            params.enlarger.lens_blur = 0.0
            params.enlarger.m_filter_shift = 0.0
            params.enlarger.print_exposure = 1.0
            params.enlarger.print_exposure_compensation = True
            params.enlarger.y_filter_shift = 0.0
            params.io.compute_negative = False
            params.io.crop = False
            params.io.full_image = True
            params.io.input_cctf_decoding = False
            params.io.input_color_space = 'sRGB'
            params.io.output_cctf_encoding = False
            params.io.output_color_space = 'ACES2065-1'
            params.io.preview_resize_factor = 1.0
            params.io.upscale_factor = 1.0
            params.scanner.lens_blur = 0.0
            params.scanner.unsharp_mask = (0.0, 0.0)
            params.settings.use_enlarger_lut = False
            params.settings.use_scanner_lut = False
            params.settings.rgb_to_raw_method = 'mallett2019'
            params.film_render.grain.active = False
            params.film_render.halation.active = False
            params.film_render.density_curve_gamma = 1.0
            params.film_render.dir_couplers.active = True
            params.film_render.dir_couplers.inhibition_strength = 1.0
            params.print_render.glare.active = False
            params.print_render.density_curve_gamma = 1.0
            return params

        image = np.array([[[0.184, 0.184, 0.184]]], dtype=np.float64)

        params = make_art_params()
        assert params.io.compute_negative is False
        assert params.io.full_image is True
        assert params.io.preview_resize_factor == 1.0

        output = AgXPhoto(params).process(image)
        assert output.shape == image.shape
        assert np.isfinite(output).all()

        shifted_params = make_art_params()
        shifted_params.enlarger.y_filter_shift = 0.5
        shifted_params.enlarger.m_filter_shift = -0.5
        shifted_output = AgXPhoto(shifted_params).process(image)
        assert shifted_output.shape == image.shape
        assert np.isfinite(shifted_output).all()