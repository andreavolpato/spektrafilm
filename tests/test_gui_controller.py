from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spektrafilm_gui import controller as controller_module
from spektrafilm_gui.controller import (
    GuiController,
    INPUT_PADDING_PIXELS_KEY,
    INPUT_RAW_DATA_KEY,
    OUTPUT_CCTF_ENCODING_KEY,
    OUTPUT_COLOR_SPACE_KEY,
    OUTPUT_DISPLAY_TRANSFORM_KEY,
    OUTPUT_FLOAT_DATA_KEY,
)
from spektrafilm_gui.state import PROJECT_DEFAULT_GUI_STATE
from tests.gui_test_utils import FakeLayer, FakeViewer, make_controller_gui_state


def test_load_input_image_pads_display_but_preserves_raw_metadata(monkeypatch) -> None:
    viewer = FakeViewer([
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name="older-1"),
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name="older-2"),
    ])
    widgets = SimpleNamespace(filepicker=SimpleNamespace(set_available_layers=lambda *args, **kwargs: None))
    controller = GuiController(viewer=viewer, widgets=widgets)
    gui_state = make_controller_gui_state()
    gui_state.display.white_padding = 0.5
    raw_image = np.full((2, 2, 3), 0.25, dtype=np.float32)

    monkeypatch.setattr(controller_module, "load_image_oiio", lambda path: raw_image)
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)

    controller.load_input_image("C:/tmp/example.png")

    assert len(viewer.layers) == 3
    layer = viewer.layers[-1]
    assert layer.name == "example"
    assert layer.data.shape == (4, 4, 3)
    np.testing.assert_allclose(layer.data[1:3, 1:3], raw_image)
    np.testing.assert_allclose(layer.data[[0, -1], :, :], 1.0)
    np.testing.assert_allclose(layer.metadata[INPUT_RAW_DATA_KEY], raw_image)
    assert layer.metadata[INPUT_PADDING_PIXELS_KEY] == 1.0
    assert layer.visible is True
    assert all(other_layer.visible is False for other_layer in viewer.layers[:-1])
    assert viewer.reset_view_calls == 0


def test_select_input_layer_hides_other_layers_and_moves_target_to_top() -> None:
    selected_layer = FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name="selected")
    viewer = FakeViewer([
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name="older-1"),
        selected_layer,
        FakeLayer(np.zeros((2, 2, 3), dtype=np.float32), name="older-2"),
    ])
    controller = GuiController(viewer=viewer, widgets=object())
    controller._available_input_layers = lambda: list(viewer.layers)  # type: ignore[method-assign]

    controller.select_input_layer("selected")

    assert viewer.layers[-1] is selected_layer
    assert selected_layer.visible is True
    assert all(layer.visible is False for layer in viewer.layers[:-1])


def test_run_simulation_uses_unpadded_raw_input_metadata(monkeypatch) -> None:
    raw_image = np.full((2, 2, 3), 0.25, dtype=np.float32)
    padded_image = np.pad(raw_image, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=1.0)
    input_layer = FakeLayer(
        padded_image,
        metadata={
            INPUT_RAW_DATA_KEY: raw_image,
            INPUT_PADDING_PIXELS_KEY: 1.0,
        },
        name="input",
    )
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, "_selected_input_layer", lambda: input_layer)
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, "build_params_from_state", lambda state: object())

    def fake_simulate(image, params):
        captured["processing_input"] = image.copy()
        return np.full((2, 2, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(controller_module, "simulate", fake_simulate)
    monkeypatch.setattr(
        controller,
        "_prepare_output_display_image",
        lambda image_data, *, output_color_space, use_display_transform, padding_pixels=0.0: (
            np.full((2, 2, 3), 99, dtype=np.uint8),
            "Display transform: disabled",
        ),
    )
    monkeypatch.setattr(controller, "_set_or_add_output_layer", lambda image, **kwargs: captured.setdefault("output_layer", kwargs))
    monkeypatch.setattr(controller_module, "set_status", lambda *args, **kwargs: None)

    controller._run_simulation(compute_full_image=False)

    np.testing.assert_allclose(captured["processing_input"], raw_image)


def test_save_output_layer_converts_from_recorded_render_metadata(monkeypatch) -> None:
    float_image = np.full((2, 2, 3), 0.25, dtype=np.float32)
    output_layer = FakeLayer(
        np.uint8(float_image * 255),
        metadata={
            OUTPUT_FLOAT_DATA_KEY: float_image,
            OUTPUT_COLOR_SPACE_KEY: "sRGB",
            OUTPUT_CCTF_ENCODING_KEY: True,
            OUTPUT_DISPLAY_TRANSFORM_KEY: False,
        },
    )
    widgets = object()
    controller = GuiController(viewer=object(), widgets=widgets)
    captured: dict[str, object] = {}
    gui_state = make_controller_gui_state()

    monkeypatch.setattr(controller, "_output_layer", lambda: output_layer)
    monkeypatch.setattr(controller_module, "dialog_parent", lambda viewer: None)
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))
    monkeypatch.setattr(
        controller_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: ("output.png", "Images (*.png)")),
    )
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)

    def fake_rgb_to_rgb(image_data, input_color_space, output_color_space, apply_cctf_decoding, apply_cctf_encoding):
        captured["rgb_to_rgb"] = {
            "image_data": image_data.copy(),
            "input_color_space": input_color_space,
            "output_color_space": output_color_space,
            "apply_cctf_decoding": apply_cctf_decoding,
            "apply_cctf_encoding": apply_cctf_encoding,
        }
        return image_data + 0.1

    monkeypatch.setattr(controller_module.colour, "RGB_to_RGB", fake_rgb_to_rgb)
    monkeypatch.setattr(
        controller_module,
        "save_image_oiio",
        lambda filepath, image_data: captured.setdefault("saved", (filepath, image_data.copy())),
    )

    controller.save_output_layer()

    rgb_to_rgb_call = captured["rgb_to_rgb"]
    np.testing.assert_allclose(rgb_to_rgb_call["image_data"], float_image)
    assert rgb_to_rgb_call["input_color_space"] == "sRGB"
    assert rgb_to_rgb_call["output_color_space"] == "Display P3"
    assert rgb_to_rgb_call["apply_cctf_decoding"] is True
    assert rgb_to_rgb_call["apply_cctf_encoding"] is False
    saved_path, saved_image = captured["saved"]
    assert saved_path == "output.png"
    np.testing.assert_allclose(saved_image, float_image + 0.1)


def test_save_output_layer_skips_conversion_when_color_spaces_match(monkeypatch) -> None:
    float_image = np.full((2, 2, 3), 0.5, dtype=np.float32)
    output_layer = FakeLayer(
        np.uint8(float_image * 255),
        metadata={
            OUTPUT_FLOAT_DATA_KEY: float_image,
            OUTPUT_COLOR_SPACE_KEY: "Display P3",
            OUTPUT_CCTF_ENCODING_KEY: True,
            OUTPUT_DISPLAY_TRANSFORM_KEY: False,
        },
    )
    widgets = object()
    controller = GuiController(viewer=object(), widgets=widgets)
    captured: dict[str, object] = {}
    gui_state = make_controller_gui_state()
    gui_state.simulation.saving_color_space = "Display P3"
    gui_state.simulation.saving_cctf_encoding = True

    monkeypatch.setattr(controller, "_output_layer", lambda: output_layer)
    monkeypatch.setattr(controller_module, "dialog_parent", lambda viewer: None)
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))
    monkeypatch.setattr(
        controller_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: ("output.png", "Images (*.png)")),
    )
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)

    def fail_rgb_to_rgb(*args, **kwargs):
        raise AssertionError("RGB_to_RGB should not be called when color spaces match")

    monkeypatch.setattr(controller_module.colour, "RGB_to_RGB", fail_rgb_to_rgb)
    monkeypatch.setattr(
        controller_module,
        "save_image_oiio",
        lambda filepath, image_data: captured.setdefault("saved", (filepath, image_data.copy())),
    )

    controller.save_output_layer()

    saved_path, saved_image = captured["saved"]
    assert saved_path == "output.png"
    np.testing.assert_allclose(saved_image, float_image)


def test_save_output_layer_reencodes_when_only_cctf_flag_changes(monkeypatch) -> None:
    float_image = np.full((2, 2, 3), 0.5, dtype=np.float32)
    output_layer = FakeLayer(
        np.uint8(float_image * 255),
        metadata={
            OUTPUT_FLOAT_DATA_KEY: float_image,
            OUTPUT_COLOR_SPACE_KEY: "Display P3",
            OUTPUT_CCTF_ENCODING_KEY: True,
            OUTPUT_DISPLAY_TRANSFORM_KEY: False,
        },
    )
    widgets = object()
    controller = GuiController(viewer=object(), widgets=widgets)
    captured: dict[str, object] = {}
    gui_state = make_controller_gui_state()
    gui_state.simulation.saving_color_space = "Display P3"
    gui_state.simulation.saving_cctf_encoding = False

    monkeypatch.setattr(controller, "_output_layer", lambda: output_layer)
    monkeypatch.setattr(controller_module, "dialog_parent", lambda viewer: None)
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))
    monkeypatch.setattr(
        controller_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: ("output.png", "Images (*.png)")),
    )
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)

    def fake_rgb_to_rgb(image_data, input_color_space, output_color_space, apply_cctf_decoding, apply_cctf_encoding):
        captured["rgb_to_rgb"] = {
            "image_data": image_data.copy(),
            "input_color_space": input_color_space,
            "output_color_space": output_color_space,
            "apply_cctf_decoding": apply_cctf_decoding,
            "apply_cctf_encoding": apply_cctf_encoding,
        }
        return image_data - 0.1

    monkeypatch.setattr(controller_module.colour, "RGB_to_RGB", fake_rgb_to_rgb)
    monkeypatch.setattr(
        controller_module,
        "save_image_oiio",
        lambda filepath, image_data: captured.setdefault("saved", (filepath, image_data.copy())),
    )

    controller.save_output_layer()

    rgb_to_rgb_call = captured["rgb_to_rgb"]
    np.testing.assert_allclose(rgb_to_rgb_call["image_data"], float_image)
    assert rgb_to_rgb_call["input_color_space"] == "Display P3"
    assert rgb_to_rgb_call["output_color_space"] == "Display P3"
    assert rgb_to_rgb_call["apply_cctf_decoding"] is True
    assert rgb_to_rgb_call["apply_cctf_encoding"] is False
    saved_path, saved_image = captured["saved"]
    assert saved_path == "output.png"
    np.testing.assert_allclose(saved_image, float_image - 0.1)


def test_prepare_output_display_image_returns_simple_8bit_preview() -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space="sRGB",
        use_display_transform=False,
    )

    assert preview.dtype == np.uint8
    assert status == "Display transform: disabled"
    np.testing.assert_array_equal(preview, np.array([[[0, 127, 255]]], dtype=np.uint8))


def test_prepare_output_display_image_applies_padding_only_to_preview() -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space="sRGB",
        use_display_transform=False,
        padding_pixels=1.0,
    )

    assert preview.shape == (3, 3, 3)
    assert status == "Display transform: disabled"
    np.testing.assert_array_equal(preview[1, 1], np.array([63, 127, 191], dtype=np.uint8))
    np.testing.assert_array_equal(preview[0, 0], np.array([255, 255, 255], dtype=np.uint8))


def test_prepare_output_display_image_uses_imagecms_transform(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
    captured: dict[str, object] = {}

    class FakePILImage:
        def __init__(self, array: np.ndarray):
            self.array = array

    def fake_get_display_profile():
        return object()

    monkeypatch.setattr(controller_module.ImageCms, "get_display_profile", fake_get_display_profile)
    monkeypatch.setattr(controller_module.colour, "RGB_to_RGB", lambda *args, **kwargs: np.full((1, 1, 3), 0.5, dtype=np.float32))
    monkeypatch.setattr(controller_module.ImageCms, "createProfile", lambda name: f"profile:{name}")
    monkeypatch.setattr(
        controller_module.PILImage,
        "fromarray",
        lambda array, mode='RGB': captured.setdefault("source_image", FakePILImage(array.copy())),
    )

    def fake_profile_to_profile(source_image, source_profile, display_profile, outputMode='RGB'):
        captured["profile_to_profile"] = {
            "source_profile": source_profile,
            "display_profile": display_profile,
            "output_mode": outputMode,
            "image_data": source_image.array.copy(),
        }
        return np.full((1, 1, 3), 64, dtype=np.uint8)

    monkeypatch.setattr(controller_module.ImageCms, "profileToProfile", fake_profile_to_profile)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space="Display P3",
        use_display_transform=True,
    )

    np.testing.assert_array_equal(preview, np.full((1, 1, 3), 64, dtype=np.uint8))
    assert status == "Display transform: active"
    assert captured["profile_to_profile"]["source_profile"] == "profile:sRGB"
    assert captured["profile_to_profile"]["output_mode"] == "RGB"
    np.testing.assert_array_equal(captured["profile_to_profile"]["image_data"], np.full((1, 1, 3), 127, dtype=np.uint8))


def test_run_simulation_uses_display_transform_preview_when_enabled(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32))
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_controller_gui_state()
    gui_state.display.use_display_transform = True
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, "_selected_input_layer", lambda: input_layer)
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, "build_params_from_state", lambda state: object())
    monkeypatch.setattr(controller_module, "simulate", lambda image, params: np.full((2, 2, 3), 0.5, dtype=np.float32))

    def fake_prepare_output_display_image(image_data, *, output_color_space, use_display_transform, padding_pixels=0.0):
        captured["display_args"] = {
            "image_data": image_data.copy(),
            "output_color_space": output_color_space,
            "use_display_transform": use_display_transform,
            "padding_pixels": padding_pixels,
        }
        return np.full((2, 2, 3), 99, dtype=np.uint8), "Display transform: active"

    def fake_set_or_add_output_layer(image, **kwargs):
        captured["output_layer"] = {"image": image.copy(), **kwargs}

    monkeypatch.setattr(controller, "_prepare_output_display_image", fake_prepare_output_display_image)
    monkeypatch.setattr(controller, "_set_or_add_output_layer", fake_set_or_add_output_layer)

    controller._run_simulation(compute_full_image=False)

    np.testing.assert_allclose(captured["display_args"]["image_data"], np.full((2, 2, 3), 0.5, dtype=np.float32))
    assert captured["display_args"]["output_color_space"] == gui_state.simulation.output_color_space
    assert captured["display_args"]["use_display_transform"] is True
    assert captured["display_args"]["padding_pixels"] == 0.0
    np.testing.assert_array_equal(captured["output_layer"]["image"], np.full((2, 2, 3), 99, dtype=np.uint8))
    np.testing.assert_allclose(captured["output_layer"]["float_image"], np.full((2, 2, 3), 0.5, dtype=np.float32))


def test_run_preview_starts_async_preview(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        controller,
        "_start_simulation",
        lambda *, compute_full_image, mode_label: captured.setdefault(
            "call",
            {"compute_full_image": compute_full_image, "mode_label": mode_label},
        ),
    )

    controller.run_preview()

    assert captured["call"] == {"compute_full_image": False, "mode_label": "Preview"}


def test_run_scan_starts_async_scan(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        controller,
        "_start_simulation",
        lambda *, compute_full_image, mode_label: captured.setdefault(
            "call",
            {"compute_full_image": compute_full_image, "mode_label": mode_label},
        ),
    )

    controller.run_scan()

    assert captured["call"] == {"compute_full_image": True, "mode_label": "Scan"}


def test_start_simulation_reports_persistent_computing_status(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32))
    simulation_section = SimpleNamespace(preview_button=None, scan_button=None, save_button=None)
    widgets = SimpleNamespace(simulation=simulation_section)
    controller = GuiController(viewer=object(), widgets=widgets)
    gui_state = make_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, "_selected_input_layer", lambda: input_layer)
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, "build_params_from_state", lambda state: object())
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message, timeout_ms=5000: captured.setdefault("status", (message, timeout_ms)))
    monkeypatch.setattr(controller._thread_pool, "start", lambda worker: captured.setdefault("worker", worker))

    controller._start_simulation(compute_full_image=False, mode_label="Preview")

    assert captured["status"] == ("Computing preview...", 0)
    assert controller._active_simulation_label == "Preview"


def test_on_simulation_finished_reports_completed_status(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=SimpleNamespace(simulation=SimpleNamespace(preview_button=None, scan_button=None, save_button=None)))
    controller._active_simulation_label = "Preview"
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, "_set_or_add_output_layer", lambda image, **kwargs: captured.setdefault("output", (image, kwargs)))
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message, timeout_ms=5000: captured.setdefault("status", (message, timeout_ms)))

    controller._on_simulation_finished(
        controller_module.SimulationResult(
            mode_label="Preview",
            display_image=np.full((2, 2, 3), 9, dtype=np.uint8),
            float_image=np.full((2, 2, 3), 0.5, dtype=np.float32),
            output_color_space="sRGB",
            use_display_transform=False,
            status_message="Display transform: disabled",
        )
    )

    assert captured["status"] == ("Preview completed. Display transform: disabled", 5000)


def test_run_simulation_applies_white_padding_only_to_preview(monkeypatch) -> None:
    input_layer = SimpleNamespace(data=np.full((2, 2, 3), 0.25, dtype=np.float32))
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_controller_gui_state()
    gui_state.display.white_padding = 0.5
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, "_selected_input_layer", lambda: input_layer)
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, "build_params_from_state", lambda state: object())
    monkeypatch.setattr(controller_module, "simulate", lambda image, params: np.full((2, 2, 3), 0.5, dtype=np.float32))

    def fake_prepare_output_display_image(image_data, *, output_color_space, use_display_transform, padding_pixels=0.0):
        captured["display_input"] = image_data.copy()
        captured["display_padding_pixels"] = padding_pixels
        return np.full((4, 4, 3), 77, dtype=np.uint8), "Display transform: disabled"

    def fake_set_or_add_output_layer(image, **kwargs):
        captured["output_layer"] = {"image": image.copy(), **kwargs}

    monkeypatch.setattr(controller, "_prepare_output_display_image", fake_prepare_output_display_image)
    monkeypatch.setattr(controller, "_set_or_add_output_layer", fake_set_or_add_output_layer)

    controller._run_simulation(compute_full_image=False)

    display_input = captured["display_input"]
    assert display_input.shape == (2, 2, 3)
    np.testing.assert_allclose(display_input, np.full((2, 2, 3), 0.5, dtype=np.float32))
    assert captured["display_padding_pixels"] == 1.0
    np.testing.assert_array_equal(captured["output_layer"]["image"], np.full((4, 4, 3), 77, dtype=np.uint8))
    np.testing.assert_allclose(captured["output_layer"]["float_image"], np.full((2, 2, 3), 0.5, dtype=np.float32))


def test_padding_pixels_uses_fraction_of_long_edge() -> None:
    controller = GuiController(viewer=object(), widgets=object())

    assert controller._padding_pixels_for_image(np.zeros((40, 100, 3), dtype=np.float32), 0.05) == 5
    assert controller._padding_pixels_for_image(np.zeros((40, 100, 3), dtype=np.float32), 0.019) == 1
    assert controller._padding_pixels_for_image(np.zeros((40, 100, 3), dtype=np.float32), 0.009) == 0


def test_report_display_transform_status_disabled(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.report_display_transform_status(False)

    assert captured["status"] == "Display transform: disabled"


def test_report_display_transform_status_found(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module.ImageCms, "get_display_profile", lambda: object())
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.report_display_transform_status(True)

    assert captured["status"] == "Display transform: display profile found"


def test_report_display_transform_status_missing_profile(monkeypatch) -> None:
    class StubToggle:
        def __init__(self) -> None:
            self.checked = True
            self._signals_blocked = False

        def blockSignals(self, blocked: bool) -> bool:  # noqa: N802 - Qt API name
            previous = self._signals_blocked
            self._signals_blocked = blocked
            return previous

        def setChecked(self, checked: bool) -> None:  # noqa: N802 - Qt API name
            self.checked = checked

    toggle = StubToggle()
    controller = GuiController(
        viewer=object(),
        widgets=SimpleNamespace(display=SimpleNamespace(use_display_transform=toggle)),
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module.ImageCms, "get_display_profile", lambda: None)
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.report_display_transform_status(True)

    assert captured["status"] == "Display transform unavailable: no display profile detected, disabled"
    assert toggle.checked is False


def test_sync_display_transform_availability_unchecks_when_profile_missing(monkeypatch) -> None:
    class StubToggle:
        def __init__(self) -> None:
            self.checked = True
            self._signals_blocked = False

        def blockSignals(self, blocked: bool) -> bool:  # noqa: N802 - Qt API name
            previous = self._signals_blocked
            self._signals_blocked = blocked
            return previous

        def setChecked(self, checked: bool) -> None:  # noqa: N802 - Qt API name
            self.checked = checked

    toggle = StubToggle()
    controller = GuiController(
        viewer=object(),
        widgets=SimpleNamespace(display=SimpleNamespace(use_display_transform=toggle)),
    )

    monkeypatch.setattr(controller_module.ImageCms, "get_display_profile", lambda: None)

    available = controller.sync_display_transform_availability(report_status=False)

    assert available is False
    assert toggle.checked is False


def test_prepare_output_display_image_reports_missing_display_profile(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)

    monkeypatch.setattr(controller_module.ImageCms, "get_display_profile", lambda: None)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space="Display P3",
        use_display_transform=True,
    )

    assert preview.dtype == np.uint8
    assert status == "Display transform: no display profile, using raw preview"


def test_prepare_output_display_image_reports_transform_failure(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    image_data = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)

    monkeypatch.setattr(controller_module.ImageCms, "get_display_profile", lambda: object())
    monkeypatch.setattr(controller_module.colour, "RGB_to_RGB", lambda *args, **kwargs: np.full((1, 1, 3), 0.5, dtype=np.float32))
    monkeypatch.setattr(controller_module.ImageCms, "createProfile", lambda name: f"profile:{name}")
    monkeypatch.setattr(controller_module.PILImage, "fromarray", lambda array, mode='RGB': object())

    def raise_transform_error(*args, **kwargs):
        raise controller_module.ImageCms.PyCMSError("bad transform")

    monkeypatch.setattr(controller_module.ImageCms, "profileToProfile", raise_transform_error)

    preview, status = controller._prepare_output_display_image(
        image_data,
        output_color_space="Display P3",
        use_display_transform=True,
    )

    assert preview.dtype == np.uint8
    assert status == "Display transform: transform failed, using raw preview"


def test_save_current_as_default_persists_collected_state(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, "save_default_gui_state", lambda state: captured.setdefault("state", state))
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.save_current_as_default()

    assert captured["state"] is gui_state
    assert captured["status"] == "Saved current GUI state as the startup default"


def test_save_current_state_to_file_persists_json(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "dialog_parent", lambda viewer: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: ("gui_state.json", "JSON (*.json)")),
    )
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, "save_gui_state_to_path", lambda state, path: captured.setdefault("saved", (state, path)))
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.save_current_state_to_file()

    assert captured["saved"] == (gui_state, "gui_state.json")
    assert captured["status"] == "Saved GUI state to gui_state.json"


def test_load_state_from_file_applies_loaded_state(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    gui_state = make_controller_gui_state()
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "dialog_parent", lambda viewer: None)
    monkeypatch.setattr(
        controller_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: ("gui_state.json", "JSON (*.json)")),
    )
    monkeypatch.setattr(controller_module, "load_gui_state_from_path", lambda path: gui_state)
    monkeypatch.setattr(controller_module, "apply_gui_state", lambda state, *, widgets: captured.setdefault("applied", (state, widgets)))
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.load_state_from_file()

    assert captured["applied"] == (gui_state, controller._widgets)
    assert captured["status"] == "Loaded GUI state from gui_state.json"


def test_restore_factory_default_clears_saved_default_and_applies_factory_state(monkeypatch) -> None:
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module, "clear_saved_default_gui_state", lambda: captured.setdefault("cleared", True))
    monkeypatch.setattr(controller_module, "apply_gui_state", lambda state, *, widgets: captured.setdefault("applied", (state, widgets)))
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.restore_factory_default()

    assert captured["cleared"] is True
    assert captured["applied"] == (PROJECT_DEFAULT_GUI_STATE, controller._widgets)
    assert captured["status"] == "Restored factory default GUI state"