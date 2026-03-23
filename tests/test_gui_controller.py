from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from spectral_film_lab.gui import controller as controller_module
from spectral_film_lab.gui.controller import (
    GuiController,
    OUTPUT_CCTF_ENCODING_KEY,
    OUTPUT_COLOR_SPACE_KEY,
    OUTPUT_DISPLAY_TRANSFORM_KEY,
    OUTPUT_FLOAT_DATA_KEY,
)
from spectral_film_lab.gui.state import PROJECT_DEFAULT_GUI_STATE


class FakeLayer:
    def __init__(self, data: np.ndarray, metadata: dict[str, object] | None = None):
        self.data = data
        self.metadata = metadata or {}


def make_gui_state():
    state = deepcopy(PROJECT_DEFAULT_GUI_STATE)
    state.simulation.output_color_space = "ACES2065-1"
    state.simulation.saving_color_space = "Display P3"
    state.simulation.saving_cctf_encoding = False
    state.simulation.use_display_transform = False
    return state


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
    gui_state = make_gui_state()

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
    gui_state = make_gui_state()
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
    gui_state = make_gui_state()
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
    gui_state = make_gui_state()
    gui_state.simulation.use_display_transform = True
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller, "_selected_input_layer", lambda: input_layer)
    monkeypatch.setattr(controller_module, "collect_gui_state", lambda *, widgets: gui_state)
    monkeypatch.setattr(controller_module, "build_params_from_state", lambda state: object())
    monkeypatch.setattr(controller_module, "photo_process", lambda image, params: np.full((2, 2, 3), 0.5, dtype=np.float32))

    def fake_prepare_output_display_image(image_data, *, output_color_space, use_display_transform):
        captured["display_args"] = {
            "image_data": image_data.copy(),
            "output_color_space": output_color_space,
            "use_display_transform": use_display_transform,
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
    np.testing.assert_array_equal(captured["output_layer"]["image"], np.full((2, 2, 3), 99, dtype=np.uint8))
    np.testing.assert_allclose(captured["output_layer"]["float_image"], np.full((2, 2, 3), 0.5, dtype=np.float32))


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
    controller = GuiController(viewer=object(), widgets=object())
    captured: dict[str, object] = {}

    monkeypatch.setattr(controller_module.ImageCms, "get_display_profile", lambda: None)
    monkeypatch.setattr(controller_module, "set_status", lambda viewer, message: captured.setdefault("status", message))

    controller.report_display_transform_status(True)

    assert captured["status"] == "Display transform: no display profile, using raw preview"


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