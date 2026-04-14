from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import colour
import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms

from spektrafilm.runtime.api import Simulator, digest_params
from spektrafilm.utils.io import load_image_oiio
from spektrafilm.utils.preview import resize_for_preview
from spektrafilm_gui import controller_layers as controller_layers_module
from spektrafilm_gui import controller_runtime
from spektrafilm_gui.controller import (
    OUTPUT_CCTF_ENCODING_KEY,
    OUTPUT_COLOR_SPACE_KEY,
    OUTPUT_DISPLAY_TRANSFORM_KEY,
    OUTPUT_FLOAT_DATA_KEY,
)
from spektrafilm_gui.controller_layers import (
    OUTPUT_LAYER_ANIMATION_DURATION_MS,
    OUTPUT_LAYER_ANIMATION_INTERVAL_MS,
    ViewerLayerService,
)
from spektrafilm_gui.params_mapper import build_params_from_state
from spektrafilm_gui.persistence import load_default_gui_state
from spektrafilm_gui.polaroid_animation import prepare_polaroid_state, render_polaroid_frame


DEFAULT_IMAGE_PATH = Path("img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif")


def _mean_ms(samples: list[float]) -> float:
    return float(statistics.mean(samples)) if samples else 0.0


def _build_frame_times() -> np.ndarray:
    frame_count = max(
        int(np.ceil(float(OUTPUT_LAYER_ANIMATION_DURATION_MS) / float(max(OUTPUT_LAYER_ANIMATION_INTERVAL_MS, 1)))),
        2,
    )
    return np.linspace(0.0, 1.0, num=frame_count, dtype=np.float32)


def _warm_preview_runtime(
    preview_source: np.ndarray,
    state: Any,
    *,
    use_display_transform: bool,
) -> tuple[Simulator, np.ndarray, np.ndarray]:
    controller_runtime.prepare_input_color_preview_image(
        preview_source,
        input_color_space=state.input_image.input_color_space,
        apply_cctf_decoding=state.input_image.apply_cctf_decoding,
        colour_module=colour,
    )

    params = build_params_from_state(state)
    params.settings.preview_mode = True
    simulator = Simulator(digest_params(params, apply_stocks_specifics=True))
    scan = np.asarray(simulator.process(preview_source), dtype=np.float32)
    output_display, _ = controller_runtime.prepare_output_display_image(
        scan,
        output_color_space=state.simulation.output_color_space,
        use_display_transform=use_display_transform,
        imagecms_module=ImageCms,
        colour_module=colour,
        pil_image_module=PILImage,
    )
    polaroid_state = prepare_polaroid_state(output_display)
    render_polaroid_frame(polaroid_state, 0.0)
    return simulator, scan, output_display


def benchmark_warm_import_path(
    image_path: Path,
    *,
    iterations: int,
    use_display_transform: bool,
) -> dict[str, float | tuple[int, ...] | int]:
    state = load_default_gui_state()
    image = load_image_oiio(str(image_path))[..., :3]
    preview_source = resize_for_preview(image, state.display.preview_max_size)
    simulator, scan, output_display = _warm_preview_runtime(
        preview_source,
        state,
        use_display_transform=use_display_transform,
    )

    frame_times = _build_frame_times()
    timings: dict[str, list[float]] = defaultdict(list)

    for _ in range(iterations):
        start = time.perf_counter()
        image = load_image_oiio(str(image_path))[..., :3]
        timings["load_image_oiio_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        preview_source = resize_for_preview(image, state.display.preview_max_size)
        timings["resize_for_preview_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        controller_runtime.prepare_input_color_preview_image(
            preview_source,
            input_color_space=state.input_image.input_color_space,
            apply_cctf_decoding=state.input_image.apply_cctf_decoding,
            colour_module=colour,
        )
        timings["prepare_input_preview_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        params = build_params_from_state(state)
        params.settings.preview_mode = True
        timings["build_params_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        digested = digest_params(params, apply_stocks_specifics=False)
        timings["digest_params_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        simulator.update_params(digested)
        scan = np.asarray(simulator.process(preview_source), dtype=np.float32)
        timings["simulator_update_process_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        output_display, _ = controller_runtime.prepare_output_display_image(
            scan,
            output_color_space=state.simulation.output_color_space,
            use_display_transform=use_display_transform,
            imagecms_module=ImageCms,
            colour_module=colour,
            pil_image_module=PILImage,
        )
        timings["prepare_output_display_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        polaroid_state = prepare_polaroid_state(output_display)
        timings["prepare_polaroid_state_ms"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        render_polaroid_frame(polaroid_state, float(frame_times[0]))
        timings["first_polaroid_frame_ms"].append((time.perf_counter() - start) * 1000.0)

        frame_samples: list[float] = []
        all_frames_start = time.perf_counter()
        for frame_t in frame_times:
            frame_start = time.perf_counter()
            render_polaroid_frame(polaroid_state, float(frame_t))
            frame_samples.append((time.perf_counter() - frame_start) * 1000.0)
        timings["all_polaroid_frames_ms"].append((time.perf_counter() - all_frames_start) * 1000.0)
        timings["per_polaroid_frame_avg_ms"].append(float(statistics.mean(frame_samples)))
        timings["per_polaroid_frame_max_ms"].append(max(frame_samples))

    summary: dict[str, float | tuple[int, ...] | int] = {
        key: round(_mean_ms(values), 3)
        for key, values in timings.items()
    }
    summary["preview_source_shape"] = tuple(int(v) for v in preview_source.shape)
    summary["output_display_shape"] = tuple(int(v) for v in output_display.shape)
    summary["animation_frame_count"] = int(frame_times.shape[0])
    summary["animation_budget_total_ms"] = int(OUTPUT_LAYER_ANIMATION_DURATION_MS)
    summary["animation_budget_per_frame_ms"] = int(OUTPUT_LAYER_ANIMATION_INTERVAL_MS)
    return summary


def benchmark_napari_layer_service(
    preview_display: np.ndarray,
    output_display: np.ndarray,
    float_image: np.ndarray,
    *,
    iterations: int,
) -> dict[str, float | tuple[int, ...] | str]:
    try:
        import napari
    except Exception as exc:  # pragma: no cover - best effort diagnostic script
        return {"napari_unavailable": str(exc)}

    viewer = napari.Viewer(show=False)
    service = ViewerLayerService(
        viewer=viewer,
        output_float_data_key=OUTPUT_FLOAT_DATA_KEY,
        output_color_space_key=OUTPUT_COLOR_SPACE_KEY,
        output_cctf_encoding_key=OUTPUT_CCTF_ENCODING_KEY,
        output_display_transform_key=OUTPUT_DISPLAY_TRANSFORM_KEY,
    )
    original_max_pixels = controller_layers_module.OUTPUT_LAYER_ANIMATION_MAX_PIXELS
    controller_layers_module.OUTPUT_LAYER_ANIMATION_MAX_PIXELS = 1
    state = load_default_gui_state()
    timings: dict[str, list[float]] = defaultdict(list)

    try:
        for _ in range(iterations):
            for layer_name in ("white_border", "input_preview", "output"):
                service.remove_layer(layer_name)

            start = time.perf_counter()
            service.set_or_add_input_preview_layer(preview_display, white_padding=state.display.white_padding)
            timings["first_input_stack_ms"].append((time.perf_counter() - start) * 1000.0)

            service.set_or_add_output_layer(
                output_display,
                float_image=float_image,
                output_color_space=state.simulation.output_color_space,
                output_cctf_encoding=True,
                use_display_transform=False,
            )

            start = time.perf_counter()
            service.set_or_add_input_preview_layer(preview_display, white_padding=state.display.white_padding)
            timings["repeat_input_stack_ms"].append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            service.set_or_add_output_layer(
                output_display,
                float_image=float_image,
                output_color_space=state.simulation.output_color_space,
                output_cctf_encoding=True,
                use_display_transform=False,
            )
            timings["hidden_output_reshow_ms"].append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            service.set_or_add_output_layer(
                output_display,
                float_image=float_image,
                output_color_space=state.simulation.output_color_space,
                output_cctf_encoding=True,
                use_display_transform=False,
            )
            timings["visible_output_refresh_ms"].append((time.perf_counter() - start) * 1000.0)
    finally:
        controller_layers_module.OUTPUT_LAYER_ANIMATION_MAX_PIXELS = original_max_pixels
        close = getattr(viewer, "close", None)
        if callable(close):
            close()

    summary: dict[str, float | tuple[int, ...] | str] = {
        key: round(_mean_ms(values), 3)
        for key, values in timings.items()
    }
    summary["preview_display_shape"] = tuple(int(v) for v in preview_display.shape)
    summary["output_display_shape"] = tuple(int(v) for v in output_display.shape)
    return summary


def print_report(core: dict[str, Any], napari_summary: dict[str, Any]) -> None:
    import_to_white_border_ms = round(
        float(core["load_image_oiio_ms"])
        + float(core["resize_for_preview_ms"])
        + float(core["prepare_input_preview_ms"])
        + float(napari_summary.get("repeat_input_stack_ms", 0.0)),
        3,
    )
    white_border_to_animation_start_ms = round(
        float(core["build_params_ms"])
        + float(core["digest_params_ms"])
        + float(core["simulator_update_process_ms"])
        + float(core["prepare_output_display_ms"])
        + float(napari_summary.get("hidden_output_reshow_ms", 0.0))
        + float(core["prepare_polaroid_state_ms"])
        + float(core["first_polaroid_frame_ms"]),
        3,
    )
    import_to_animation_end_ms = round(
        import_to_white_border_ms
        + white_border_to_animation_start_ms
        + max(float(core["animation_budget_total_ms"]), float(core["all_polaroid_frames_ms"])),
        3,
    )

    print("GUI import -> animation timing profile (warm path)")
    print(f"Preview shape: {core['preview_source_shape']}")
    print(f"Output display shape: {core['output_display_shape']}")
    print()
    print("Core phases (ms):")
    for key in (
        "load_image_oiio_ms",
        "resize_for_preview_ms",
        "prepare_input_preview_ms",
        "build_params_ms",
        "digest_params_ms",
        "simulator_update_process_ms",
        "prepare_output_display_ms",
        "prepare_polaroid_state_ms",
        "first_polaroid_frame_ms",
        "all_polaroid_frames_ms",
        "per_polaroid_frame_avg_ms",
        "per_polaroid_frame_max_ms",
    ):
        print(f"  {key}: {core[key]}")
    print()
    print("Napari layer phases (ms):")
    for key in (
        "first_input_stack_ms",
        "repeat_input_stack_ms",
        "hidden_output_reshow_ms",
        "visible_output_refresh_ms",
    ):
        value = napari_summary.get(key)
        if value is not None:
            print(f"  {key}: {value}")
    if "napari_unavailable" in napari_summary:
        print(f"  napari_unavailable: {napari_summary['napari_unavailable']}")
    print()
    print("Derived totals (ms):")
    print(f"  import_to_white_border_ms: {import_to_white_border_ms}")
    print(f"  white_border_to_animation_start_ms: {white_border_to_animation_start_ms}")
    print(f"  import_to_animation_end_ms_estimate: {import_to_animation_end_ms}")
    print(f"  animation_budget_total_ms: {core['animation_budget_total_ms']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile warm GUI import-to-animation timings.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Image to load for the benchmark.")
    parser.add_argument("--iterations", type=int, default=4, help="Number of warm-path iterations.")
    parser.add_argument(
        "--use-display-transform",
        action="store_true",
        help="Measure output display conversion with display transform enabled.",
    )
    args = parser.parse_args()

    core = benchmark_warm_import_path(
        args.image,
        iterations=max(args.iterations, 1),
        use_display_transform=args.use_display_transform,
    )

    image = load_image_oiio(str(args.image))[..., :3]
    state = load_default_gui_state()
    preview_source = resize_for_preview(image, state.display.preview_max_size)
    preview_display = controller_runtime.prepare_input_color_preview_image(
        preview_source,
        input_color_space=state.input_image.input_color_space,
        apply_cctf_decoding=state.input_image.apply_cctf_decoding,
        colour_module=colour,
    )
    output_display, _ = controller_runtime.prepare_output_display_image(
        np.asarray(preview_source, dtype=np.float32),
        output_color_space=state.simulation.output_color_space,
        use_display_transform=False,
        imagecms_module=ImageCms,
        colour_module=colour,
        pil_image_module=PILImage,
    )
    napari_summary = benchmark_napari_layer_service(
        preview_display,
        output_display,
        np.asarray(preview_source, dtype=np.float32),
        iterations=max(args.iterations, 1),
    )
    print_report(core, napari_summary)


if __name__ == "__main__":
    main()