"""Profile GUI preview responsiveness.

Companion to ``runtime_process_timing.py`` (one full-res run) and
``runtime_process_memory.py`` (leak/peak memory). This one answers the
question the user actually feels: "when I drag a slider with auto-preview
on, how long until the image updates, and where does that time go?"

It mirrors the controller's preview hot path
(``_process_image_with_runtime``): a single *persistent* ``Simulator`` on a
fixed, preview-sized image, with ``update_params`` + ``process`` called every
iteration — exactly what happens on each slider move. Unlike a scan, the
preview runs ``preview_mode=True`` (grain / lens-blur / glare-blur / unsharp
zeroed in ``digest_params``) on an image downsampled to ``preview_max_size``.

Two regimes are reported separately:
  * cold first refresh — includes the one-time LUT bakes (paid once when the
    image or profile changes, not on every slider move)
  * steady state — the per-edit latency the user feels dragging a slider,
    summarized as mean / median / p95 and an effective updates/sec

Per refresh it splits the cost into ``update_params`` (pipeline rebuild) vs
``process`` (the actual render), and prints a representative per-stage
breakdown.

Usage:
    python scripts/preview_performance.py                  # 40 refreshes, 640 px
    python scripts/preview_performance.py -n 100
    python scripts/preview_performance.py --preview-size 1024
    python scripts/preview_performance.py --soft            # soft_update path
"""
from __future__ import annotations

import argparse
from statistics import mean, median
from time import perf_counter

import numpy as np

from spektrafilm.utils.io import load_image_oiio
from spektrafilm.utils.numba_warmup import warmup
from spektrafilm.utils.preview import resize_for_preview
from spektrafilm.runtime import init_params, digest_params
from spektrafilm.runtime.process import Simulator


IMAGE_PATH = "img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif"


def build_base_params():
    """Same stock/render configuration as runtime_process_timing.py.

    Edit this directly to test a different stock / gamut / setting.
    """
    params = init_params(print_profile="kodak_portra_endura")
    params.io.input_cctf_decoding = False
    params.print_render.glare.active = True
    params.debug.deactivate_stochastic_effects = False
    params.camera.auto_exposure = True
    params.io.upscale_factor = 1.0
    params.io.scan_film = False
    params.film_render.grain.active = True
    params.film_render.grain.particle_area_um2 = 1
    params.enlarger.print_exposure_compensation = True
    params.enlarger.print_exposure = 1.0
    params.settings.use_fast_stats = True
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 17
    # The GUI sets preview_mode for the preview layer; digest_params then zeroes
    # the expensive spatial/stochastic effects that the preview omits.
    params.settings.preview_mode = True
    return params


def pct(values, p):
    """Simple percentile (nearest-rank) over a list of floats."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--iterations", type=int, default=40,
                    help="number of preview refreshes to simulate")
    ap.add_argument("--warmup-iters", type=int, default=3,
                    help="refreshes to discard before steady-state stats "
                         "(absorbs the cold LUT bake and JIT settle)")
    ap.add_argument("--preview-size", type=int, default=None,
                    help="override preview_max_size (default: the params value)")
    ap.add_argument("--soft", action="store_true",
                    help="use soft_update (cheap exposure-only path) instead of "
                         "the full update_params rebuild")
    args = ap.parse_args()

    print("Warming up numba / libraries...")
    warmup()

    base = build_base_params()
    preview_size = args.preview_size or base.settings.preview_max_size

    full = np.double(load_image_oiio(IMAGE_PATH))
    image = resize_for_preview(full, preview_size)
    del full
    print(f"Preview image: {image.shape}  (max edge {preview_size}px, preview_mode=True)")
    print(f"Refreshes: {args.iterations}   update path: "
          f"{'soft_update' if args.soft else 'update_params (full rebuild)'}\n")

    simulator = Simulator(digest_params(build_base_params()))

    update_ms: list[float] = []
    process_ms: list[float] = []
    total_ms: list[float] = []
    rep_timings = None  # representative per-stage breakdown (a steady-state iter)

    for i in range(args.iterations):
        # Each refresh = one "slider move". Nudge a cheap param so the update is
        # a real refresh, not a no-op.
        exposure = 1.0 + 0.002 * (i % 11)

        t0 = perf_counter()
        if args.soft:
            simulator.soft_update(print_exposure=exposure)
        else:
            params = build_base_params()
            params.enlarger.print_exposure = exposure
            params.camera.exposure_compensation_ev = 0.01 * (i % 5)
            # apply_stocks_specifics=False mirrors the controller after the first
            # build: a slider move does not re-derive stock specifics.
            digested = digest_params(params, apply_stocks_specifics=False)
            simulator.update_params(digested)
        t1 = perf_counter()
        result = simulator.process(image)
        t2 = perf_counter()

        u = (t1 - t0) * 1000.0
        p = (t2 - t1) * 1000.0
        update_ms.append(u)
        process_ms.append(p)
        total_ms.append(u + p)
        if i == max(args.warmup_iters, args.iterations - 1):
            rep_timings = simulator.format_timings()

        tag = "  (cold)" if i == 0 else ""
        print(f"  refresh {i:3d}   update {u:7.1f} ms   process {p:7.1f} ms   "
              f"total {u + p:7.1f} ms{tag}")
        del result

    # ---- steady-state summary ----------------------------------------------
    w = min(args.warmup_iters, max(0, args.iterations - 2))
    st_total = total_ms[w:]
    st_update = update_ms[w:]
    st_process = process_ms[w:]

    print("\n" + "=" * 64)
    print(f"COLD FIRST REFRESH: {total_ms[0]:.1f} ms "
          f"(update {update_ms[0]:.1f} + process {process_ms[0]:.1f}) "
          f"— one-time LUT bake, paid on image/profile change only")
    print("=" * 64)
    print(f"STEADY STATE (refreshes {w}..{args.iterations - 1}, the per-slider latency)")
    print("=" * 64)
    if st_total:
        fps = 1000.0 / mean(st_total) if mean(st_total) > 0 else float("inf")
        print(f"  total    mean {mean(st_total):7.1f}  median {median(st_total):7.1f}  "
              f"p95 {pct(st_total, 95):7.1f}  min {min(st_total):7.1f} ms")
        print(f"  update   mean {mean(st_update):7.1f}  median {median(st_update):7.1f} ms"
              f"   ({100 * mean(st_update) / mean(st_total):.0f}% of total)")
        print(f"  process  mean {mean(st_process):7.1f}  median {median(st_process):7.1f} ms"
              f"   ({100 * mean(st_process) / mean(st_total):.0f}% of total)")
        print(f"  -> {fps:.1f} preview updates/sec")
        if mean(st_total) > 100:
            print("  note: >100 ms/refresh feels laggy when dragging a slider.")

    if rep_timings:
        print("\n" + "=" * 64)
        print("REPRESENTATIVE PER-STAGE BREAKDOWN (one steady-state refresh)")
        print("=" * 64)
        print(rep_timings)


if __name__ == "__main__":
    main()
