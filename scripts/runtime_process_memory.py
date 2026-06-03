"""Profile runtime memory growth across repeated simulations.

Companion to ``runtime_process_timing.py``. Where that script answers
"where does the time go in one run", this answers "does memory grow when
we run the pipeline over and over" — i.e. is there a leak.

It mirrors the GUI's hot path (``controller._process_image_with_runtime``):
a single *persistent* ``Simulator`` whose ``update_params`` + ``process``
are called every iteration. A leak in the runtime accumulates here; if RSS
stays flat here but the GUI still grows, the leak is in the napari/Qt layer
instead.

Three independent signals, so a leak is hard to miss:
  1. process RSS per iteration (psutil) + a linear fit -> MB/iteration slope
  2. tracemalloc top growing allocations (file:line) between checkpoints
  3. gc object-count growth by type (catches reference cycles / caches)

Usage:
    python scripts/runtime_process_memory.py                 # 40 iters, scan
    python scripts/runtime_process_memory.py -n 100          # more iterations
    python scripts/runtime_process_memory.py --preview       # preview-size path
    python scripts/runtime_process_memory.py --no-mutate     # repeat identical params
    python scripts/runtime_process_memory.py --gamut cam16ucs # stress a heavier tail
"""
from __future__ import annotations

import argparse
import gc
import tracemalloc
from collections import Counter

import numpy as np
import psutil

from spektrafilm.utils.io import load_image_oiio
from spektrafilm.utils.numba_warmup import warmup
from spektrafilm.utils.preview import resize_for_preview
from spektrafilm.runtime import init_params, digest_params
from spektrafilm.runtime.process import Simulator


IMAGE_PATH = "img/test/portrait_leaves_32bit_linear_prophoto_rgb.tif"


def build_base_params(gamut: str | None):
    """Same configuration as runtime_process_timing.py, optionally forcing
    the output gamut-compression algorithm to stress the scanner tail."""
    params = init_params(print_profile="kodak_portra_endura")
    params.io.input_cctf_decoding = False
    params.print_render.glare.active = True
    params.debug.deactivate_stochastic_effects = False
    params.camera.auto_exposure = True
    params.io.upscale_factor = 3.0
    params.io.scan_film = False
    params.film_render.grain.active = True
    params.film_render.grain.particle_area_um2 = 1
    params.enlarger.print_exposure_compensation = True
    params.enlarger.print_exposure = 1.0
    params.settings.use_fast_stats = True
    params.settings.use_enlarger_lut = True
    params.settings.use_scanner_lut = True
    params.settings.lut_resolution = 17
    if gamut is not None:
        params.io.output_gamut_compress.algorithm = gamut
    return params


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def linfit_slope(xs, ys) -> float:
    """Least-squares slope (units of ys per unit x). MB per iteration."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x * (y - y.mean())).sum() / denom) if denom else 0.0


# Exclude the profiler's own bookkeeping (sample lists, tracemalloc, gc
# histogram) so what remains is allocations from the runtime pipeline.
_NOISE_FILTERS = (
    tracemalloc.Filter(False, __file__),
    tracemalloc.Filter(False, tracemalloc.__file__),
    tracemalloc.Filter(False, "<frozen *>"),
)


def top_tracemalloc_growth(snap_a, snap_b, limit=15):
    snap_a = snap_a.filter_traces(_NOISE_FILTERS)
    snap_b = snap_b.filter_traces(_NOISE_FILTERS)
    stats = snap_b.compare_to(snap_a, "lineno")
    return [s for s in stats if s.size_diff > 0][:limit]


def gc_type_histogram() -> Counter:
    counts: Counter = Counter()
    for obj in gc.get_objects():
        counts[type(obj).__name__] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--iterations", type=int, default=40)
    ap.add_argument("--warmup-iters", type=int, default=5,
                    help="iterations to run before taking the memory baseline "
                         "(lets one-time caches/JIT settle)")
    ap.add_argument("--preview", action="store_true",
                    help="run the preview-sized path instead of a full-res scan")
    ap.add_argument("--no-mutate", action="store_true",
                    help="repeat identical params instead of nudging one each iter")
    ap.add_argument("--gamut", default=None,
                    help="force output gamut-compress algorithm (e.g. cam16ucs)")
    ap.add_argument("--checkpoints", type=int, default=4,
                    help="number of tracemalloc compare checkpoints")
    args = ap.parse_args()

    # Keep at least two post-warmup samples so the trend/report is defined,
    # even for tiny runs like `-n 4` (which is below the default warmup of 5).
    warmup_iters = min(args.warmup_iters, max(0, args.iterations - 2))
    if warmup_iters != args.warmup_iters:
        print(f"(warmup window clamped {args.warmup_iters} -> {warmup_iters} "
              f"to fit {args.iterations} iterations)")

    print("Warming up numba / libraries...")
    warmup()

    image = np.double(load_image_oiio(IMAGE_PATH))
    base = build_base_params(args.gamut)
    if args.preview:
        image = resize_for_preview(image, base.settings.preview_max_size)
    print(f"Image shape: {image.shape}  ({'preview' if args.preview else 'scan'} path)")
    print(f"Output gamut algorithm: {base.io.output_gamut_compress.algorithm}")
    print(f"Iterations: {args.iterations}  mutate params: {not args.no_mutate}\n")

    # Persistent simulator, exactly like the GUI keeps self._runtime_simulator.
    simulator = Simulator(digest_params(build_base_params(args.gamut)))

    tracemalloc.start(25)
    gc.collect()

    samples_iter: list[int] = []
    samples_rss: list[float] = []
    snap_baseline = None
    gc_baseline = None
    checkpoint_every = max(1, args.iterations // max(1, args.checkpoints))
    checkpoint_snaps = []  # (iter, snapshot)

    for i in range(args.iterations):
        params = build_base_params(args.gamut)
        if not args.no_mutate:
            # Nudge a param so update_params takes the full rebuild path the
            # GUI hits on every slider move (not a no-op soft path).
            params.enlarger.print_exposure = 1.0 + 0.001 * (i % 7)
            params.camera.exposure_compensation_ev = 0.01 * (i % 5)
        digested = digest_params(params)
        simulator.update_params(digested)
        result = simulator.process(image)

        # Drop our own references so anything still alive is the pipeline's.
        del params, digested, result
        gc.collect()

        rss = rss_mb()
        samples_iter.append(i)
        samples_rss.append(rss)

        # Baseline after warmup window; measure growth only past that point.
        if i == warmup_iters:
            snap_baseline = tracemalloc.take_snapshot()
            gc_baseline = gc_type_histogram()

        marker = ""
        if i >= warmup_iters and (i - warmup_iters) % checkpoint_every == 0:
            checkpoint_snaps.append((i, tracemalloc.take_snapshot()))
            marker = "  <- checkpoint"
        print(f"  iter {i:3d}   RSS {rss:8.1f} MB{marker}")

    # ---- RSS trend (post-warmup) -------------------------------------------
    post = [(i, m) for i, m in zip(samples_iter, samples_rss) if i >= warmup_iters]
    pi = [i for i, _ in post]
    pm = [m for _, m in post]
    slope = linfit_slope(pi, pm)
    total_growth = pm[-1] - pm[0] if pm else 0.0

    print("\n" + "=" * 64)
    print("RSS TREND (after warmup window)")
    print("=" * 64)
    if not pm:
        print("  not enough post-warmup samples to compute a trend; "
              "increase -n or lower --warmup-iters")
        tracemalloc.stop()
        return
    print(f"  first {pm[0]:8.1f} MB   last {pm[-1]:8.1f} MB   delta {total_growth:+8.1f} MB")
    print(f"  slope {slope:+.3f} MB / iteration "
          f"({slope * 1000:+.1f} MB per 1000 scans)")
    verdict = "LEAK LIKELY" if slope > 0.5 else ("possible drift" if slope > 0.1 else "flat / no runtime leak")
    print(f"  verdict: {verdict}")

    # ---- tracemalloc top growth --------------------------------------------
    if snap_baseline is not None and checkpoint_snaps:
        last_snap = checkpoint_snaps[-1][1]
        print("\n" + "=" * 64)
        print("TOP GROWING ALLOCATIONS (tracemalloc, baseline -> last checkpoint)")
        print("=" * 64)
        for s in top_tracemalloc_growth(snap_baseline, last_snap):
            frame = s.traceback[0]
            print(f"  {s.size_diff/1024:+9.1f} KB  (count {s.count_diff:+6d})  "
                  f"{frame.filename}:{frame.lineno}")

    # ---- gc object-type growth ---------------------------------------------
    if gc_baseline is not None:
        gc_now = gc_type_histogram()
        print("\n" + "=" * 64)
        print("TOP OBJECT-COUNT GROWTH BY TYPE (gc)")
        print("=" * 64)
        growth = sorted(
            ((gc_now[k] - gc_baseline.get(k, 0), k) for k in gc_now),
            reverse=True,
        )
        for diff, name in growth[:15]:
            if diff <= 0:
                break
            print(f"  {diff:+8d}  {name}")

    # ---- uncollectable / cycles --------------------------------------------
    gc.collect()
    if gc.garbage:
        print(f"\n  WARNING: gc.garbage holds {len(gc.garbage)} uncollectable objects")

    tracemalloc.stop()
    print("\nDone. If the slope is flat here but the GUI still grows, the leak "
          "is in the napari/Qt layer, not the runtime pipeline.")


if __name__ == "__main__":
    main()
