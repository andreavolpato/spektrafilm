"""Pipeline topology primitives: nodes, taps, and a state-dict dispatcher.

See studies/a40_lut_system/n020_dispatcher_design.md for the design rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any


class Tap:
    """Canonical tap names for the spektrafilm pipeline.

    Tap names identify *data* at named pipeline boundaries. The UPPER
    attribute name follows PEP 8 for class-level constants; the lowercase
    string value is the wire identifier used in state dicts.
    """

    RGB_IN = "rgb_in"
    RGB_PRE = "rgb_pre"
    LOG_E_FILM = "log_e_film"
    CMY_FILM = "cmy_film"
    LOG_E_PRINT = "log_e_print"
    CMY_PRINT = "cmy_print"
    RGB_OUT = "rgb_out"


@dataclass(frozen=True)
class Node:
    """One unit of computation in the pipeline topology.

    A node declares which taps it reads, which it writes, and the callable
    that performs the transform. Single-write nodes have ``run`` return a
    value; multi-write nodes return a tuple aligned with ``writes``.
    """

    reads: tuple[str, ...]
    writes: tuple[str, ...]
    run: Callable[..., Any]
    label: str = ""

    def fire(self, state: dict[str, Any]) -> None:
        args = tuple(state[k] for k in self.reads)
        result = self.run(*args)
        if len(self.writes) == 1:
            assert not isinstance(result, tuple), (
                f"node {self.label!r} declares one write but run() returned a tuple"
            )
            state[self.writes[0]] = result
        else:
            assert isinstance(result, tuple) and len(result) == len(self.writes), (
                f"node {self.label!r} declares {len(self.writes)} writes but "
                f"run() returned {type(result).__name__} "
                f"of length {len(result) if hasattr(result, '__len__') else '?'}"
            )
            for k, v in zip(self.writes, result):
                state[k] = v


def run_topology(
    topology: list[Node],
    inject: str,
    collect: str,
    image: Any,
    *,
    on_fire: Callable[[Node, float], None] | None = None,
) -> Any:
    """Walk ``topology`` in declared order, firing every node whose reads
    are satisfied. Stops as soon as ``collect`` is in state and returns it.

    ``on_fire`` is invoked as ``on_fire(node, elapsed_seconds)`` after each
    node fires; the dispatcher in :class:`SimulationPipeline` uses it to
    record per-node timings.
    """
    state: dict[str, Any] = {inject: image}
    for node in topology:
        if all(k in state for k in node.reads):
            t0 = perf_counter()
            node.fire(state)
            elapsed = perf_counter() - t0
            if on_fire is not None:
                on_fire(node, elapsed)
            if collect in state:
                return state[collect]
    raise RuntimeError(f"no node path reaches tap {collect!r} from {inject!r}")
