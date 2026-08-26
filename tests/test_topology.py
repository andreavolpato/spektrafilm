"""Unit tests for the topology primitives (Node, Tap, run_topology).

These tests are independent of the spektrafilm pipeline and cover the
dispatcher's invariants in isolation.
"""

from __future__ import annotations

import pytest

from spektrafilm.runtime.topology import Node, Tap, run_topology


class TestNode:
    def test_single_write(self):
        n = Node(reads=("a",), writes=("b",), run=lambda x: x * 2)
        state = {"a": 3}
        n.fire(state)
        assert state["b"] == 6

    def test_multi_read_single_write(self):
        n = Node(reads=("a", "b"), writes=("c",), run=lambda x, y: x + y)
        state = {"a": 2, "b": 5}
        n.fire(state)
        assert state["c"] == 7

    def test_multi_write(self):
        n = Node(reads=("a",), writes=("b", "c"), run=lambda x: (x, x * 2))
        state = {"a": 3}
        n.fire(state)
        assert state == {"a": 3, "b": 3, "c": 6}

    def test_tuple_into_single_write_asserts(self):
        n = Node(reads=("a",), writes=("b",), run=lambda x: (x, x))
        with pytest.raises(AssertionError):
            n.fire({"a": 1})

    def test_multi_write_length_mismatch_asserts(self):
        n = Node(reads=("a",), writes=("b", "c"), run=lambda x: (x,))
        with pytest.raises(AssertionError):
            n.fire({"a": 1})


class TestTap:
    def test_attribute_names_match_lowercase_values(self):
        for attr in (
            "RGB_IN",
            "RGB_PRE",
            "LOG_E_FILM",
            "CMY_FILM",
            "LOG_E_PRINT",
            "CMY_PRINT",
            "RGB_OUT",
        ):
            assert getattr(Tap, attr) == attr.lower()


class TestRunTopology:
    def _linear_chain(self):
        return [
            Node(("a",), ("b",), lambda x: x + 1, "step1"),
            Node(("b",), ("c",), lambda x: x * 2, "step2"),
            Node(("c",), ("d",), lambda x: x - 3, "step3"),
        ]

    def test_full_run_returns_terminal_tap(self):
        result = run_topology(self._linear_chain(), inject="a", collect="d", image=10)
        assert result == ((10 + 1) * 2) - 3

    def test_intermediate_collect_stops_early(self):
        fired = []
        result = run_topology(
            self._linear_chain(),
            inject="a",
            collect="c",
            image=10,
            on_fire=lambda node, elapsed: fired.append(node.label),
        )
        assert result == (10 + 1) * 2
        assert fired == ["step1", "step2"]

    def test_inject_at_intermediate_skips_upstream_nodes(self):
        fired = []
        result = run_topology(
            self._linear_chain(),
            inject="b",
            collect="d",
            image=100,
            on_fire=lambda node, elapsed: fired.append(node.label),
        )
        assert result == (100 * 2) - 3
        assert fired == ["step2", "step3"]

    def test_unreachable_collect_raises(self):
        with pytest.raises(RuntimeError, match="no node path reaches"):
            run_topology(self._linear_chain(), inject="a", collect="z", image=10)

    def test_multi_input_fan_in(self):
        """A node that reads two taps waits until both producers have fired."""
        topology = [
            Node(("a",), ("x",), lambda v: v + 1, "produce_x"),
            Node(("a",), ("y",), lambda v: v * 10, "produce_y"),
            Node(("x", "y"), ("z",), lambda x, y: x + y, "combine"),
        ]
        fired = []
        result = run_topology(
            topology,
            inject="a",
            collect="z",
            image=5,
            on_fire=lambda node, elapsed: fired.append(node.label),
        )
        assert result == (5 + 1) + (5 * 10)
        assert fired == ["produce_x", "produce_y", "combine"]
