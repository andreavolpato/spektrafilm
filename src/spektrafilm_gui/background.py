"""A small reusable primitive for running heavy work off the GUI thread.

The GUI thread must stay responsive at all times: every expensive operation
(simulation, RAW decode, ...) runs on a worker thread and reports its result
back on the GUI thread through Qt signals. `BackgroundRunner` is the single
mechanism the controller uses for all of them.

Two properties make it suitable as the one shared primitive:

* **Per-channel coalescing.** Each independent kind of work runs on its own
  named ``channel`` (e.g. ``'simulation'``, ``'raw'``). A channel holds at most
  one running task plus one pending request; submitting again while busy simply
  replaces the pending request. So a burst of parameter changes never piles up
  on the pool and never lands a stale result -- only the latest survives.

* **Below-GUI priority.** Heavy numpy / OCIO work releases the GIL, so the GUI
  Python thread keeps running while it computes. Running the worker thread just
  below the GUI thread lets the OS scheduler favour redraws and input on
  contended cores, while the computation still spreads across every spare core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from qtpy import QtCore

QObject = getattr(QtCore, 'QObject')
QRunnable = getattr(QtCore, 'QRunnable')
QThread = getattr(QtCore, 'QThread')
QThreadPool = getattr(QtCore, 'QThreadPool')
Signal = getattr(QtCore, 'Signal')


@dataclass(slots=True)
class _Request:
    work: Callable[[], Any]
    on_done: Callable[[Any], None]
    on_error: Callable[[str], None] | None


class _RunnerSignals(QObject):
    # Channel name routes the result back to the request that started it. Only
    # one task per channel ever runs, so the name alone identifies it.
    done = Signal(str, object)
    error = Signal(str, str)


class _Task(QRunnable):
    def __init__(self, channel: str, work: Callable[[], Any], signals: _RunnerSignals) -> None:
        super().__init__()
        self._channel = channel
        self._work = work
        self._signals = signals

    def run(self) -> None:
        thread = QThread.currentThread()
        if thread is not None and thread.isRunning():
            # Stay below the GUI thread so the UI wins core contention; the
            # computation still uses every otherwise-idle core. Guarded by
            # isRunning() so a synchronous (non-pool) call is a no-op rather
            # than a Qt warning.
            thread.setPriority(QThread.LowPriority)
        try:
            result = self._work()
        except Exception as exc:  # report to the GUI thread; never wedge the channel
            self._signals.error.emit(self._channel, f'{type(exc).__name__}: {exc}')
            return
        self._signals.done.emit(self._channel, result)


class BackgroundRunner:
    """Runs callables off the GUI thread and delivers their result back on it.

    Drive many independent kinds of work through one runner by giving each a
    distinct ``channel``. Within a channel, work is coalesced: at most one task
    runs and one request waits, the waiting request always being the most recent
    submission.
    """

    def __init__(
        self,
        thread_pool: Any | None = None,
        *,
        on_busy_changed: Callable[[bool], None] | None = None,
    ) -> None:
        self._pool = thread_pool if thread_pool is not None else QThreadPool.globalInstance()
        self._signals = _RunnerSignals()
        self._signals.done.connect(self._on_done)
        self._signals.error.connect(self._on_error)
        self._active: dict[str, _Request] = {}
        self._pending: dict[str, _Request] = {}
        # Fires (on the GUI thread) when the runner crosses idle <-> busy, i.e.
        # whenever any channel has a task running. Drives the "working thread
        # engaged" UI state, such as graying out the action buttons.
        self._on_busy_changed = on_busy_changed
        self._busy = False

    def submit(
        self,
        channel: str,
        work: Callable[[], Any],
        *,
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        """Run ``work()`` on ``channel``. If the channel is idle it starts now;
        if it is busy this request becomes the channel's pending request,
        replacing any earlier one so only the latest is kept."""
        request = _Request(work=work, on_done=on_done, on_error=on_error)
        if channel in self._active:
            self._pending[channel] = request
            return
        self._start(channel, request)

    def is_busy(self, channel: str) -> bool:
        """True while a task is running or waiting on ``channel``."""
        return channel in self._active or channel in self._pending

    def _start(self, channel: str, request: _Request) -> None:
        self._active[channel] = request
        self._refresh_busy()
        self._pool.start(_Task(channel, request.work, self._signals))

    def _on_done(self, channel: str, result: Any) -> None:
        request = self._active.pop(channel, None)
        if request is not None and request.on_done is not None:
            request.on_done(result)
        self._drain(channel)
        self._refresh_busy()

    def _on_error(self, channel: str, message: str) -> None:
        request = self._active.pop(channel, None)
        if request is not None and request.on_error is not None:
            request.on_error(message)
        self._drain(channel)
        self._refresh_busy()

    def _drain(self, channel: str) -> None:
        request = self._pending.pop(channel, None)
        if request is not None:
            self._start(channel, request)

    def _refresh_busy(self) -> None:
        busy = bool(self._active)
        if busy != self._busy:
            self._busy = busy
            if self._on_busy_changed is not None:
                self._on_busy_changed(busy)
