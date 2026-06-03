from __future__ import annotations

from spektrafilm_gui.background import BackgroundRunner


class FakePool:
    """A thread pool stand-in that queues tasks instead of running them, so a
    test can control exactly when each one completes."""

    def __init__(self) -> None:
        self.tasks: list[object] = []

    def start(self, task) -> None:
        self.tasks.append(task)

    def run_next(self) -> None:
        # Tasks emit on the calling (test) thread, so delivery is synchronous.
        self.tasks.pop(0).run()

    def __len__(self) -> int:
        return len(self.tasks)


def test_submit_runs_work_and_delivers_result() -> None:
    pool = FakePool()
    runner = BackgroundRunner(pool)
    results: list[object] = []

    runner.submit('c', lambda: 42, on_done=results.append)
    assert runner.is_busy('c')

    pool.run_next()

    assert results == [42]
    assert not runner.is_busy('c')


def test_error_is_reported_with_type_prefix_and_frees_channel() -> None:
    pool = FakePool()
    runner = BackgroundRunner(pool)
    errors: list[str] = []

    def boom():
        raise ValueError('bad')

    runner.submit('c', boom, on_done=lambda result: None, on_error=errors.append)
    pool.run_next()

    assert errors == ['ValueError: bad']
    assert not runner.is_busy('c')


def test_busy_channel_coalesces_to_latest_pending() -> None:
    pool = FakePool()
    runner = BackgroundRunner(pool)
    done: list[object] = []

    runner.submit('c', lambda: 'first', on_done=done.append)
    # While 'first' is in flight, two more arrive; only the latest is kept.
    runner.submit('c', lambda: 'stale', on_done=done.append)
    runner.submit('c', lambda: 'latest', on_done=done.append)
    assert len(pool) == 1  # only 'first' is queued; the rest collapse to pending

    pool.run_next()  # 'first' completes, then the pending 'latest' is started
    assert done == ['first']
    assert len(pool) == 1

    pool.run_next()
    assert done == ['first', 'latest']  # 'stale' never ran
    assert not runner.is_busy('c')


def test_channels_run_independently() -> None:
    pool = FakePool()
    runner = BackgroundRunner(pool)
    out: list[object] = []

    runner.submit('a', lambda: 'a', on_done=out.append)
    runner.submit('b', lambda: 'b', on_done=out.append)
    assert len(pool) == 2  # different channels do not coalesce

    pool.run_next()
    pool.run_next()
    assert sorted(out) == ['a', 'b']
