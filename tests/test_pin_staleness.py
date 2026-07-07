"""Tests for staleness pinning in BatchTaskDispatcher.active_submit_and_wait."""

import threading
from dataclasses import dataclass

from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import BatchTaskDispatcher
from areal.utils import logging

logger = logging.getLogger("TestPinStaleness")


class FakeVersionProvider:
    def __init__(self):
        self.version = 0

    def get_version(self) -> int:
        return self.version


@dataclass
class _TaskInput:
    task_id: int


def _input_generator():
    i = 0
    while True:
        yield _TaskInput(task_id=i)
        i += 1


def _make_dispatcher(
    pin_staleness: bool,
    max_staleness: int,
    consumer_batch_size: int,
    max_concurrent_rollouts: int = 64,
):
    manager = StalenessManager(
        version_provider=FakeVersionProvider(),
        max_concurrent_rollouts=max_concurrent_rollouts,
        consumer_batch_size=consumer_batch_size,
        max_staleness=max_staleness,
    )

    def task_factory(task_input: _TaskInput):
        async def run():
            # Mirror _execute_workflow: acceptance is accounted inside the task.
            manager.on_rollout_accepted()
            return {"task_id": task_input.task_id}

        return run

    dispatcher = BatchTaskDispatcher(
        max_queue_size=1024,
        task_factory=task_factory,
        staleness_manager=manager,
        pin_staleness=pin_staleness,
    )
    dispatcher.initialize(logger=logger)
    return dispatcher


def _run_with_timeout(fn, timeout: float = 60.0):
    out = {}

    def target():
        try:
            out["result"] = fn()
        except Exception as exc:  # noqa: BLE001
            out["exception"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), (
        "active_submit_and_wait did not return within timeout (possible deadlock)"
    )
    if "exception" in out:
        raise out["exception"]
    return out["result"]


def test_pin_staleness_holds_batch_until_backlog_floor():
    # floor = max_staleness * batch_size = 6;
    # capacity ceiling at version 0 = (max_staleness + 1) * consumer_bs = 8.
    dispatcher = _make_dispatcher(
        pin_staleness=True, max_staleness=3, consumer_batch_size=2
    )
    try:
        results = _run_with_timeout(
            lambda: dispatcher.active_submit_and_wait(_input_generator(), batch_size=2)
        )
        assert len(results) == 2
        assert dispatcher.get_result_backlog() >= 6
    finally:
        dispatcher.destroy()


def test_pin_staleness_unreachable_floor_does_not_deadlock():
    # Ceiling allows only (3 + 1) * 1 = 4 samples at version 0, but
    # floor = 3 * 4 = 12. The deadlock guard must yield the batch.
    dispatcher = _make_dispatcher(
        pin_staleness=True, max_staleness=3, consumer_batch_size=1
    )
    try:
        results = _run_with_timeout(
            lambda: dispatcher.active_submit_and_wait(_input_generator(), batch_size=4)
        )
        assert len(results) == 4
    finally:
        dispatcher.destroy()


def test_pin_staleness_noop_when_max_staleness_low():
    # max_staleness = 0 gives floor 0; behaves like pinning disabled.
    dispatcher = _make_dispatcher(
        pin_staleness=True, max_staleness=0, consumer_batch_size=2
    )
    try:
        results = _run_with_timeout(
            lambda: dispatcher.active_submit_and_wait(_input_generator(), batch_size=2)
        )
        assert len(results) == 2
    finally:
        dispatcher.destroy()


def test_pin_staleness_disabled_returns_batch():
    dispatcher = _make_dispatcher(
        pin_staleness=False, max_staleness=3, consumer_batch_size=2
    )
    try:
        results = _run_with_timeout(
            lambda: dispatcher.active_submit_and_wait(_input_generator(), batch_size=2)
        )
        assert len(results) == 2
    finally:
        dispatcher.destroy()
