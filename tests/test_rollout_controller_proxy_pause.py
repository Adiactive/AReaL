# SPDX-License-Identifier: Apache-2.0

"""Pause/resume must reach proxy workers, not just rollout workers.

Proxy workers run their own ``RemoteInfEngine`` against the same inference
servers. If they are not paused, they keep issuing generation requests during
the weight-update window while ``pause_generation`` clears those servers'
caches — tokens get attributed to the wrong weight version, and VLM rollouts
crash outright on a stale multimodal cache reference.
"""

from areal.infra.controller.rollout_controller import RolloutController


class _RecordingController:
    """Minimal stand-in exercising only the pause/resume branch logic."""

    def __init__(self, proxy_started: bool):
        self.calls: list[str] = []
        self._proxy_started = proxy_started

        class _Dispatcher:
            def __init__(self, calls):
                self._calls = calls

            def pause(self):
                self._calls.append("dispatcher.pause")

            def resume(self):
                self._calls.append("dispatcher.resume")

        self.dispatcher = _Dispatcher(self.calls)

    def _collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"workers.{method}")

    def _proxy_collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"proxy.{method}")

    pause = RolloutController.pause
    resume = RolloutController.resume


def test_pause_reaches_proxy_workers_when_started():
    """Test that pause is broadcast to proxy workers once the proxy is up.

    Proxy workers are paused before rollout workers so the fewest requests are
    still in flight when ``pause_generation`` later aborts and clears caches.
    """
    controller = _RecordingController(proxy_started=True)

    controller.pause()

    assert controller.calls == [
        "dispatcher.pause",
        "proxy.pause",
        "workers.pause",
    ]


def test_resume_reaches_proxy_workers_when_started():
    """Test that resume is broadcast to proxy workers before dispatching resumes."""
    controller = _RecordingController(proxy_started=True)

    controller.resume()

    # Proxy engines must be un-paused before the dispatcher starts handing out
    # new episodes, otherwise the first requests stall on the pause gate.
    assert controller.calls == [
        "workers.resume",
        "proxy.resume",
        "dispatcher.resume",
    ]


def test_pause_skips_proxy_rpc_without_proxy_workers():
    """Test that plain RolloutWorkflow runs issue no proxy RPC."""
    controller = _RecordingController(proxy_started=False)

    controller.pause()
    controller.resume()

    assert controller.calls == [
        "dispatcher.pause",
        "workers.pause",
        "workers.resume",
        "dispatcher.resume",
    ]
    assert not any(call.startswith("proxy.") for call in controller.calls)
