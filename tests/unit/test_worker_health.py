"""Test worker death detection and fail-fast behavior (TDD).

These tests exercise the public API of AsyncVoxCPMServer/AsyncVoxCPM2Server
with real multiprocessing queues and the actual bridge thread + recv_queue.

Test behaviour:
- WITHOUT the worker-health fix: tests FAIL with asyncio.TimeoutError
  (submit() hangs forever after worker death)
- WITH the fix: tests PASS (submit() raises RuntimeError within ~0.2s)

No test changes are needed between fail→pass — proper TDD.
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from typing import Any

import pytest
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Fake worker processes (used instead of the real GPU-dependent main_loop)
# ---------------------------------------------------------------------------


def _fake_worker_sleep(queue_in, queue_out, args, kwargs):
    """Worker that acks init then sleeps forever (will be SIGKILL'd)."""
    queue_out.put({"type": "init_ok"})
    while True:
        time.sleep(60)


def _fake_worker_echo(queue_in, queue_out, args, kwargs):
    """Worker that responds to all requests (for normal-path test)."""
    queue_out.put({"type": "init_ok"})
    while True:
        try:
            msg = queue_in.get(timeout=5)
        except Exception:
            return
        if msg["type"] == "stop":
            queue_out.put({"id": msg["id"], "type": "response", "data": None})
            return
        queue_out.put({"id": msg["id"], "type": "response", "data": {"status": "ok"}})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _build_server():
    """Factory fixture: builds an AsyncVoxCPM(2)Server with a fake worker."""
    servers = []

    async def _make(server_cls, worker_fn):
        server = server_cls.__new__(server_cls)
        ctx = mp.get_context("spawn")
        server.queue_in = ctx.Queue()
        server.queue_out = ctx.Queue()
        server.process = ctx.Process(
            target=worker_fn,
            args=(server.queue_in, server.queue_out, (), {}),
            daemon=True,
        )
        server.process.start()

        loop = asyncio.get_running_loop()
        server._init_fut = loop.create_future()
        server._worker_dead = False
        server.op_table: dict[str, asyncio.Future[Any]] = {}
        server.stream_table: dict[str, asyncio.Queue] = {}
        server._queue_out_async = asyncio.Queue()
        server._queue_out_stop = threading.Event()
        server._queue_out_thread = threading.Thread(
            target=server._queue_out_bridge,
            args=(loop,),
            daemon=True,
        )
        server._queue_out_thread.start()
        server.recv_task = asyncio.create_task(server.recv_queue())

        await asyncio.wait_for(server._init_fut, timeout=5.0)
        servers.append(server)
        return server

    yield _make

    # Cleanup all servers created during the test
    for server in servers:
        server._queue_out_stop.set()
        server._queue_out_thread.join(timeout=1.0)
        server.recv_task.cancel()


# ---------------------------------------------------------------------------
# VoxCPM v1 tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning:torch")
async def test_pending_request_fails_on_worker_death(_build_server, monkeypatch):
    """After worker death, pending submit() raises RuntimeError (not hang)."""
    import nanovllm_voxcpm.models.voxcpm.server as mod

    monkeypatch.setattr(mod, "main_loop", _fake_worker_sleep)
    server = await _build_server(mod.AsyncVoxCPMServer, _fake_worker_sleep)

    # Submit a request (worker will never respond)
    submit_task = asyncio.create_task(server.submit("health"))
    await asyncio.sleep(0)  # let submit register in op_table

    # Kill the worker (simulate OOM crash)
    os.kill(server.process.pid, signal.SIGKILL)

    # With fix: RuntimeError raised quickly. Without fix: hangs → TimeoutError
    with pytest.raises(RuntimeError, match="worker process died"):
        await asyncio.wait_for(submit_task, timeout=3.0)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning:torch")
async def test_new_submit_rejects_after_death(_build_server, monkeypatch):
    """After worker death, new submit() raises immediately (not hang)."""
    import nanovllm_voxcpm.models.voxcpm.server as mod

    monkeypatch.setattr(mod, "main_loop", _fake_worker_sleep)
    server = await _build_server(mod.AsyncVoxCPMServer, _fake_worker_sleep)

    # Kill the worker
    os.kill(server.process.pid, signal.SIGKILL)
    await asyncio.sleep(0.2)  # bridge polls every 0.1s

    # New submission should fail immediately
    with pytest.raises(RuntimeError, match="worker process died|server must be restarted"):
        await asyncio.wait_for(server.submit("health"), timeout=3.0)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning:torch")
async def test_normal_operation_unaffected(_build_server, monkeypatch):
    """When worker is healthy, submit() resolves normally."""
    import nanovllm_voxcpm.models.voxcpm.server as mod

    monkeypatch.setattr(mod, "main_loop", _fake_worker_echo)
    server = await _build_server(mod.AsyncVoxCPMServer, _fake_worker_echo)

    result = await asyncio.wait_for(server.submit("health"), timeout=3.0)
    assert result == {"status": "ok"}

    await asyncio.wait_for(server.submit("stop"), timeout=3.0)


# ---------------------------------------------------------------------------
# VoxCPM2 tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning:torch")
async def test_pending_request_fails_on_worker_death_v2(_build_server, monkeypatch):
    """After worker death, pending submit() raises RuntimeError (voxcpm2)."""
    import nanovllm_voxcpm.models.voxcpm2.server as mod

    monkeypatch.setattr(mod, "main_loop", _fake_worker_sleep)
    server = await _build_server(mod.AsyncVoxCPM2Server, _fake_worker_sleep)

    submit_task = asyncio.create_task(server.submit("health"))
    await asyncio.sleep(0)  # let submit register in op_table

    os.kill(server.process.pid, signal.SIGKILL)

    with pytest.raises(RuntimeError, match="worker process died"):
        await asyncio.wait_for(submit_task, timeout=3.0)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning:torch")
async def test_new_submit_rejects_after_death_v2(_build_server, monkeypatch):
    """After worker death, new submit() raises immediately (voxcpm2)."""
    import nanovllm_voxcpm.models.voxcpm2.server as mod

    monkeypatch.setattr(mod, "main_loop", _fake_worker_sleep)
    server = await _build_server(mod.AsyncVoxCPM2Server, _fake_worker_sleep)

    os.kill(server.process.pid, signal.SIGKILL)
    await asyncio.sleep(0.2)  # bridge polls every 0.1s

    with pytest.raises(RuntimeError, match="worker process died|server must be restarted"):
        await asyncio.wait_for(server.submit("health"), timeout=3.0)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore::DeprecationWarning:torch")
async def test_normal_operation_unaffected_v2(_build_server, monkeypatch):
    """When worker is healthy, submit() resolves normally (voxcpm2)."""
    import nanovllm_voxcpm.models.voxcpm2.server as mod

    monkeypatch.setattr(mod, "main_loop", _fake_worker_echo)
    server = await _build_server(mod.AsyncVoxCPM2Server, _fake_worker_echo)

    result = await asyncio.wait_for(server.submit("health"), timeout=3.0)
    assert result == {"status": "ok"}

    await asyncio.wait_for(server.submit("stop"), timeout=3.0)
