#!/usr/bin/env python3
"""
Manual verification: Worker death detection in AsyncVoxCPMServer.

This script demonstrates that when the worker process dies unexpectedly,
pending requests fail fast with a clear error instead of hanging forever.

It imports the ACTUAL AsyncVoxCPMServer and replaces main_loop with a fake
worker that sleeps (no GPU required).

Usage:
    python examples/verify_worker_health.py

Expected output (WITH fix):
    [0.0s] Starting fake worker process...
    [0.0s] Worker PID: <pid>
    [0.3s] Worker ready (init_ok received)
    [0.3s] Submitting request...
    [0.8s] Killing worker (simulating OOM crash)...
    [~0.9s] ✅ Request failed with: RuntimeError('VoxCPM worker process died unexpectedly')
    [~0.9s] Attempting new submission after death...
    [~0.9s] ✅ New submission rejected: RuntimeError('...')

    PASS: Worker death detected in ~0.1s (would hang forever without fix)

Expected output (WITHOUT fix — before this PR):
    ... (hangs at "Killing worker" step, times out after 5s)
    FAIL: Worker death was not detected (hung forever)
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
import time

# Ensure the project root is importable when running standalone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

# Disable torch.compile (must happen BEFORE any nanovllm import)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from _shims import install_gpu_shims  # noqa: E402

install_gpu_shims()

import torch  # noqa: E402

# Replace @torch.compile with a no-op so activation.py doesn't trigger
# the _inductor import chain (which fails with incompatible triton).
torch.compile = lambda fn=None, *args, **kwargs: fn if fn is not None else (lambda f: f)  # type: ignore[assignment]

import torch.multiprocessing as mp  # noqa: E402


# ---------------------------------------------------------------------------
# Fake worker (no GPU needed)
# ---------------------------------------------------------------------------


def _fake_worker_sleep(queue_in, queue_out, args, kwargs):
    """Worker that acks init then sleeps forever (will be SIGKILL'd)."""
    queue_out.put({"type": "init_ok"})
    while True:
        time.sleep(60)


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------


async def main() -> None:
    # Import the real server module and monkeypatch main_loop
    import nanovllm_voxcpm.models.voxcpm.server as mod

    mod.main_loop = _fake_worker_sleep

    t0 = time.monotonic()

    def elapsed() -> str:
        return f"[{time.monotonic() - t0:.1f}s]"

    print(f"{elapsed()} Starting fake worker process...")

    # Build the server using the real class (bypassing __init__ to avoid
    # requiring model_path argument, but using all real queue infrastructure)
    server = mod.AsyncVoxCPMServer.__new__(mod.AsyncVoxCPMServer)
    ctx = mp.get_context("spawn")
    server.queue_in = ctx.Queue()
    server.queue_out = ctx.Queue()
    server.process = ctx.Process(
        target=_fake_worker_sleep,
        args=(server.queue_in, server.queue_out, (), {}),
        daemon=True,
    )
    server.process.start()

    loop = asyncio.get_running_loop()
    server._init_fut = loop.create_future()
    server._worker_dead = False
    server.op_table = {}
    server.stream_table = {}
    server._queue_out_async = asyncio.Queue()
    server._queue_out_stop = threading.Event()
    server._queue_out_thread = threading.Thread(
        target=server._queue_out_bridge,
        args=(loop,),
        daemon=True,
    )
    server._queue_out_thread.start()
    server.recv_task = asyncio.create_task(server.recv_queue())

    print(f"{elapsed()} Worker PID: {server.process.pid}")

    # Wait for init
    await asyncio.wait_for(server._init_fut, timeout=5.0)
    print(f"{elapsed()} Worker ready (init_ok received)")

    # Submit a request (worker will never respond)
    print(f"{elapsed()} Submitting request...")
    submit_task = asyncio.create_task(server.submit("health"))

    # Give the request time to be queued
    await asyncio.sleep(0.5)

    # Kill the worker process (simulating OOM/segfault)
    print(f"{elapsed()} Killing worker (simulating OOM crash)...")
    os.kill(server.process.pid, signal.SIGKILL)

    # Wait for the pending request to fail (with timeout)
    detection_start = time.monotonic()
    try:
        result = await asyncio.wait_for(submit_task, timeout=5.0)
        print(f"{elapsed()} ❌ UNEXPECTED: Request returned normally: {result}")
        print("\nFAIL: Request should have raised RuntimeError")
        sys.exit(1)
    except RuntimeError as e:
        detection_time = time.monotonic() - detection_start
        print(f"{elapsed()} ✅ Request failed with: {type(e).__name__}('{e}')")
    except asyncio.TimeoutError:
        print(f"{elapsed()} ❌ TIMEOUT: Request hung for 5s — fix not working!")
        print("\nFAIL: Worker death was not detected (hung forever)")
        sys.exit(1)

    # Try submitting after death — should fail immediately
    print(f"{elapsed()} Attempting new submission after death...")
    try:
        await asyncio.wait_for(server.submit("health"), timeout=3.0)
        print(f"{elapsed()} ❌ UNEXPECTED: New submission succeeded")
        print("\nFAIL: submit() should reject after worker death")
        sys.exit(1)
    except RuntimeError as e:
        print(f"{elapsed()} ✅ New submission rejected: {type(e).__name__}('{e}')")
    except asyncio.TimeoutError:
        print(f"{elapsed()} ❌ TIMEOUT: New submission hung — fix not working!")
        print("\nFAIL: submit() should reject immediately after worker death")
        sys.exit(1)

    # Cleanup
    server._queue_out_stop.set()
    server._queue_out_thread.join(timeout=1.0)
    server.recv_task.cancel()
    try:
        await server.recv_task
    except asyncio.CancelledError:
        pass

    print(f"\nPASS: Worker death detected in {detection_time:.2f}s (would hang forever without fix)")


if __name__ == "__main__":
    asyncio.run(main())
