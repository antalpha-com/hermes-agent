"""The reconnect budget is per-outage, not per-process-lifetime.

``MCPServerTask.run()`` initialises ``retries`` once, outside its
``while True`` loop.  Before this fix nothing reset it after a successful
reconnect, so ``_MAX_RECONNECT_RETRIES`` acted as a lifetime allowance:
five unrelated blips — each individually recovered, days apart — exhausted
it and the sixth drop returned out of ``run()`` for good.  Every later tool
call then failed instantly with ``"MCP server 'X' is not connected"`` until
the gateway was restarted.

Observed in production on 2026-08-18: ``armada`` burned attempts 1/5 through
5/5 between 08-14 and 08-17 (all recovered within 16s, hundreds of successful
calls in between) and died on the 08-18 drop, 27 hours after the previous one.
"""

import asyncio
from unittest.mock import patch

import pytest

from tools.mcp_tool import MCPServerTask, _MAX_RECONNECT_RETRIES


@pytest.fixture
def instant_backoff():
    """Collapse the reconnect backoff sleeps so the loop runs at full speed.

    Only ``run()``'s backoff awaits ``asyncio.sleep`` during these tests, and
    the real coroutine is kept for the yield point, so the event loop still
    gets to breathe between iterations.
    """
    real_sleep = asyncio.sleep

    async def _no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    with patch.object(asyncio, "sleep", _no_wait):
        yield


def _drive(coro_fn):
    """Run one coroutine to completion on a private event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro_fn())
    finally:
        loop.close()


def test_stable_session_resets_reconnect_budget(instant_backoff):
    """Drops separated by healthy uptime never exhaust the budget."""
    # Well past _MAX_RECONNECT_RETRIES: the pre-fix code gave up on drop 6.
    total_drops = _MAX_RECONNECT_RETRIES + 3
    drops = 0

    async def _run():
        nonlocal drops
        server = MCPServerTask("test-stable-drops")

        async def fake_run_stdio(self_inner, config):
            nonlocal drops
            self_inner._mark_session_ready()
            # Backdate the connect stamp: this session served traffic for
            # six hours before an upstream proxy cut the stream.
            self_inner._connected_at -= 6 * 3600
            if drops >= total_drops:
                await self_inner._shutdown_event.wait()
                return
            drops += 1
            raise ConnectionError(f"proxy cut the stream (drop {drops})")

        with patch.object(MCPServerTask, "_run_stdio", fake_run_stdio):
            task = asyncio.ensure_future(server.run({"command": "fake"}))
            # Park the loop until the fake has ridden out every drop and is
            # waiting on shutdown.
            while drops < total_drops and not task.done():
                await asyncio.sleep(0)

            assert not task.done(), (
                f"run() gave up after {drops} recovered drops — the reconnect "
                "budget is still a per-process lifetime allowance"
            )
            assert drops == total_drops

            server._shutdown_event.set()
            await task

    _drive(_run)


def test_crash_loop_still_gives_up(instant_backoff):
    """Sessions that die on arrival keep accumulating and still hit the cap.

    The reset must not turn a genuine connect-crash loop into an infinite
    reconnect storm — only uptime clears the counter.
    """
    attempts = 0

    async def _run():
        nonlocal attempts
        server = MCPServerTask("test-crash-loop")

        async def fake_run_stdio(self_inner, config):
            nonlocal attempts
            attempts += 1
            # Ready, then dead immediately — zero seconds of uptime.
            self_inner._mark_session_ready()
            raise ConnectionError("server exits right after initialize")

        with patch.object(MCPServerTask, "_run_stdio", fake_run_stdio):
            await server.run({"command": "fake"})

        # One initial failure plus _MAX_RECONNECT_RETRIES reconnects.
        assert attempts == _MAX_RECONNECT_RETRIES + 1

    _drive(_run)


def test_connected_at_cleared_between_sessions(instant_backoff):
    """A stale connect stamp must not credit uptime the next session never had.

    Without clearing it, the backdated stamp from a long-lived session would
    still be readable on the *following* failure, resetting the counter for a
    session that actually crashed on arrival.
    """
    attempts = 0

    async def _run():
        nonlocal attempts
        server = MCPServerTask("test-stale-stamp")

        async def fake_run_stdio(self_inner, config):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                # One genuinely healthy session, then it drops.
                self_inner._mark_session_ready()
                self_inner._connected_at -= 6 * 3600
            else:
                # Every later attempt dies before establishing a session, so
                # _connected_at must be None rather than the stale stamp.
                assert self_inner._connected_at is None, (
                    "connect stamp leaked from the previous session"
                )
            raise ConnectionError("drop")

        with patch.object(MCPServerTask, "_run_stdio", fake_run_stdio):
            await server.run({"command": "fake"})

        # Drop 1 clears the counter and then takes slot 1 of the fresh
        # budget, so the crash loop that follows still gives up after
        # _MAX_RECONNECT_RETRIES reconnects rather than running forever.
        assert attempts == _MAX_RECONNECT_RETRIES + 1

    _drive(_run)
