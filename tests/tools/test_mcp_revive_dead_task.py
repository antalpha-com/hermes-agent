"""A server task that exited its reconnect loop must be restartable.

``MCPServerTask.run()`` returns for good once the reconnect budget is spent.
Nothing ever re-entered it: every later tool call short-circuited on
``not server.session`` with ``"MCP server 'X' is not connected"``, while the
circuit breaker kept advertising ``"Auto-retry available in ~58s"`` — an
auto-retry that could never happen, because the only thing it re-armed was
``_reconnect_event``, which no one was left listening to.  The observed
consequence was a counter climbing forever (``49 → 60 → 62 → 65 consecutive
failures``) with no recovery short of a gateway restart.
"""

import asyncio
import concurrent.futures
import time
from unittest.mock import patch

import pytest

import tools.mcp_tool as m
from tools.mcp_tool import MCPServerTask


@pytest.fixture
def instant_backoff():
    """Collapse reconnect backoff sleeps so a give-up happens promptly."""
    real_sleep = asyncio.sleep

    async def _no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    with patch.object(asyncio, "sleep", _no_wait):
        yield


@pytest.fixture
def mcp_loop():
    """Run the real background MCP loop, and clear the registry afterwards."""
    m._ensure_mcp_loop()
    try:
        yield m
    finally:
        with m._lock:
            m._servers.clear()
        m._server_error_counts.clear()
        m._server_breaker_opened_at.clear()
        m._server_revived_at.clear()
        m._stop_mcp_loop()


def _start_on_loop(server, config):
    """Start ``server.run(config)`` on the MCP loop, as production does."""
    started: concurrent.futures.Future = concurrent.futures.Future()

    def _start():
        server._task = asyncio.ensure_future(server.run(config))
        started.set_result(server._task)

    m._mcp_loop.call_soon_threadsafe(_start)
    return started.result(timeout=5)


def _stop(server):
    """Set the shutdown event from the loop thread.

    ``asyncio.Event.set()`` is not thread-safe: calling it from the test
    thread resolves the waiter future without waking the loop, so the task
    would sit unfinished until something else happened to wake it.
    """
    m._mcp_loop.call_soon_threadsafe(server._shutdown_event.set)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _register(server):
    with m._lock:
        m._servers[server.name] = server


def test_revive_restarts_a_task_that_gave_up(mcp_loop, instant_backoff):
    """Once the server is viable again, a revive brings the session back."""
    server = MCPServerTask("test-revive")
    _register(server)
    healthy = []

    async def fake_run_stdio(self_inner, config):
        if not healthy:
            raise ConnectionError("server down")
        self_inner.session = object()
        self_inner._mark_session_ready()
        await self_inner._shutdown_event.wait()

    with patch.object(MCPServerTask, "_run_stdio", fake_run_stdio):
        task = _start_on_loop(server, {"command": "fake"})
        assert _wait_until(lambda: task.done()), "task should have given up"
        assert server.session is None

        # Upstream recovers; the next tool call triggers a revive.
        healthy.append(True)
        assert m._revive_dead_server("test-revive") is True
        assert _wait_until(lambda: server.session is not None), (
            "revive did not bring the session back"
        )
        assert server._task is not task, "a fresh task should be running"

        _stop(server)
        assert _wait_until(lambda: server._task.done())


def test_no_revive_while_the_task_is_alive(mcp_loop):
    """A healthy (or still-reconnecting) task is left alone."""
    server = MCPServerTask("test-alive")
    _register(server)

    async def fake_run_stdio(self_inner, config):
        self_inner.session = object()
        self_inner._mark_session_ready()
        await self_inner._shutdown_event.wait()

    with patch.object(MCPServerTask, "_run_stdio", fake_run_stdio):
        task = _start_on_loop(server, {"command": "fake"})
        assert _wait_until(lambda: server.session is not None)

        assert m._revive_dead_server("test-alive") is False
        assert server._task is task

        _stop(server)
        assert _wait_until(lambda: task.done())


def test_no_revive_after_shutdown(mcp_loop, instant_backoff):
    """A deliberately stopped server stays stopped."""
    server = MCPServerTask("test-stopped")
    _register(server)

    async def fake_run_stdio(self_inner, config):
        self_inner.session = object()
        self_inner._mark_session_ready()
        await self_inner._shutdown_event.wait()

    with patch.object(MCPServerTask, "_run_stdio", fake_run_stdio):
        task = _start_on_loop(server, {"command": "fake"})
        assert _wait_until(lambda: server.session is not None)
        _stop(server)
        assert _wait_until(lambda: task.done())

    assert m._revive_dead_server("test-stopped") is False


def test_revive_is_rate_limited(mcp_loop, instant_backoff):
    """A burst of tool calls must not spawn a pile of revival tasks."""
    server = MCPServerTask("test-burst")
    _register(server)

    async def fake_run_stdio(self_inner, config):
        raise ConnectionError("still down")

    with patch.object(MCPServerTask, "_run_stdio", fake_run_stdio):
        task = _start_on_loop(server, {"command": "fake"})
        assert _wait_until(lambda: task.done())

        assert m._revive_dead_server("test-burst") is True
        # Second caller, same moment — still inside the cooldown.
        assert m._revive_dead_server("test-burst") is False


def test_revive_is_unknown_server_safe(mcp_loop):
    """An unregistered name is a no-op, not an exception."""
    assert m._revive_dead_server("no-such-server") is False


def test_not_connected_path_attempts_a_revive(mcp_loop):
    """The handler's 'not connected' branch is what re-arms a dead task."""
    server = MCPServerTask("test-handler")
    _register(server)  # registered but never started → no session

    handler = m._make_tool_handler("test-handler", "some_tool", 30.0)

    with patch.object(m, "_revive_dead_server", return_value=False) as revive:
        result = handler({})

    assert "is not connected" in result
    revive.assert_called_once_with("test-handler")
