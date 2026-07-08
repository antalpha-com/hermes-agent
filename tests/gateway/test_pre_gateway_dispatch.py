"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"}.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "hello", platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner, adapter


@pytest.mark.asyncio
async def test_hook_skip_short_circuits_dispatch(monkeypatch):
    """A plugin returning {'action': 'skip'} drops the message before auth."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.asyncio
async def test_hook_rewrite_replaces_event_text(monkeypatch):
    """A plugin returning {'action': 'rewrite', 'text': ...} mutates event.text."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen_text = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "REWRITTEN"}]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        seen_text["value"] = event.text
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("original"))

    assert seen_text.get("value") == "REWRITTEN"


@pytest.mark.asyncio
async def test_hook_allow_falls_through_to_auth(monkeypatch):
    """A plugin returning {'action': 'allow'} continues to normal dispatch."""
    _clear_auth_env(monkeypatch)
    # No allowed users set → auth fails → pairing flow triggers.
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "allow"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    # auth chain ran → pairing code was generated
    assert result is None
    runner.pairing_store.generate_code.assert_called_once()


@pytest.mark.asyncio
async def test_hook_exception_does_not_break_dispatch(monkeypatch):
    """A raising plugin hook does not break the gateway."""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        raise RuntimeError("plugin blew up")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = None

    # Should not raise; falls through to auth chain.
    result = await runner._handle_message(_make_event("hi"))
    assert result is None


@pytest.mark.asyncio
async def test_internal_events_bypass_hook(monkeypatch):
    """Internal events (event.internal=True) skip the plugin hook entirely."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    event.internal = True

    # Even though the hook would say skip, internal events bypass it.
    await runner._handle_message(event)
    assert called["count"] == 0


# ---------------------------------------------------------------------------
# Busy path (_handle_active_session_busy_message)
#
# Messages arriving while a turn is running are queued/interrupt-injected and
# never reach _handle_message, so the hook must also run at the busy-path
# entry.  Incident 2026-07-08: a Slack button click ("EMAIL_CONFIRM:<nonce>")
# landed mid-turn and bypassed the email-confirm-guard rewrite entirely.
# ---------------------------------------------------------------------------


def _make_busy_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(
        send=AsyncMock(),
        _send_with_retry=AsyncMock(),
        _pending_messages={},
    )
    runner.adapters = {platform: adapter}
    runner.session_store = MagicMock()
    runner._draining = False
    runner._busy_ack_ts = {}
    runner._is_user_authorized = MagicMock(return_value=True)
    running_agent = MagicMock()
    runner._running_agents = {"sess1": running_agent}
    return runner, adapter, running_agent


@pytest.mark.asyncio
async def test_busy_path_rewrite_replaces_queued_text(monkeypatch):
    """A rewrite from the hook must land in the queued event AND the interrupt."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "REWRITTEN"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter, running_agent = _make_busy_runner(Platform.WHATSAPP)

    handled = await runner._handle_active_session_busy_message(
        _make_event("EMAIL_CONFIRM:abc"), "sess1"
    )

    assert handled is True
    queued = adapter._pending_messages.get("sess1")
    assert queued is not None and queued.text == "REWRITTEN"
    running_agent.interrupt.assert_called_once_with("REWRITTEN")


@pytest.mark.asyncio
async def test_busy_path_skip_drops_message(monkeypatch):
    """A skip from the hook drops the message: nothing queued, no interrupt."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "stale nonce"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter, running_agent = _make_busy_runner(Platform.WHATSAPP)

    handled = await runner._handle_active_session_busy_message(
        _make_event("EMAIL_CONFIRM:stale"), "sess1"
    )

    assert handled is True
    assert adapter._pending_messages == {}
    running_agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_busy_path_internal_event_bypasses_hook(monkeypatch):
    """Internal events skip the hook on the busy path too."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter, running_agent = _make_busy_runner(Platform.WHATSAPP)

    event = _make_event("bg done")
    event.internal = True
    handled = await runner._handle_active_session_busy_message(event, "sess1")

    assert handled is True
    assert called["count"] == 0
    queued = adapter._pending_messages.get("sess1")
    assert queued is not None and queued.text == "bg done"
    running_agent.interrupt.assert_called_once_with("bg done")


@pytest.mark.asyncio
async def test_busy_path_hook_exception_falls_through(monkeypatch):
    """A raising hook must not break the busy path — original text is queued."""
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    def _fake_hook(name, **kwargs):
        raise RuntimeError("plugin blew up")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter, running_agent = _make_busy_runner(Platform.WHATSAPP)

    handled = await runner._handle_active_session_busy_message(
        _make_event("hello"), "sess1"
    )

    assert handled is True
    queued = adapter._pending_messages.get("sess1")
    assert queued is not None and queued.text == "hello"
    running_agent.interrupt.assert_called_once_with("hello")
