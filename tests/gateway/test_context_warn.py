"""Tests for gateway context-length warning — one-time /new reminder.

Covers:
- SessionEntry.ctx_warn_sent field defaults, serialization, deserialization
- update_session writing ctx_warn_sent
- _maybe_send_context_warn triggering, dedup, fallback context_length
- reset_session clearing the flag
"""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from gateway.session import SessionEntry, SessionSource, SessionStore
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(**kwargs) -> SessionEntry:
    defaults = dict(
        session_key="sk-test",
        session_id="sid-test",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    defaults.update(kwargs)
    return SessionEntry(**defaults)


def _make_store(tmp_path) -> SessionStore:
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=MagicMock())
    store._ensure_loaded_locked()
    return store


def _seed_entry(store: SessionStore) -> SessionEntry:
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1")
    with patch.object(store, "_db", None):
        return store.get_or_create_session(source)


class WarnCaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self): return True
    async def disconnect(self): return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="warn-1")

    async def get_chat_info(self, chat_id): return {"id": chat_id}


def _make_runner(adapter, tmp_path):
    import gateway.run as gr
    runner = gr.GatewayRunner.__new__(gr.GatewayRunner)
    runner.config = MagicMock()
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=MagicMock())
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


async def _call_warn(runner, entry, agent_result, source):
    import gateway.run as gr
    await gr._maybe_send_context_warn(
        runner=runner,
        session_entry=entry,
        agent_result=agent_result,
        source=source,
    )


# ---------------------------------------------------------------------------
# Task 1: SessionEntry.ctx_warn_sent field
# ---------------------------------------------------------------------------

def test_ctx_warn_sent_defaults_to_false():
    entry = _make_entry()
    assert entry.ctx_warn_sent is False


def test_ctx_warn_sent_persists_in_to_dict():
    entry = _make_entry(ctx_warn_sent=True)
    d = entry.to_dict()
    assert d["ctx_warn_sent"] is True


def test_ctx_warn_sent_roundtrips_from_dict():
    entry = _make_entry(ctx_warn_sent=True)
    restored = SessionEntry.from_dict(entry.to_dict())
    assert restored.ctx_warn_sent is True


def test_ctx_warn_sent_missing_from_dict_defaults_false():
    entry = _make_entry()
    d = entry.to_dict()
    del d["ctx_warn_sent"]
    restored = SessionEntry.from_dict(d)
    assert restored.ctx_warn_sent is False


# ---------------------------------------------------------------------------
# Task 2: update_session writes ctx_warn_sent
# ---------------------------------------------------------------------------

def test_update_session_sets_ctx_warn_sent(tmp_path):
    store = _make_store(tmp_path)
    entry = _seed_entry(store)
    assert entry.ctx_warn_sent is False
    store.update_session(entry.session_key, ctx_warn_sent=True)
    assert store._entries[entry.session_key].ctx_warn_sent is True


# ---------------------------------------------------------------------------
# Task 3: _maybe_send_context_warn logic
# ---------------------------------------------------------------------------

def test_warn_sent_when_context_above_threshold(tmp_path):
    import asyncio
    adapter = WarnCaptureAdapter()
    runner = _make_runner(adapter, tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1")
    entry = _make_entry(session_key="sk1", session_id="sid1", ctx_warn_sent=False)
    runner.session_store._entries["sk1"] = entry

    # 80% > 75% threshold → warn
    asyncio.run(_call_warn(runner, entry, {"last_prompt_tokens": 160_000, "context_length": 200_000}, source))

    assert len(adapter.sent) == 1
    assert "/new" in adapter.sent[0]["content"]
    assert entry.ctx_warn_sent is True


def test_warn_not_sent_when_below_threshold(tmp_path):
    import asyncio
    adapter = WarnCaptureAdapter()
    runner = _make_runner(adapter, tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1")
    entry = _make_entry(session_key="sk2", session_id="sid2", ctx_warn_sent=False)
    runner.session_store._entries["sk2"] = entry

    # 50% < 75% threshold → no warn
    asyncio.run(_call_warn(runner, entry, {"last_prompt_tokens": 100_000, "context_length": 200_000}, source))

    assert len(adapter.sent) == 0
    assert entry.ctx_warn_sent is False


def test_warn_not_sent_twice(tmp_path):
    import asyncio
    adapter = WarnCaptureAdapter()
    runner = _make_runner(adapter, tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1")
    entry = _make_entry(session_key="sk3", session_id="sid3", ctx_warn_sent=True)
    runner.session_store._entries["sk3"] = entry

    # already sent → skip
    asyncio.run(_call_warn(runner, entry, {"last_prompt_tokens": 180_000, "context_length": 200_000}, source))

    assert len(adapter.sent) == 0


def test_warn_sent_when_context_length_missing_uses_200k_default(tmp_path):
    import asyncio
    adapter = WarnCaptureAdapter()
    runner = _make_runner(adapter, tmp_path)

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat-1", user_id="u1")
    entry = _make_entry(session_key="sk4", session_id="sid4", ctx_warn_sent=False)
    runner.session_store._entries["sk4"] = entry

    # 160,000 / 200,000 (fallback) = 80%, 超过 75% 阈值 → 应触发提醒
    asyncio.run(_call_warn(runner, entry, {"last_prompt_tokens": 160_000}, source))

    assert len(adapter.sent) == 1
    assert "/new" in adapter.sent[0]["content"]


# ---------------------------------------------------------------------------
# reset_session clears ctx_warn_sent
# ---------------------------------------------------------------------------

def test_warn_reset_after_new_session(tmp_path):
    store = _make_store(tmp_path)
    entry = _seed_entry(store)

    store.update_session(entry.session_key, ctx_warn_sent=True)
    assert store._entries[entry.session_key].ctx_warn_sent is True

    with patch.object(store, "_db", None):
        store.reset_session(entry.session_key)

    assert store._entries[entry.session_key].ctx_warn_sent is False
