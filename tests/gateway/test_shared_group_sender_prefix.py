import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_prefixes_sender_for_shared_non_thread_group_session():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice] hello"


@pytest.mark.asyncio
async def test_preprocess_keeps_plain_text_for_default_group_sessions():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_preprocess_appends_slack_channel_block_for_shared_sessions():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C047QE96GGP",
        chat_name="Test Channel",
        chat_type="channel",
        user_name="Alice",
        user_id="U094B755UQ3",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice (ID: U094B755UQ3)] [Channel: C047QE96GGP] hello"


@pytest.mark.asyncio
async def test_preprocess_slack_channel_block_prefers_parent_chat_id_for_threads():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.SLACK: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C047QE96GGP:1234.5678",
        parent_chat_id="C047QE96GGP",
        chat_name="Test Channel",
        chat_type="channel",
        user_name="Alice",
        user_id="U094B755UQ3",
        thread_id="1234.5678",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice (ID: U094B755UQ3)] [Channel: C047QE96GGP] hello"
