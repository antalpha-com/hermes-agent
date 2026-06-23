"""
deliver_report tool — structured cron delivery: main channel summary + thread details.

Replaces the fragile ===THREAD=== text-delimiter approach. Skills call this
tool explicitly with structured parameters; the scheduler detects the call
and skips its own text-based delivery.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Thread-safe session tracking (GIL-protected dict).
# deliver_report sets session_id here on success; run_job pops it after the
# agent finishes to decide whether to suppress text-based delivery.
_DELIVER_REPORT_SESSIONS: Dict[str, bool] = {}

DELIVER_REPORT_SCHEMA = {
    "name": "deliver_report",
    "description": (
        "Deliver a report to the current cron job's configured channel. "
        "Posts `summary` to the main channel, then posts `details` as a "
        "threaded reply under that message. "
        "Only works inside a cron job that has a `deliver` target configured. "
        "Call this tool instead of outputting text with a separator. "
        "After a successful call the scheduler suppresses your final text "
        "response automatically — you do not need to output anything else."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "Brief summary to post to the main channel. "
                    "Should be concise (under ~3000 chars)."
                ),
            },
            "details": {
                "type": "string",
                "description": (
                    "Full detail report to post as a thread reply. "
                    "Can be long — the adapter will chunk it automatically. "
                    "Optional: omit if you only need a main-channel post."
                ),
            },
        },
        "required": ["summary"],
    },
}


def deliver_report_tool(params: Dict[str, Any], **kwargs) -> str:
    summary = (params.get("summary") or "").strip()
    details = (params.get("details") or "").strip()

    if not summary:
        return "error: summary is required and cannot be empty"

    from gateway.session_context import get_session_env

    platform_name = get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip().lower()
    chat_id = get_session_env("HERMES_CRON_AUTO_DELIVER_CHAT_ID", "").strip()
    session_id = get_session_env("HERMES_SESSION_ID", "").strip()

    if not platform_name or not chat_id:
        return "error: deliver_report is only available inside a cron job with a configured 'deliver' target"

    from gateway.config import load_gateway_config, Platform

    try:
        platform = Platform(platform_name)
    except ValueError:
        return f"error: unknown delivery platform {platform_name!r}"

    try:
        config = load_gateway_config()
    except Exception as exc:
        return f"error: failed to load gateway config: {exc}"

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        return f"error: platform {platform_name!r} is not configured or not enabled"

    from tools.send_message_tool import _send_to_platform
    from model_tools import _run_async

    # Step 1: post summary to main channel
    r1 = _run_async(_send_to_platform(platform, pconfig, chat_id, summary))
    if r1 and r1.get("error"):
        return f"error: summary delivery failed: {r1['error']}"

    parent_ts = (r1 or {}).get("message_id")
    thread_ok = False

    # Step 2: post details as thread reply
    if details and parent_ts:
        r2 = _run_async(_send_to_platform(platform, pconfig, chat_id, details, thread_id=parent_ts))
        if r2 and r2.get("error"):
            logger.warning("deliver_report: thread post failed: %s", r2["error"])
        else:
            thread_ok = True
    elif details and not parent_ts:
        logger.warning(
            "deliver_report: summary sent but no message_id returned "
            "— cannot post thread details"
        )

    # Mark this session so run_job suppresses the redundant text delivery
    if session_id:
        _DELIVER_REPORT_SESSIONS[session_id] = True

    status = "ok" if thread_ok else ("ok_no_thread" if not details else "summary_only")
    return f"delivered: status={status} parent_ts={parent_ts}"


def _check_deliver_report(**_) -> bool:
    """Only expose this tool when inside a cron job with a delivery target."""
    from gateway.session_context import get_session_env
    return bool(get_session_env("HERMES_CRON_AUTO_DELIVER_PLATFORM", "").strip())


from tools.registry import registry

registry.register(
    name="deliver_report",
    toolset="cron_delivery",
    schema=DELIVER_REPORT_SCHEMA,
    handler=deliver_report_tool,
    check_fn=_check_deliver_report,
    emoji="📬",
)
