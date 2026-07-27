"""Local regression coverage for Katie's deterministic pre-model guard.

The guard response is persisted as transcript truth before the short-lived SSE
buffer is exposed. No Hermes agent stream, provider, or tool run is created.
"""

from __future__ import annotations

import json
import queue
import sys
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import api.config as config
import api.routes as routes


@pytest.fixture(autouse=True)
def _clear_guard_streams():
    with routes._KATIE_ACADEMIC_GUARD_STREAMS_LOCK:
        routes._KATIE_ACADEMIC_GUARD_STREAMS.clear()
    yield
    with routes._KATIE_ACADEMIC_GUARD_STREAMS_LOCK:
        routes._KATIE_ACADEMIC_GUARD_STREAMS.clear()


class _FakeSession:
    def __init__(self, session_id="katie-test-session"):
        self.session_id = session_id
        self.profile = "katie"
        self.messages = []
        self.context_messages = []
        self.active_stream_id = None
        self.pending_user_message = "stale"
        self.pending_attachments = [{"name": "stale.png"}]
        self.pending_started_at = 1.0
        self.pending_user_source = "webui"
        self.saved = 0

    def save(self):
        self.saved += 1

    def compact(self, **_kwargs):
        return {
            "session_id": self.session_id,
            "profile": self.profile,
        }


class _Handler:
    headers = {}

    def __init__(self):
        self.status = None
        self.response_headers = []
        self.wfile = BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.response_headers.append((key, value))

    def end_headers(self):
        pass

    def payload(self):
        return json.loads(self.wfile.getvalue().decode("utf-8"))


def test_katie_guard_decision_is_profile_scoped_and_fails_closed(monkeypatch):
    blocked = SimpleNamespace(
        blocked=True,
        reason_code="misrepresent_authorship",
        response="coaching response",
    )
    module = SimpleNamespace(
        evaluate=lambda _message, recent_messages=None: blocked,
        unavailable_decision=lambda: SimpleNamespace(
            blocked=True,
            reason_code="guard_unavailable",
            response="safe unavailable response",
        ),
    )
    monkeypatch.setitem(sys.modules, "katie_academic_guard", module)
    monkeypatch.setattr(routes, "_get_active_profile_name", lambda: "katie")

    session = _FakeSession()
    assert routes._katie_academic_guard_decision(session, "blocked request") is blocked

    session.profile = "default"
    assert routes._katie_academic_guard_decision(session, "same text") is None

    session.profile = "katie"
    module.evaluate = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    decision = routes._katie_academic_guard_decision(session, "guard failure")
    assert decision.blocked is True
    assert decision.reason_code == "guard_unavailable"


def test_katie_guard_decision_includes_pending_active_turn(monkeypatch):
    captured = {}
    module = SimpleNamespace(
        evaluate=lambda message, recent_messages=None: captured.update(
            message=message,
            recent_messages=list(recent_messages or []),
        ) or SimpleNamespace(blocked=False),
        unavailable_decision=lambda: SimpleNamespace(blocked=True),
    )
    monkeypatch.setitem(sys.modules, "katie_academic_guard", module)
    session = _FakeSession()
    session.messages = [{"role": "user", "content": "Earlier context"}]
    session.pending_user_message = "I have an essay due tomorrow."

    assert routes._katie_academic_guard_decision(session, "Do it for me") is None
    assert captured["recent_messages"][-1] == {
        "role": "user",
        "content": "I have an essay due tomorrow.",
    }


def test_katie_steer_blocks_before_agent_and_covers_pending_short_followup(monkeypatch):
    import api.streaming as streaming

    sid, stream_id = "katie-steer", "katie-stream"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    session.pending_user_message = "I have a school essay due tomorrow."
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    decision = SimpleNamespace(
        blocked=True,
        reason_code="multi_turn_outsourcing",
        response="coaching response",
    )
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda s, text: decision)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
    try:
        handler = _Handler()
        streaming._handle_chat_steer(handler, {"session_id": sid, "text": "Do it for me"})
        assert handler.status == 200
        assert handler.payload() == {
            "accepted": False,
            "fallback": "academic_integrity",
            "stream_id": stream_id,
            "policy_guard": "academic_integrity",
            "message": "coaching response",
        }
        agent.steer.assert_not_called()
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)


def test_katie_steer_allows_owned_work_after_guard_passes(monkeypatch):
    import api.streaming as streaming

    sid, stream_id = "katie-owned", "katie-owned-stream"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda s, text: None)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
    try:
        handler = _Handler()
        streaming._handle_chat_steer(
            handler,
            {"session_id": sid, "text": "Explain the next step; I will solve it."},
        )
        assert handler.payload()["accepted"] is True
        agent.steer.assert_called_once()
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"session_id": ["sid"], "text": "hint"},
        {"session_id": "sid", "text": {"message": "hint"}},
        {"session_id": "sid", "text": "x" * 10001},
    ],
)
def test_steer_malformed_bodies_fail_before_cache_or_agent(body):
    from api.streaming import _handle_chat_steer

    handler = _Handler()
    _handle_chat_steer(handler, body)
    assert handler.status == 400


def test_steer_wrong_session_owner_never_reaches_guard_or_agent(monkeypatch):
    import api.streaming as streaming

    sid = "other-profile-session"
    session = _FakeSession(sid)
    session.profile = "default"
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: False)
    monkeypatch.setattr(
        routes,
        "_katie_academic_guard_decision",
        lambda *_args: pytest.fail("guard must not inspect another profile's transcript"),
    )
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    try:
        handler = _Handler()
        streaming._handle_chat_steer(handler, {"session_id": sid, "text": "hint"})
        assert handler.payload()["fallback"] == "session_not_found"
        agent.steer.assert_not_called()
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)


def test_katie_clarify_response_blocks_before_resuming_agent(monkeypatch):
    session = _FakeSession("katie-clarify")
    decision = SimpleNamespace(
        blocked=True,
        reason_code="outsourced_schoolwork",
        response="coaching response",
    )
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: decision)
    monkeypatch.setattr(
        routes,
        "_resolve_clarify_legacy",
        lambda *_args: pytest.fail("blocked clarification must not resume the run"),
    )

    handler = _Handler()
    routes._handle_clarify_respond(
        handler,
        {"session_id": session.session_id, "response": "Write the assignment for me"},
    )
    assert handler.status == 200
    assert handler.payload() == {
        "ok": False,
        "blocked": True,
        "error": "coaching response",
        "policy_guard": "academic_integrity",
    }


def test_katie_clarify_owned_work_response_resumes_normally(monkeypatch):
    session = _FakeSession("katie-clarify-owned")
    resolved = []
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    monkeypatch.setattr(
        routes,
        "_resolve_clarify_legacy",
        lambda *args: resolved.append(args) or True,
    )
    monkeypatch.setattr(
        "api.runtime_adapter.runtime_adapter_enabled",
        lambda: False,
    )

    handler = _Handler()
    routes._handle_clarify_respond(
        handler,
        {"session_id": session.session_id, "response": "I will try option B myself."},
    )
    assert handler.payload()["ok"] is True
    assert resolved == [(session.session_id, "", "I will try option B myself.")]


def test_katie_sync_fallback_blocks_before_provider_or_agent(monkeypatch):
    session = _FakeSession("katie-sync")
    decision = SimpleNamespace(
        blocked=True,
        reason_code="outsourced_schoolwork",
        response="coaching response",
    )
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: decision)
    monkeypatch.setattr(
        routes,
        "resolve_trusted_workspace",
        lambda *_args: pytest.fail("blocked sync chat must not resolve workspace or provider"),
    )

    handler = _Handler()
    routes._handle_chat_sync(
        handler,
        {"session_id": session.session_id, "message": "Finish my homework for me"},
    )
    assert handler.payload() == {
        "answer": "coaching response",
        "status": "blocked",
        "policy_guard": "academic_integrity",
    }


def test_guarded_turn_persists_transcript_and_zero_usage_without_agent_stream(monkeypatch):
    session = _FakeSession()
    published = []
    monkeypatch.setattr(routes, "redact_session_data", lambda value: value)
    monkeypatch.setattr(
        routes,
        "_publish_session_list_changed",
        lambda event, **kwargs: published.append((event, kwargs)),
    )
    decision = SimpleNamespace(
        blocked=True,
        reason_code="outsourced_schoolwork",
        response="coaching response",
    )

    before_agent_streams = set(routes.STREAMS)
    result = routes._start_katie_guarded_stream(
        session,
        message="do my assignment for me",
        attachments=[],
        decision=decision,
    )

    assert result["policy_guard"] == "academic_integrity"
    assert result["stream_id"].startswith("katie-guard-")
    assert session.saved == 1
    assert [row["role"] for row in session.messages] == ["user", "assistant"]
    assert session.messages[-1]["content"] == "coaching response"
    assert session.messages[-1]["policy_guard"] == "academic_integrity"
    assert session.context_messages[-1]["policy_guard"] == "academic_integrity"
    assert session.active_stream_id is None
    assert session.pending_user_message is None
    assert session.pending_attachments == []
    assert set(routes.STREAMS) == before_agent_streams
    assert published == [("session_message", {"profile": "katie", "session_id": session.session_id})]

    buffered = routes._KATIE_ACADEMIC_GUARD_STREAMS[result["stream_id"]]["events"]
    assert [event for event, _data in buffered] == ["token", "done", "stream_end"]
    done = dict(buffered)["done"]
    assert done["policy_guard"] == "academic_integrity"
    assert done["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "duration_seconds": 0.0,
    }


def test_guard_stream_buffer_prunes_expired_entries_and_stays_bounded(monkeypatch):
    monkeypatch.setattr(routes, "redact_session_data", lambda value: value)
    monkeypatch.setattr(routes, "_publish_session_list_changed", lambda *_args, **_kwargs: None)
    now = time.time()
    with routes._KATIE_ACADEMIC_GUARD_STREAMS_LOCK:
        routes._KATIE_ACADEMIC_GUARD_STREAMS["expired"] = {
            "events": [],
            "expires_at": now - 1,
        }
        for index in range(routes._KATIE_ACADEMIC_GUARD_STREAM_MAX):
            routes._KATIE_ACADEMIC_GUARD_STREAMS[f"live-{index}"] = {
                "events": [],
                "expires_at": now + 10 + index,
            }

    result = routes._start_katie_guarded_stream(
        _FakeSession("bounded-session"),
        message="submit this as my own",
        attachments=[],
        decision=SimpleNamespace(
            blocked=True,
            reason_code="misrepresent_authorship",
            response="coaching response",
        ),
    )

    with routes._KATIE_ACADEMIC_GUARD_STREAMS_LOCK:
        assert "expired" not in routes._KATIE_ACADEMIC_GUARD_STREAMS
        assert len(routes._KATIE_ACADEMIC_GUARD_STREAMS) <= routes._KATIE_ACADEMIC_GUARD_STREAM_MAX
        assert result["stream_id"] in routes._KATIE_ACADEMIC_GUARD_STREAMS
