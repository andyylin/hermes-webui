"""Local regression coverage for Katie's deterministic pre-model guard.

The guard response is persisted as transcript truth before the short-lived SSE
buffer is exposed. No Hermes agent stream, provider, or tool run is created.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

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
