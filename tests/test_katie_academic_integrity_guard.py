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
        self.active_stream_id: str | None = None
        self.katie_academic_guard_context: dict | None = None
        self.pending_user_message = "stale"
        self.pending_attachments = [{"name": "stale.png"}]
        self.pending_started_at = 1.0
        self.pending_user_source = "webui"
        self.saved = 0

    def save(self, **_kwargs):
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


def test_katie_server_side_turn_blocks_before_workspace_or_provider(monkeypatch):
    session = _FakeSession("katie-server-turn")
    decision = SimpleNamespace(
        blocked=True,
        reason_code="outsourced_schoolwork",
        response="coaching response",
    )
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args, **_kwargs: decision)
    monkeypatch.setattr(
        routes,
        "_resolve_chat_workspace_with_recovery",
        lambda *_args: pytest.fail("blocked server turn must not resolve workspace"),
    )

    result = routes.start_session_turn(
        session.session_id,
        "Write my chemistry report for me",
        source="process_wakeup",
    )

    assert result["policy_guard"] == "academic_integrity"
    assert session.messages[0]["source"] == "process_wakeup"
    assert session.saved == 1


def test_katie_final_chat_binding_rechecks_guard_under_session_lock(monkeypatch):
    session = _FakeSession("katie-final-recheck")
    decision = SimpleNamespace(
        blocked=True,
        reason_code="multi_turn_outsourcing",
        response="coaching response",
    )
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args, **_kwargs: decision)

    result = routes._start_chat_stream_for_session(
        session,
        msg="Do it for me",
        attachments=[],
        workspace="/tmp",
        model="test-model",
    )

    assert result["policy_guard"] == "academic_integrity"
    assert session.active_stream_id is None
    assert session.saved == 1


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


def test_katie_control_context_survives_stream_rotation(monkeypatch):
    captured = {}
    allowed = SimpleNamespace(blocked=False)

    def evaluate(_message, recent_messages=None):
        captured["recent_messages"] = list(recent_messages or [])
        return allowed

    monkeypatch.setitem(
        sys.modules,
        "katie_academic_guard",
        SimpleNamespace(evaluate=evaluate, unavailable_decision=lambda: allowed),
    )
    session = _FakeSession()
    session.active_stream_id = "new-stream"
    session.katie_academic_guard_context = {
        "version": 1,
        "stream_id": "old-stream",
        "messages": [{
            "text": "I have a school essay.",
            "user_message_count": 1,
        }],
        "overflow": False,
        "refusal_pending": False,
    }

    assert routes._katie_academic_guard_decision(session, "Do it for me") is None
    assert captured["recent_messages"] == [
        {"role": "user", "content": "stale"},
        {"role": "user", "content": "I have a school essay."},
    ]


def test_katie_control_context_is_one_newest_aggregate_row(monkeypatch):
    captured = {}
    allowed = SimpleNamespace(blocked=False)

    def evaluate(_message, recent_messages=None):
        captured["recent_messages"] = list(recent_messages or [])
        return allowed

    monkeypatch.setitem(
        sys.modules,
        "katie_academic_guard",
        SimpleNamespace(evaluate=evaluate, unavailable_decision=lambda: allowed),
    )
    session = _FakeSession()
    session.active_stream_id = "active-stream"
    session.pending_user_message = "Tell me about dogs."
    session.katie_academic_guard_context = {
        "version": 1,
        "stream_id": "active-stream",
        "messages": [
            {
                "text": "I have a school essay due tomorrow.",
                "user_message_count": 1,
            },
            *[
                {"text": f"Benign note {index}.", "user_message_count": 1}
                for index in range(8)
            ],
        ],
        "overflow": False,
        "refusal_pending": False,
    }

    assert routes._katie_academic_guard_decision(session, "Do it for me") is None
    assert len(captured["recent_messages"]) == 2
    aggregate = captured["recent_messages"][-1]
    assert aggregate["role"] == "user"
    assert "I have a school essay due tomorrow." in aggregate["content"]
    assert "Benign note 7." in aggregate["content"]


def test_katie_control_context_overflow_fails_closed(monkeypatch):
    unavailable = SimpleNamespace(
        blocked=True,
        reason_code="guard_unavailable",
        response="temporarily unavailable",
    )
    evaluate = MagicMock(return_value=SimpleNamespace(blocked=False))
    monkeypatch.setitem(
        sys.modules,
        "katie_academic_guard",
        SimpleNamespace(evaluate=evaluate, unavailable_decision=lambda: unavailable),
    )
    session = _FakeSession()
    session.active_stream_id = "active-stream"
    session.katie_academic_guard_context = {
        "version": 1,
        "stream_id": "active-stream",
        "messages": [
            {"text": f"note {index}", "user_message_count": 1}
            for index in range(20)
        ],
        "overflow": True,
        "overflow_user_message_count": 1,
        "refusal_pending": False,
    }

    assert routes._katie_academic_guard_decision(session, "one more") is unavailable
    evaluate.assert_not_called()


def test_katie_model_visible_attachments_fail_closed(monkeypatch):
    unavailable = SimpleNamespace(
        blocked=True,
        reason_code="guard_unavailable",
        response="temporarily unavailable",
    )
    evaluate = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "katie_academic_guard",
        SimpleNamespace(evaluate=evaluate, unavailable_decision=lambda: unavailable),
    )
    session = _FakeSession()

    decision = routes._katie_academic_guard_decision(
        session,
        "Please describe this image.",
        attachments=[{"name": "homework.png", "path": "/safe/homework.png"}],
    )

    assert decision is unavailable
    evaluate.assert_not_called()


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
        config.AGENT_INSTANCES[stream_id] = agent
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
        assert session.saved == 1
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


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
        config.AGENT_INSTANCES[stream_id] = agent
    try:
        handler = _Handler()
        streaming._handle_chat_steer(
            handler,
            {"session_id": sid, "text": "Explain the next step; I will solve it."},
        )
        assert handler.payload()["accepted"] is True
        agent.steer.assert_called_once()
        assert session.saved == 1
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


def test_katie_steer_fails_closed_when_control_checkpoint_fails(monkeypatch):
    import api.streaming as streaming

    sid, stream_id = "katie-persist-fail", "stream-persist-fail"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    session.save = MagicMock(side_effect=OSError("disk unavailable"))
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent
    try:
        handler = _Handler()
        streaming._handle_chat_steer(
            handler,
            {"session_id": sid, "text": "I will try this part myself."},
        )
        assert handler.status == 503
        assert handler.payload()["policy_guard"] == "academic_integrity"
        agent.steer.assert_not_called()
        assert session.katie_academic_guard_context is None
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


@pytest.mark.parametrize("rotation", ["stream", "agent"])
def test_katie_steer_rejects_stream_or_agent_rotation_after_guard(monkeypatch, rotation):
    import api.streaming as streaming

    sid, stream_id = "katie-rotate", "katie-rotate-stream"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    replacement = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))

    def rotate_after_guard(_session, _text):
        with config.STREAMS_LOCK:
            if rotation == "stream":
                session.active_stream_id = "replacement-stream"
                config.STREAMS["replacement-stream"] = queue.Queue()
                config.AGENT_INSTANCES["replacement-stream"] = replacement
            else:
                config.AGENT_INSTANCES[stream_id] = replacement
        return None

    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", rotate_after_guard)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent

    try:
        handler = _Handler()
        streaming._handle_chat_steer(
            handler,
            {"session_id": sid, "text": "Explain the next step; I will solve it."},
        )
        assert handler.payload() == {
            "accepted": False,
            "fallback": "stream_dead",
            "stream_id": None,
        }
        agent.steer.assert_not_called()
        replacement.steer.assert_not_called()
        assert session.katie_academic_guard_context is None
        assert session.saved == 2
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            for candidate in (stream_id, "replacement-stream"):
                config.STREAMS.pop(candidate, None)
                config.AGENT_INSTANCES.pop(candidate, None)


def test_katie_failed_steer_delivery_is_not_retained(monkeypatch):
    import api.streaming as streaming

    sid, stream_id = "katie-steer-rejected", "stream-steer-rejected"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    session.pending_user_message = "Tell me about dogs."
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=False))
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent

    try:
        handler = _Handler()
        streaming._handle_chat_steer(
            handler,
            {"session_id": sid, "text": "Keep this only if delivery succeeds."},
        )

        assert handler.payload()["accepted"] is False
        agent.steer.assert_called_once_with("Keep this only if delivery succeeds.")
        assert session.katie_academic_guard_context is None
        assert session.saved == 2
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


def test_katie_failed_steer_rollback_save_fails_closed_with_safe_checkpoint(monkeypatch):
    import api.streaming as streaming

    sid, stream_id = "katie-steer-rollback-fail", "stream-steer-rollback-fail"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    session.pending_user_message = "Tell me about dogs."
    session.save = MagicMock(side_effect=[None, OSError("rollback unavailable")])
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=False))
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent

    try:
        handler = _Handler()
        streaming._handle_chat_steer(
            handler,
            {"session_id": sid, "text": "Keep a conservative checkpoint."},
        )

        assert handler.status == 503
        assert handler.payload()["policy_guard"] == "academic_integrity"
        agent.steer.assert_called_once_with("Keep a conservative checkpoint.")
        assert session.save.call_count == 2
        assert session.katie_academic_guard_context["messages"][-1]["text"] == (
            "Keep a conservative checkpoint."
        )
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


def test_katie_steer_raise_fails_closed_with_durable_checkpoint(monkeypatch):
    import api.streaming as streaming

    sid, stream_id = "katie-steer-raise", "stream-steer-raise"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    agent = SimpleNamespace(
        session_id=sid,
        steer=MagicMock(side_effect=RuntimeError("ambiguous delivery")),
    )
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent

    try:
        handler = _Handler()
        streaming._handle_chat_steer(
            handler,
            {"session_id": sid, "text": "Possibly delivered control."},
        )

        assert handler.status == 503
        assert handler.payload()["policy_guard"] == "academic_integrity"
        assert session.saved == 1
        assert session.katie_academic_guard_context["messages"][-1]["text"] == (
            "Possibly delivered control."
        )
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


def test_katie_split_steers_share_persisted_guard_context(monkeypatch):
    import api.streaming as streaming

    blocked = SimpleNamespace(
        blocked=True,
        reason_code="multi_turn_outsourcing",
        response="policy response",
    )
    allowed = SimpleNamespace(blocked=False, reason_code=None, response="")

    def evaluate(message, recent_messages=None):
        prior = " ".join(
            str(item.get("content") or "")
            for item in (recent_messages or [])
            if isinstance(item, dict)
        ).casefold()
        combined = f"{prior} {message}".casefold()
        if "school essay" in combined and "please do it for me" in combined:
            return blocked
        return allowed

    monkeypatch.setitem(
        sys.modules,
        "katie_academic_guard",
        SimpleNamespace(evaluate=evaluate, unavailable_decision=lambda: blocked),
    )
    sid, stream_id = "katie-split-steer", "stream-split-steer"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    session.pending_user_message = "Tell me about dogs."
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent

    try:
        first = _Handler()
        streaming._handle_chat_steer(
            first,
            {"session_id": sid, "text": "I have a school essay due tomorrow."},
        )
        second = _Handler()
        streaming._handle_chat_steer(
            second,
            {"session_id": sid, "text": "Please do it"},
        )
        assert session.katie_academic_guard_context == {
            "version": 1,
            "stream_id": stream_id,
            "messages": [
                {
                    "text": "I have a school essay due tomorrow.",
                    "user_message_count": 1,
                },
                {"text": "Please do it", "user_message_count": 1},
            ],
            "overflow": False,
            "overflow_user_message_count": None,
            "refusal_pending": False,
        }
        third = _Handler()
        streaming._handle_chat_steer(
            third,
            {"session_id": sid, "text": "for me"},
        )

        assert first.payload()["accepted"] is True
        assert second.payload()["accepted"] is True
        assert third.payload()["accepted"] is False
        assert third.payload()["fallback"] == "academic_integrity"
        assert agent.steer.call_args_list == [
            (("I have a school essay due tomorrow.",), {}),
            (("Please do it",), {}),
        ]
        assert session.katie_academic_guard_context == {
            "version": 1,
            "stream_id": stream_id,
            "messages": [
                {
                    "text": "I have a school essay due tomorrow.",
                    "user_message_count": 1,
                },
                {"text": "Please do it", "user_message_count": 1},
            ],
            "overflow": False,
            "overflow_user_message_count": None,
            "refusal_pending": True,
            "refusal_user_message_count": 1,
        }
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


def test_katie_delivered_steer_context_survives_completion_and_ages(monkeypatch):
    import api.streaming as streaming

    blocked = SimpleNamespace(
        blocked=True,
        reason_code="multi_turn_outsourcing",
        response="policy response",
    )
    allowed = SimpleNamespace(blocked=False, reason_code=None, response="")

    def evaluate(message, recent_messages=None):
        prior = " ".join(
            str(item.get("content") or "")
            for item in (recent_messages or [])
            if isinstance(item, dict)
        ).casefold()
        combined = f"{prior} {message}".casefold()
        if "school essay" in combined and "do it for me" in combined:
            return blocked
        return allowed

    monkeypatch.setitem(
        sys.modules,
        "katie_academic_guard",
        SimpleNamespace(evaluate=evaluate, unavailable_decision=lambda: blocked),
    )
    sid, stream_id = "katie-cross-stream", "stream-cross-stream"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    session.pending_user_message = "Tell me about dogs."
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent

    try:
        accepted = _Handler()
        streaming._handle_chat_steer(
            accepted,
            {"session_id": sid, "text": "I have a school essay due tomorrow."},
        )
        assert accepted.payload()["accepted"] is True
        agent.steer.assert_called_once_with("I have a school essay due tomorrow.")

        # Model-visible steer context must outlive stream cleanup and ordinary
        # benign turns, but should age out with the classifier's bounded window.
        session.active_stream_id = None
        session.pending_user_message = None
        session.messages = [
            {"role": "user", "content": "Tell me about dogs."},
            {"role": "assistant", "content": "Dogs are mammals."},
            *[
                {"role": "user", "content": f"Benign filler {index}."}
                for index in range(5)
            ],
        ]
        assert routes._katie_academic_guard_decision(session, "Do it for me") is blocked

        session.messages.extend([
            {"role": "user", "content": "Benign filler 5."},
            {"role": "user", "content": "Benign filler 6."},
        ])
        assert routes._katie_academic_guard_decision(session, "Do it for me") is None
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


def test_katie_guard_context_round_trips_with_session(tmp_path, monkeypatch):
    from api import models

    monkeypatch.setattr(models, "SESSION_DIR", tmp_path)
    context = {
        "version": 1,
        "stream_id": "stream-persisted",
        "messages": [{
            "text": "I have a school essay due tomorrow.",
            "user_message_count": 1,
        }],
        "overflow": False,
        "refusal_pending": False,
    }
    session = models.Session(
        session_id="katie-guard-persistence",
        workspace=str(tmp_path),
        profile="katie",
        messages=[{"role": "user", "content": "Tell me about dogs."}],
        katie_academic_guard_context=context,
    )

    session.save(skip_index=True)
    loaded = models.Session.load(session.session_id)

    assert loaded.katie_academic_guard_context == context


def test_katie_guard_context_is_not_exposed_in_session_payload(monkeypatch):
    from api import config as api_config
    from api.helpers import redact_session_data

    monkeypatch.setattr(api_config, "load_settings", lambda: {"api_redact_enabled": True})
    payload = redact_session_data({
        "session_id": "katie-internal-state",
        "katie_academic_guard_context": {
            "messages": [{"text": "private accepted control"}],
        },
    })

    assert payload == {"session_id": "katie-internal-state"}


def test_katie_clarify_then_steer_shares_control_context(monkeypatch):
    import api.streaming as streaming

    blocked = SimpleNamespace(
        blocked=True,
        reason_code="multi_turn_outsourcing",
        response="policy response",
    )
    allowed = SimpleNamespace(blocked=False, reason_code=None, response="")

    def evaluate(message, recent_messages=None):
        prior = " ".join(
            str(item.get("content") or "")
            for item in (recent_messages or [])
            if isinstance(item, dict)
        ).casefold()
        return blocked if message == "Do it for me" and "essay" in prior else allowed

    monkeypatch.setitem(
        sys.modules,
        "katie_academic_guard",
        SimpleNamespace(evaluate=evaluate, unavailable_decision=lambda: blocked),
    )
    sid, stream_id = "katie-clarify-split", "stream-clarify-split"
    session = _FakeSession(sid)
    session.active_stream_id = stream_id
    session.pending_user_message = "Tell me about dogs."
    agent = SimpleNamespace(session_id=sid, steer=MagicMock(return_value=True))
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(streaming, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_resolve_clarify_legacy", lambda *_args: True)
    monkeypatch.setattr("api.clarify.pending_contains", lambda *_args: True)
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_enabled", lambda: False)
    with config.SESSION_AGENT_CACHE_LOCK:
        config.SESSION_AGENT_CACHE[sid] = (agent, "sig")
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = queue.Queue()
        config.AGENT_INSTANCES[stream_id] = agent

    try:
        clarify = _Handler()
        routes._handle_clarify_respond(
            clarify,
            {
                "session_id": sid,
                "response": "It is for my school essay.",
                "clarify_id": "clarify-split",
            },
        )
        steer = _Handler()
        streaming._handle_chat_steer(
            steer,
            {"session_id": sid, "text": "Do it for me"},
        )

        assert clarify.payload()["ok"] is True
        assert steer.payload()["accepted"] is False
        assert steer.payload()["fallback"] == "academic_integrity"
        agent.steer.assert_not_called()
    finally:
        with config.SESSION_AGENT_CACHE_LOCK:
            config.SESSION_AGENT_CACHE.pop(sid, None)
        with config.STREAMS_LOCK:
            config.STREAMS.pop(stream_id, None)
            config.AGENT_INSTANCES.pop(stream_id, None)


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


def test_katie_clarify_requires_exact_id_before_guard(monkeypatch):
    session = _FakeSession("katie-clarify-no-id")
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(
        routes,
        "_katie_academic_guard_decision",
        lambda *_args: pytest.fail("missing clarify id must fail before policy evaluation"),
    )

    handler = _Handler()
    routes._handle_clarify_respond(
        handler,
        {"session_id": session.session_id, "response": "I will solve it myself."},
    )
    assert handler.status == 400
    assert handler.payload()["error"] == "clarify_id is required"


def test_katie_clarify_rejects_dead_pending_stream_before_guard(monkeypatch):
    session = _FakeSession("katie-clarify-dead")
    session.active_stream_id = "missing-stream"
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(
        routes,
        "_katie_academic_guard_decision",
        lambda *_args: pytest.fail("dead clarify stream must fail before policy evaluation"),
    )

    handler = _Handler()
    routes._handle_clarify_respond(
        handler,
        {
            "session_id": session.session_id,
            "response": "I will solve it myself.",
            "clarify_id": "stale-clarify",
        },
    )
    assert handler.status == 409
    assert handler.payload()["stale"] is True


def test_katie_clarify_rejects_wrong_exact_id_before_guard(monkeypatch):
    session = _FakeSession("katie-clarify-wrong-id")
    session.active_stream_id = "katie-clarify-live-stream"
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr("api.clarify.pending_contains", lambda *_args: False)
    monkeypatch.setattr(
        routes,
        "_katie_academic_guard_decision",
        lambda *_args: pytest.fail("wrong clarify id must fail before policy evaluation"),
    )

    with config.STREAMS_LOCK:
        config.STREAMS[session.active_stream_id] = queue.Queue()
    try:
        handler = _Handler()
        routes._handle_clarify_respond(
            handler,
            {
                "session_id": session.session_id,
                "response": "Write the assignment for me",
                "clarify_id": "wrong-id",
            },
        )
        assert handler.status == 409
        assert handler.payload()["stale"] is True
        assert session.saved == 0
    finally:
        with config.STREAMS_LOCK:
            config.STREAMS.pop(session.active_stream_id, None)


def test_katie_clarify_response_blocks_before_resuming_agent(monkeypatch):
    session = _FakeSession("katie-clarify")
    session.active_stream_id = "katie-clarify-stream"
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
    monkeypatch.setattr("api.clarify.pending_contains", lambda *_args: True)

    with config.STREAMS_LOCK:
        config.STREAMS[session.active_stream_id] = queue.Queue()
    try:
        handler = _Handler()
        routes._handle_clarify_respond(
            handler,
            {
                "session_id": session.session_id,
                "response": "Write the assignment for me",
                "clarify_id": "clarify-blocked",
            },
        )
        assert handler.status == 200
        assert handler.payload() == {
            "ok": False,
            "blocked": True,
            "error": "coaching response",
            "policy_guard": "academic_integrity",
        }
        assert session.saved == 1
    finally:
        with config.STREAMS_LOCK:
            config.STREAMS.pop(session.active_stream_id, None)


def test_katie_clarify_owned_work_response_resumes_normally(monkeypatch):
    session = _FakeSession("katie-clarify-owned")
    session.active_stream_id = "katie-clarify-owned-stream"
    resolved = []
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    monkeypatch.setattr(
        routes,
        "_resolve_clarify_legacy",
        lambda *args: resolved.append(args) or True,
    )
    monkeypatch.setattr("api.clarify.pending_contains", lambda *_args: True)
    monkeypatch.setattr(
        "api.runtime_adapter.runtime_adapter_enabled",
        lambda: False,
    )

    with config.STREAMS_LOCK:
        config.STREAMS[session.active_stream_id] = queue.Queue()
    try:
        handler = _Handler()
        routes._handle_clarify_respond(
            handler,
            {
                "session_id": session.session_id,
                "response": "I will try option B myself.",
                "clarify_id": "clarify-owned",
            },
        )
        assert handler.payload()["ok"] is True
        assert resolved == [
            (session.session_id, "clarify-owned", "I will try option B myself.")
        ]
        assert session.saved == 1
    finally:
        with config.STREAMS_LOCK:
            config.STREAMS.pop(session.active_stream_id, None)


def test_katie_clarify_rollback_save_fails_closed_with_safe_checkpoint(monkeypatch):
    session = _FakeSession("katie-clarify-rollback-fail")
    session.active_stream_id = "katie-clarify-rollback-stream"
    session.save = MagicMock(side_effect=[None, OSError("rollback unavailable")])
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    monkeypatch.setattr(routes, "_resolve_clarify_legacy", lambda *_args: False)
    monkeypatch.setattr("api.clarify.pending_contains", lambda *_args: True)
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_enabled", lambda: False)

    with config.STREAMS_LOCK:
        config.STREAMS[session.active_stream_id] = queue.Queue()
    try:
        handler = _Handler()
        routes._handle_clarify_respond(
            handler,
            {
                "session_id": session.session_id,
                "response": "Keep a conservative clarify checkpoint.",
                "clarify_id": "clarify-rollback-fail",
            },
        )

        assert handler.status == 503
        assert handler.payload()["policy_guard"] == "academic_integrity"
        assert session.save.call_count == 2
        assert session.katie_academic_guard_context["messages"][-1]["text"] == (
            "Keep a conservative clarify checkpoint."
        )
    finally:
        with config.STREAMS_LOCK:
            config.STREAMS.pop(session.active_stream_id, None)


def test_katie_clarify_raise_fails_closed_with_durable_checkpoint(monkeypatch):
    session = _FakeSession("katie-clarify-raise")
    session.active_stream_id = "katie-clarify-raise-stream"
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    monkeypatch.setattr(
        routes,
        "_resolve_clarify_legacy",
        MagicMock(side_effect=RuntimeError("ambiguous delivery")),
    )
    monkeypatch.setattr("api.clarify.pending_contains", lambda *_args: True)
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_enabled", lambda: False)

    with config.STREAMS_LOCK:
        config.STREAMS[session.active_stream_id] = queue.Queue()
    try:
        handler = _Handler()
        routes._handle_clarify_respond(
            handler,
            {
                "session_id": session.session_id,
                "response": "Possibly delivered clarify response.",
                "clarify_id": "clarify-raise",
            },
        )

        assert handler.status == 503
        assert handler.payload()["policy_guard"] == "academic_integrity"
        assert session.saved == 1
        assert session.katie_academic_guard_context["messages"][-1]["text"] == (
            "Possibly delivered clarify response."
        )
    finally:
        with config.STREAMS_LOCK:
            config.STREAMS.pop(session.active_stream_id, None)


def test_katie_clarify_stream_rotation_rolls_back_durably(monkeypatch):
    session = _FakeSession("katie-clarify-rotation")
    original_stream_id = "katie-clarify-original-stream"
    session.active_stream_id = original_stream_id
    resolve = MagicMock(return_value=True)

    def save_and_rotate(**_kwargs):
        session.saved += 1
        if session.saved == 1:
            session.active_stream_id = "katie-clarify-replacement-stream"

    session.save = save_and_rotate
    monkeypatch.setattr(routes, "get_session", lambda _sid: session)
    monkeypatch.setattr(routes, "_session_visible_to_active_profile", lambda *_args: True)
    monkeypatch.setattr(routes, "_katie_academic_guard_decision", lambda *_args: None)
    monkeypatch.setattr(routes, "_resolve_clarify_legacy", resolve)
    monkeypatch.setattr("api.clarify.pending_contains", lambda *_args: True)
    monkeypatch.setattr("api.runtime_adapter.runtime_adapter_enabled", lambda: False)

    with config.STREAMS_LOCK:
        config.STREAMS[original_stream_id] = queue.Queue()
    try:
        handler = _Handler()
        routes._handle_clarify_respond(
            handler,
            {
                "session_id": session.session_id,
                "response": "I will work through it myself.",
                "clarify_id": "clarify-rotation",
            },
        )

        assert handler.status == 409
        assert handler.payload()["stale"] is True
        resolve.assert_not_called()
        assert session.katie_academic_guard_context is None
        assert session.saved == 2
    finally:
        with config.STREAMS_LOCK:
            config.STREAMS.pop(original_stream_id, None)


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
    assert session.saved == 1


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
