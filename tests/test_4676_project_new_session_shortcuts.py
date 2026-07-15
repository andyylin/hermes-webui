"""Regression coverage for #4676: project-scope quick conversation creation."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SESSIONS_JS = ROOT / "static" / "sessions.js"
STYLE_CSS = ROOT / "static" / "style.css"
NODE = shutil.which("node")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_function(source: str, name: str) -> str:
    marker = f"function {name}("
    start = source.find(marker)
    assert start >= 0, f"{name} function not found in static/sessions.js"
    brace = source.find("{", start)
    assert brace >= 0, f"{name} declaration has no opening brace"
    depth = 0
    for idx in range(brace, len(source)):
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"{name} function body not closed")


def test_new_session_uses_explicit_project_override_before_active_filter():
    src = _read(SESSIONS_JS)
    assert "Object.prototype.hasOwnProperty.call(options,'project_id')" in src
    assert "reqBody.project_id=options.project_id" in src


def test_quick_create_button_attaches_filter_align_and_request_path():
    src = _read(SESSIONS_JS)
    helper = _extract_function(src, "_attachProjectQuickCreateButton")
    assert "project-chip-quick-create" in helper
    assert "_setActiveProjectFilter(project)" in helper
    assert "newSession(false,{project_id:project.project_id})" in helper
    assert "if(_newSessionInFlight)" in helper
    assert "_setActiveProjectFilter(previousProject)" in helper
    assert "btn.ondblclick" in helper
    assert "btn.oncontextmenu" in helper
    assert "btn.ontouchstart" in helper
    assert "btn.ontouchend" in helper


def test_quick_create_button_render_is_gated_off_by_default():
    """#4676 quick-create buttons must be opt-in: the chip render site only
    attaches the per-project '+' button when window._projectQuickCreate is set."""
    src = _read(SESSIONS_JS)
    assert "if(window._projectQuickCreate) _attachProjectQuickCreateButton(chip,p);" in src
    # The attach call must never run unconditionally at the render site.
    assert "\n      _attachProjectQuickCreateButton(chip,p);" not in src


def test_project_quick_create_styles_exist_and_are_discrete_to_pointer_layouts():
    css = _read(STYLE_CSS)
    assert ".project-chip-quick-create" in css
    assert ".project-chip:hover .project-chip-quick-create" in css
    assert ".project-chip:focus-within .project-chip-quick-create" in css
    assert ".project-chip-quick-create:hover" in css
    assert "@media (hover:none) and (pointer:coarse)" in css


def _run_new_session_case(
    options,
    active_project=None,
    all_projects=None,
    profile_default_workspace=None,
    switch_workspace=None,
    session=None,
):
    _DRIVER = r"""
const fs = require('fs');
const [path, argsJson] = process.argv.slice(-2);
const args = JSON.parse(argsJson);
const src = fs.readFileSync(path, 'utf8');

function extractFunction(source, name) {
  const marker = `function ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0) throw new Error(name + ' not found');
  const brace = source.indexOf('{', start);
  let depth = 0;
  for (let i = brace; i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}') {
      depth--;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error('function body not closed for ' + name);
}

function extractAsyncFunction(source, name) {
  const marker = `async function ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0) throw new Error(name + ' not found');
  const brace = source.indexOf('{', source.indexOf(')', start));
  let depth = 0;
  for (let i = brace; i < source.length; i++) {
    if (source[i] === '{') depth += 1;
    else if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error('function body not closed for ' + name);
}

const resolverSrc = extractFunction(src, '_resolveProjectForNewSession');
const newSessionSrc = extractAsyncFunction(src, 'newSession');

globalThis.window = globalThis;
globalThis.document = {
  baseURI: 'http://example.test/',
  createElement(tag) {
    const node = {
      tagName: String(tag || '').toUpperCase(),
      children: [],
      appendChild(child) { this.children.push(child); },
      textContent: '',
      value: '',
      selectedOptions: [{ dataset: { provider: '' } }],
      dataset: {},
    };
    return node;
  },
};
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
globalThis.history = { replaceState: () => {} };
globalThis.NO_PROJECT_FILTER = '__none__';
globalThis._activeProject = args.activeProject;
globalThis._allProjects = args.allProjects || [];
globalThis._sessionSourceFilter = 'webui';
globalThis._newSessionInFlight = null;
globalThis._messagesTruncated = false;
globalThis._oldestIdx = 0;
globalThis.INFLIGHT = {};
globalThis.S = {
  session: args.session || null,
  toolCalls: [],
  messages: [],
  activeProfile: 'default',
  _pendingSessionToolsets: null,
  _profileSwitchWorkspace: args.switchWorkspace || null,
  _profileDefaultWorkspace: args.profileDefaultWorkspace || null,
};
globalThis._defaultModel = null;
globalThis._activeProvider = 'openai';
globalThis._emptyComposerModelOverride = null;
globalThis._readPersistedModelState = () => null;
globalThis._readEmptyComposerModelOverride = () => null;
globalThis._clearEmptyComposerModelOverride = () => {};
globalThis.$ = (id) => (id === 'modelSelect' ? { value: 'gpt-4', selectedOptions: [{ dataset: { provider: 'openai' } }] } : null);
for (const name of [
  '_setNewSessionPending', 'updateQueueBadge', '_clearPendingSelections',
  'clearLiveToolCards', 'setComposerStatus', 'setStatus', 'updateSendBtn',
  'syncTopbar', 'renderMessages', 'startSessionStream', '_setSessionViewedCount',
  '_setActiveSessionUrl', '_rememberNewChatDraftSession', '_hydrateTodosFromSession',
  '_setLiveAssistantTps', '_syncCtxIndicator', 'showToast'
]) {
  globalThis[name] = () => {};
}
globalThis.loadDir = async () => null;
globalThis._applyModelToDropdown = () => true;
globalThis._modelStateForSelect = () => ({ model: 'gpt-4', model_provider: 'openai' });
globalThis._readPersistedModelState = () => null;
globalThis.getModelLabel = (v) => v || '';
globalThis._defaultModel = null;

const calls = [];
globalThis.api = async (_url, opts) => {
  calls.push(JSON.parse(opts.body));
  return { session: { session_id: 's-1', messages: [], model: 'gpt-4', model_provider: 'openai', workspace: null, message_count: 0, last_usage: {} } };
};

eval(resolverSrc);
eval(newSessionSrc);

(async () => {
  await newSession(false, args.options);
  console.log(JSON.stringify({ body: calls[0] || {} }));
})().catch(err => {
  console.error(String(err && err.stack ? err.stack : err));
  process.exit(1);
});
"""

    payload = {
        "activeProject": active_project,
        "allProjects": all_projects or [],
        "options": options,
        "profileDefaultWorkspace": profile_default_workspace,
        "switchWorkspace": switch_workspace,
        "session": session if session is not None else {"session_id": "session-1"},
    }
    result = subprocess.run(
        [NODE, "-e", _DRIVER, str(SESSIONS_JS), json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"node driver failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])["body"]


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_aligns_project_id_override_when_explicitly_set():
    body = _run_new_session_case(
        {"project_id": "explicit-project"},
        active_project={"profile": "default", "project_id": "active-project"},
    )
    assert body["project_id"] == "explicit-project"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_respects_explicit_project_id_none():
    body = _run_new_session_case(
        {"project_id": None},
        active_project={"profile": "default", "project_id": "active-project"},
    )
    assert "project_id" in body
    assert body["project_id"] is None


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_falls_back_to_active_project_when_override_missing():
    body = _run_new_session_case(
        {},
        active_project={"profile": "default", "project_id": "active-project"},
    )
    assert body["project_id"] == "active-project"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_uses_explicit_project_default_workspace_before_profile_default():
    body = _run_new_session_case(
        {"project_id": "explicit-project"},
        active_project="active-project",
        all_projects=[
            {
                "project_id": "explicit-project",
                "name": "Explicit",
                "default_workspace": "/workspace/project",
            },
            {
                "project_id": "active-project",
                "name": "Active",
                "default_workspace": "/workspace/active",
            },
        ],
        profile_default_workspace="/workspace/profile",
        session={"session_id": "session-1", "workspace": "/workspace/session"},
    )
    assert body["project_id"] == "explicit-project"
    assert body["workspace"] == "/workspace/project"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_uses_active_project_default_workspace_before_profile_default():
    body = _run_new_session_case(
        {},
        active_project="active-project",
        all_projects=[
            {
                "project_id": "active-project",
                "name": "Active",
                "default_workspace": "/workspace/active",
            },
        ],
        profile_default_workspace="/workspace/profile",
        session={"session_id": "session-1", "workspace": "/workspace/session"},
    )
    assert body["project_id"] == "active-project"
    assert body["workspace"] == "/workspace/active"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_uses_profile_qualified_active_project_default_workspace():
    body = _run_new_session_case(
        {},
        active_project={"profile": "work", "project_id": "shared-project"},
        all_projects=[
            {
                "profile": "default",
                "project_id": "shared-project",
                "name": "Default",
                "default_workspace": "/workspace/default",
            },
            {
                "profile": "work",
                "project_id": "shared-project",
                "name": "Work",
                "default_workspace": "/workspace/work",
            },
        ],
        profile_default_workspace="/workspace/profile",
        session={"session_id": "session-1", "workspace": "/workspace/session"},
    )
    assert body["project_id"] == "shared-project"
    assert body["workspace"] == "/workspace/work"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_profile_switch_workspace_overrides_project_default_workspace():
    body = _run_new_session_case(
        {},
        active_project="active-project",
        all_projects=[
            {
                "project_id": "active-project",
                "name": "Active",
                "default_workspace": "/workspace/active",
            },
        ],
        profile_default_workspace="/workspace/profile",
        switch_workspace="/workspace/switch",
        session={"session_id": "session-1", "workspace": "/workspace/session"},
    )
    assert body["project_id"] == "active-project"
    assert body["workspace"] == "/workspace/switch"


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_explicit_project_id_none_does_not_use_active_project_default_workspace():
    body = _run_new_session_case(
        {"project_id": None},
        active_project="active-project",
        all_projects=[
            {
                "project_id": "active-project",
                "name": "Active",
                "default_workspace": "/workspace/active",
            },
        ],
        profile_default_workspace="/workspace/profile",
        session={"session_id": "session-1", "workspace": "/workspace/session"},
    )
    assert body["project_id"] is None
    assert body["workspace"] == "/workspace/profile"


_HELPER = r"""
const fs = require('fs');
const [sessionsPath, paramsJson] = process.argv.slice(-2);
const sessionsSrc = fs.readFileSync(sessionsPath, 'utf8');
const params = JSON.parse(paramsJson);

function extractFunction(source, name) {
  const marker = `function ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0) throw new Error(name + ' not found');
  const brace = source.indexOf('{', start);
  let depth = 0;
  for (let i = brace; i < source.length; i++) {
    if (source[i] === '{') depth++;
    else if (source[i] === '}') {
      depth--;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error('function body not closed for ' + name);
}

globalThis.window = globalThis;
globalThis.document = {
  createElement(tag) {
    return {
      tagName: String(tag || '').toUpperCase(),
      className: '',
      textContent: '',
      children: [],
      appendChild(child) { this.children.push(child); },
      appendChildCallCount: 0,
      attributes: {},
      setAttribute(name, value) { this.attributes[name] = String(value); },
      getAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null; },
      dataset: {},
      type: '',
    };
  },
};
globalThis._setActiveProjectFilter = (project) => {
  globalThis._activeProject = project;
  params.filterProject = project;
  params.filterProjectId = project && typeof project === 'object' ? project.project_id : project;
  params.calls.push({type: 'set-filter', project});
};
globalThis._activeProject = params.activeProject;
globalThis.newSession = async (flash, options) => {
  if (globalThis._newSessionInFlight) {
    params.toasts.push('New conversation already in progress');
    return globalThis._newSessionInFlight;
  }
  if (params.failNewSession) throw new Error(params.failMessage || 'request failed');
  params.newSession = {flash, options};
  params.calls.push({type: 'new-session', flash, options});
  return params.newSessionResult || { session_id: 's-1' };
};
globalThis.showToast = (message) => {
  params.toasts.push(String(message || ''));
};
globalThis._newSessionInFlight = params.newSessionInFlightReject
  ? Promise.reject(new Error(params.newSessionInFlightReject))
  : (params.newSessionInFlight
      ? Promise.resolve(params.newSessionInFlight)
      : null);

eval(extractFunction(sessionsSrc, '_attachProjectQuickCreateButton'));

const chip = {
  appended: [],
  appendChild(child) { this.appended.push(child); },
};
_attachProjectQuickCreateButton(chip, { project_id: params.projectId, profile: params.projectProfile || 'default' });
const btn = chip.appended[0];
const ev = {
  stopPropagation() { params.stopCount++; },
  preventDefault() { params.preventCount++; },
  stopImmediatePropagation() { params.stopImmediateCount++; },
};
const touchEv = {
  stopPropagation() { params.touchStopCount++; },
  preventDefault() { params.touchPreventCount++; },
  stopImmediatePropagation() { params.touchStopImmediateCount++; },
};
(async () => {
  await btn.onclick(ev);
  btn.ondblclick(ev);
  btn.oncontextmenu(ev);
  btn.ontouchstart(touchEv);
  btn.ontouchend(touchEv);
  console.log(JSON.stringify({
    buttonClass: btn.className,
    buttonTag: btn.tagName,
    buttonText: btn.textContent,
    buttonAriaLabel: btn.getAttribute('aria-label'),
    newSession: params.newSession,
    filterProject: params.filterProject,
    filterProjectId: params.filterProjectId,
    stopCount: params.stopCount,
    preventCount: params.preventCount,
    stopImmediateCount: params.stopImmediateCount,
    touchStopCount: params.touchStopCount,
    touchPreventCount: params.touchPreventCount,
    touchStopImmediateCount: params.touchStopImmediateCount,
    calls: params.calls,
    toasts: params.toasts,
  }));
})().catch(err => {
  console.error(String(err && err.stack ? err.stack : err));
  process.exit(1);
});
"""


def _run_quick_create_case(
    project_id="example-project",
    *,
    active_project=None,
    fail_new_session=False,
    new_session_inflight=None,
    new_session_inflight_reject=None,
):
    if active_project is None:
        active_project = {"profile": "default", "project_id": "active-project"}
    payload = {
        "projectId": project_id,
        "projectProfile": "default",
        "activeProject": active_project,
        "filterProject": active_project,
        "filterProjectId": active_project["project_id"],
        "calls": [],
        "stopCount": 0,
        "preventCount": 0,
        "stopImmediateCount": 0,
        "touchStopCount": 0,
        "touchPreventCount": 0,
        "touchStopImmediateCount": 0,
        "failNewSession": fail_new_session,
        "newSessionInFlight": new_session_inflight,
        "newSessionInFlightReject": new_session_inflight_reject,
        "toasts": [],
    }
    result = subprocess.run(
        [NODE, "-e", _HELPER, str(SESSIONS_JS), json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"node helper failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_project_chip_quick_create_keeps_active_filter_and_uses_project_override():
    out = _run_quick_create_case("project-123")
    assert out["buttonClass"] == "project-chip-quick-create"
    assert out["buttonTag"] == "BUTTON"
    assert out["buttonText"] == "+"
    assert out["buttonAriaLabel"] == "New conversation in this project"
    assert out["filterProject"] == {"profile": "default", "project_id": "project-123"}
    assert out["filterProjectId"] == "project-123"
    assert out["newSession"] == {"flash": False, "options": {"project_id": "project-123"}}
    assert {"type": "set-filter", "project": {"profile": "default", "project_id": "project-123"}} in out["calls"]
    assert {"type": "new-session", "flash": False, "options": {"project_id": "project-123"}} in out["calls"]
    assert out["stopCount"] >= 3
    assert out["preventCount"] >= 3
    assert out["stopImmediateCount"] >= 3
    assert out["touchStopCount"] >= 2
    assert out["touchPreventCount"] == 0
    assert out["touchStopImmediateCount"] >= 2


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_project_chip_quick_create_restores_filter_when_new_session_fails():
    out = _run_quick_create_case(
        "project-123",
        active_project={"profile": "default", "project_id": "keep-me"},
        fail_new_session=True,
    )

    assert out["filterProjectId"] == "keep-me"
    assert {"type": "set-filter", "project": {"profile": "default", "project_id": "project-123"}} in out["calls"]
    assert {"type": "set-filter", "project": {"profile": "default", "project_id": "keep-me"}} in out["calls"]
    assert any(msg.startswith("New conversation failed:") for msg in out["toasts"])


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_project_chip_quick_create_leaves_filter_unchanged_during_inflight_guard():
    out = _run_quick_create_case(
        "project-123",
        active_project={"profile": "default", "project_id": "keep-me"},
        new_session_inflight={"session_id": "existing"},
    )

    assert out["filterProjectId"] == "keep-me"
    assert {"type": "set-filter", "project": {"profile": "default", "project_id": "project-123"}} not in out["calls"]
    assert "newSession" not in out


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_project_chip_quick_create_swallows_duplicate_inflight_rejections():
    out = _run_quick_create_case(
        "project-123",
        active_project={"profile": "default", "project_id": "keep-me"},
        new_session_inflight_reject="request failed",
    )

    assert out["filterProjectId"] == "keep-me"
    assert {"type": "set-filter", "project": {"profile": "default", "project_id": "project-123"}} not in out["calls"]
    assert out["toasts"] == ["New conversation already in progress"]


# ── #5457: project default workspace selection ────────────────────────────────


def test_resolve_project_helper_exists():
    """`_resolveProjectForNewSession` must be defined in sessions.js."""
    src = _read(SESSIONS_JS)
    assert "function _resolveProjectForNewSession(" in src, (
        "_resolveProjectForNewSession helper not found in sessions.js"
    )


def test_new_session_workspace_precedence_uses_project_default_before_profile():
    """Workspace resolution uses project default_workspace before profile/session fallback."""
    src = _read(SESSIONS_JS)
    idx = src.find("async function newSession(")
    assert idx >= 0, "newSession function not found in sessions.js"
    new_session_src = src[idx: idx + 2500]
    # The project resolver must be called before building reqBody
    resolver_idx = new_session_src.find("_resolveProjectForNewSession(")
    req_body_idx = new_session_src.find("const reqBody=")
    assert resolver_idx != -1, "_resolveProjectForNewSession not called in newSession"
    assert resolver_idx < req_body_idx, (
        "_resolveProjectForNewSession must be called before reqBody is built"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_uses_active_project_default_workspace():
    """When the active project has default_workspace, newSession() uses it in reqBody.workspace."""
    all_projects = [
        {"project_id": "proj-ws", "name": "MyProject", "default_workspace": "/home/user/projws"},
    ]
    body = _run_new_session_case({}, active_project="proj-ws", all_projects=all_projects)
    assert body.get("workspace") == "/home/user/projws", (
        f"Expected project default workspace, got {body.get('workspace')!r}"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_uses_explicit_project_id_default_workspace():
    """When project_id is passed explicitly, the matching project's default_workspace is used."""
    all_projects = [
        {"project_id": "proj-ws", "name": "MyProject", "default_workspace": "/home/user/projws"},
    ]
    body = _run_new_session_case(
        {"project_id": "proj-ws"},
        active_project=None,
        all_projects=all_projects,
    )
    assert body.get("workspace") == "/home/user/projws", (
        f"Expected explicit project default workspace, got {body.get('workspace')!r}"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_project_id_none_does_not_inherit_project_workspace():
    """project_id:null means no project — project default workspace must NOT be applied."""
    all_projects = [
        {"project_id": "proj-ws", "name": "MyProject", "default_workspace": "/home/user/projws"},
    ]
    body = _run_new_session_case(
        {"project_id": None},
        active_project="proj-ws",
        all_projects=all_projects,
    )
    assert body.get("workspace") is None, (
        f"project_id:null must not inherit project workspace, got {body.get('workspace')!r}"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_new_session_project_without_default_workspace_leaves_workspace_null():
    """A project with no default_workspace leaves reqBody.workspace as null."""
    all_projects = [
        {"project_id": "proj-plain", "name": "PlainProject"},
    ]
    body = _run_new_session_case({}, active_project="proj-plain", all_projects=all_projects)
    assert body.get("workspace") is None, (
        f"Project with no default_workspace must not set workspace, got {body.get('workspace')!r}"
    )
