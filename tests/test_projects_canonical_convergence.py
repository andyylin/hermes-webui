"""Canonical Agent/WebUI Project convergence contracts."""

import importlib.util, io, json, threading
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
import pytest
from api import projects_db_adapter as adapter


@dataclass
class Project:
    id: str
    slug: str
    name: str
    color: str | None = None
    primary_path: str | None = None
    folders: list = field(default_factory=list)
    created_at: float = 1


class Conn:
    def close(self):
        pass


class DB:
    def __init__(self, projects=()):
        self.projects = {p.id: p for p in projects}
        self.n = 1

    def connect(self, db_path=None):
        self.path = db_path
        return Conn()

    def list_projects(self, _c, include_archived=False):
        return list(self.projects.values())

    def create_project(self, _c, **kw):
        pid = f"p_{self.n}"
        self.n += 1
        self.projects[pid] = Project(
            pid,
            kw.get("slug") or kw["name"].lower().replace(" ", "-"),
            kw["name"],
            kw.get("color"),
            kw.get("primary_path"),
        )
        return pid

    def get_project(self, _c, key):
        return self.projects.get(key) or next(
            (p for p in self.projects.values() if p.slug == key), None
        )

    def update_project(self, _c, pid, **kw):
        for k, v in kw.items():
            if v is not None:
                setattr(self.projects[pid], k, v)
        return True

    def add_folder(self, _c, pid, path, is_primary=False):
        if is_primary:
            self.projects[pid].primary_path = path
        return "f"

    def remove_folder(self, _c, pid, path):
        if self.projects[pid].primary_path == path:
            self.projects[pid].primary_path = None
        return True

    def delete_project(self, _c, pid):
        return self.projects.pop(pid, None) is not None


@pytest.fixture
def fake_db(monkeypatch, tmp_path):
    db = DB()
    monkeypatch.setattr(adapter, "_projects_module", lambda: db)
    monkeypatch.setattr(
        adapter,
        "_db_path",
        lambda profile_name=None: (profile_name or "default", tmp_path / "projects.db"),
    )
    return db


def test_db_rows_map_primary_workspace_and_source():
    row = adapter._project_to_webui_dict(
        Project("p", "app", "App", primary_path="/app"), "work"
    )
    assert (
        row["project_id"],
        row["canonical_id"],
        row["default_workspace"],
        row["project_source"],
    ) == ("app", "p", "/app", "projects_db")


def test_canonical_crud_and_first_workspace_binding(fake_db):
    assert (
        adapter.create_project_in_db(name="App", color=None, slug="app")["project_id"]
        == "app"
    )
    assert adapter.bind_project_workspace("app", "/app")["primary_path"] == "/app"
    assert adapter.bind_project_workspace("app", "/other")["primary_path"] == "/app"
    assert adapter.update_project_in_db("app", name="Renamed")["name"] == "Renamed"
    assert adapter.delete_project_in_db("app") is True


def migration_module():
    path = Path(__file__).parents[1] / "scripts" / "migrate_projects_to_agent_db.py"
    spec = importlib.util.spec_from_file_location("migration", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_merges_exact_names_keeps_system_and_preserves_new_ids():
    plan = migration_module().build_plan(
        DB([Project("p", "appsheet", "AppSheet")]),
        Path("x"),
        [
            {"project_id": "old", "name": "AppSheet"},
            {"project_id": "cron", "name": "Cron Jobs"},
            {"project_id": "new", "name": "Test"},
        ],
    )
    assert plan["conflicts"] == [] and plan["mapping"] == {
        "old": "appsheet",
        "new": "new",
    }
    assert [x["action"] for x in plan["actions"]] == [
        "merge",
        "keep_system_json",
        "create",
    ]


def test_migration_fails_closed_on_ambiguous_names():
    plan = migration_module().build_plan(
        DB([Project("1", "a", "App"), Project("2", "b", "app")]),
        Path("x"),
        [{"project_id": "old", "name": "APP"}],
    )
    assert plan["mapping"] == {} and plan["conflicts"]


class Handler:
    def __init__(self, body):
        raw = json.dumps(body).encode()
        self.rfile = io.BytesIO(raw)
        self.wfile = self
        self.headers = {"Content-Length": str(len(raw))}
        self.body = bytearray()
        self.status = None
        self.client_address = ("127.0.0.1", 0)

    def send_response(self, s):
        self.status = s

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass

    def write(self, d):
        self.body.extend(d)

    def payload(self):
        return json.loads(self.body)


def post(path, body):
    from api.routes import handle_post

    h = Handler(body)
    handle_post(h, urlparse(path))
    return h


def test_canonical_create_route(monkeypatch):
    monkeypatch.setattr(adapter, "canonical_projects_enabled", lambda: True)
    monkeypatch.setattr(
        adapter,
        "create_project_in_db",
        lambda **kw: {"project_id": "new", "name": kw["name"], "profile": "default"},
    )
    h = post("/api/projects/create", {"name": "New"})
    assert h.status == 200 and h.payload()["project"]["project_id"] == "new"


def test_canonical_rename_and_delete_routes(monkeypatch):
    import api.routes as routes

    row = {
        "project_id": "app",
        "canonical_id": "p",
        "project_source": "projects_db",
        "name": "App",
        "profile": "default",
    }
    monkeypatch.setattr(adapter, "canonical_projects_enabled", lambda: True)
    monkeypatch.setattr(routes, "load_projects", lambda **_kw: [row])
    monkeypatch.setattr(
        adapter,
        "update_project_in_db",
        lambda *_a, **_kw: {**row, "name": "Renamed"},
    )
    monkeypatch.setattr(adapter, "delete_project_in_db", lambda *_a, **_kw: True)
    monkeypatch.setattr(routes, "all_sessions", lambda: [])
    monkeypatch.setattr(routes, "publish_session_list_changed", lambda *_a, **_kw: None)
    renamed = post("/api/projects/rename", {"project_id": "app", "name": "Renamed"})
    assert renamed.status == 200 and renamed.payload()["project"]["name"] == "Renamed"
    deleted = post("/api/projects/delete", {"project_id": "app"})
    assert deleted.status == 200 and deleted.payload()["ok"] is True


def test_move_switches_to_canonical_workspace(monkeypatch, tmp_path):
    import api.routes as routes

    target = tmp_path / "app"
    target.mkdir()

    class Session:
        session_id = "s"
        profile = "default"
        workspace = "/old"
        worktree_path = None
        project_id = None

        def save(self):
            pass

        def compact(self):
            return {
                "session_id": "s",
                "project_id": self.project_id,
                "workspace": self.workspace,
            }

    s = Session()
    project = {
        "project_id": "app",
        "canonical_id": "p",
        "project_source": "projects_db",
        "name": "App",
        "profile": "default",
        "primary_path": str(target),
    }
    monkeypatch.setattr(adapter, "canonical_projects_enabled", lambda: True)
    monkeypatch.setattr(routes, "_get_or_materialize_session", lambda _sid: s)
    monkeypatch.setattr(routes, "load_projects", lambda **kw: [project])
    monkeypatch.setattr(routes, "resolve_trusted_workspace", lambda p: Path(p))
    monkeypatch.setattr(
        routes, "_get_session_agent_lock", lambda _sid: threading.Lock()
    )
    monkeypatch.setattr(routes, "publish_session_list_changed", lambda *a, **kw: None)
    h = post("/api/session/move", {"session_id": "s", "project_id": "app"})
    assert h.status == 200 and s.project_id == "app" and s.workspace == str(target)


def test_workspace_update_recomputes_canonical_project(monkeypatch, tmp_path):
    import api.routes as routes

    target = tmp_path / "app"
    target.mkdir()

    class Session:
        session_id = "s"
        profile = "default"
        workspace = "/old"
        worktree_path = None
        project_id = "old"
        model = "m"
        model_provider = "p"
        messages = []

        def save(self):
            pass

        def compact(self):
            return {
                "session_id": "s",
                "project_id": self.project_id,
                "workspace": self.workspace,
            }

    s = Session()
    monkeypatch.setattr(adapter, "canonical_projects_enabled", lambda: True)
    monkeypatch.setattr(
        adapter, "project_for_workspace", lambda *_a, **_kw: {"project_id": "app"}
    )
    monkeypatch.setattr(routes, "_get_or_materialize_session", lambda _sid: s)
    monkeypatch.setattr(routes, "resolve_trusted_workspace", lambda p: Path(p))
    monkeypatch.setattr(
        routes, "_get_session_agent_lock", lambda _sid: threading.Lock()
    )
    monkeypatch.setattr(routes, "set_last_workspace", lambda _path: None)
    h = post("/api/session/update", {"session_id": "s", "workspace": str(target)})
    assert h.status == 200 and s.project_id == "app" and s.workspace == str(target)
