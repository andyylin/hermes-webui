"""Read-only adapter from hermes_cli.projects_db into WebUI project dicts."""

from __future__ import annotations

import importlib
import logging
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _active_profile_name(profile_name: str | None = None) -> str:
    if profile_name:
        return str(profile_name).strip() or "default"
    try:
        from api.profiles import get_active_profile_name

        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _project_to_webui_dict(project, profile_name: str) -> dict:
    row = {
        "project_id": project.slug,
        "name": project.name,
        "color": project.color,
        "profile": profile_name,
    }
    canonical_id = getattr(project, "id", None)
    if canonical_id is not None:
        row["canonical_id"] = canonical_id
        row["project_source"] = "projects_db"
    created_at = getattr(project, "created_at", None)
    if created_at is not None:
        row["created_at"] = created_at
    primary_path = getattr(project, "primary_path", None)
    if primary_path is not None:
        row["primary_path"] = primary_path
        if canonical_id is not None:
            row["default_workspace"] = primary_path
    folders = getattr(project, "folders", None)
    if folders is not None:
        row["folders"] = [
            folder.to_dict() if hasattr(folder, "to_dict") else folder
            for folder in folders
        ]
    return row


def canonical_projects_enabled() -> bool:
    try:
        from api.config import get_config

        return (get_config().get("projects") or {}).get("canonical_store") is True
    except Exception:
        return False


def _projects_module():
    return importlib.import_module("hermes_cli.projects_db")


def _db_path(profile_name: str | None = None) -> tuple[str, Path]:
    from api.profiles import get_hermes_home_for_profile

    profile = _active_profile_name(profile_name)
    return profile, Path(get_hermes_home_for_profile(profile)) / "projects.db"


@contextmanager
def _writable_projects(profile_name: str | None = None):
    projects_db = _projects_module()
    profile, db_path = _db_path(profile_name)
    conn = projects_db.connect(db_path=db_path)
    try:
        yield projects_db, conn, profile
    finally:
        conn.close()


def create_project_in_db(
    *,
    name: str,
    color: str | None,
    profile_name: str | None = None,
    slug: str | None = None,
    primary_path: str | None = None,
) -> dict:
    with _writable_projects(profile_name) as (projects_db, conn, profile):
        project_id = projects_db.create_project(
            conn,
            name=name,
            slug=slug,
            color=color,
            primary_path=primary_path,
            folders=[primary_path] if primary_path else [],
        )
        project = projects_db.get_project(conn, project_id)
        if project is None:
            raise RuntimeError("project vanished after create")
        return _project_to_webui_dict(project, profile)


def update_project_in_db(
    project_key: str,
    *,
    profile_name: str | None = None,
    name: str | None = None,
    color: str | None = None,
    primary_path: str | None = None,
    update_primary: bool = False,
) -> dict | None:
    with _writable_projects(profile_name) as (projects_db, conn, profile):
        project = projects_db.get_project(conn, project_key)
        if project is None:
            return None
        projects_db.update_project(conn, project.id, name=name, color=color)
        if update_primary:
            if primary_path:
                projects_db.add_folder(conn, project.id, primary_path, is_primary=True)
            elif project.primary_path:
                projects_db.remove_folder(conn, project.id, project.primary_path)
        updated = projects_db.get_project(conn, project.id)
        return _project_to_webui_dict(updated, profile) if updated else None


def delete_project_in_db(project_key: str, *, profile_name: str | None = None) -> bool:
    with _writable_projects(profile_name) as (projects_db, conn, _profile):
        project = projects_db.get_project(conn, project_key)
        return bool(project and projects_db.delete_project(conn, project.id))


def bind_project_workspace(
    project_key: str, workspace: str, *, profile_name: str | None = None
) -> dict | None:
    with _writable_projects(profile_name) as (projects_db, conn, profile):
        project = projects_db.get_project(conn, project_key)
        if project is None:
            return None
        if not project.primary_path:
            projects_db.add_folder(conn, project.id, workspace, is_primary=True)
            project = projects_db.get_project(conn, project.id)
        return _project_to_webui_dict(project, profile) if project else None


def project_for_workspace(
    workspace: str, *, profile_name: str | None = None
) -> dict | None:
    """Resolve workspace membership with Agent's longest-prefix project rule."""
    with _writable_projects(profile_name) as (projects_db, conn, profile):
        project = projects_db.project_for_path(conn, workspace)
        return _project_to_webui_dict(project, profile) if project else None


def load_projects_from_db(*, profile_name: str | None = None) -> list[dict] | None:
    try:
        projects_db = importlib.import_module("hermes_cli.projects_db")
    except Exception:
        return None

    try:
        from api.profiles import get_hermes_home_for_profile

        profile = _active_profile_name(profile_name)
        db_path = Path(get_hermes_home_for_profile(profile)) / "projects.db"
    except Exception:
        return None

    if not db_path or not Path(db_path).exists():
        return None

    resolved_db_path = Path(db_path).resolve()
    wal_path = resolved_db_path.with_name(f"{resolved_db_path.name}-wal")
    shm_path = resolved_db_path.with_name(f"{resolved_db_path.name}-shm")

    def _read_projects(database_path: Path, *, immutable: bool) -> list[dict]:
        conn = None
        query = "mode=ro"
        if immutable:
            query += "&immutable=1"
        db_uri = f"{database_path.as_uri()}?{query}"
        conn = sqlite3.connect(db_uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = []
            for project in projects_db.list_projects(conn):
                if getattr(project, "archived", False):
                    continue
                rows.append(_project_to_webui_dict(project, profile))
            return rows
        finally:
            try:
                conn.close()
            except Exception:
                logger.debug("Failed to close projects_db connection", exc_info=True)

    def _sidecar_state() -> tuple[bool, bool]:
        return wal_path.exists(), shm_path.exists()

    def _read_partial_snapshot() -> list[dict] | None:
        wal_exists, shm_exists = _sidecar_state()
        if not wal_exists and shm_exists:
            return None
        with tempfile.TemporaryDirectory(prefix="hermes-projects-db-") as temp_dir:
            snapshot_db = Path(temp_dir) / resolved_db_path.name
            shutil.copy2(resolved_db_path, snapshot_db)
            if wal_exists:
                shutil.copy2(wal_path, snapshot_db.with_name(f"{snapshot_db.name}-wal"))
            if shm_exists:
                shutil.copy2(shm_path, snapshot_db.with_name(f"{snapshot_db.name}-shm"))
            return _read_projects(snapshot_db, immutable=False)

    def _read_matrix() -> list[dict] | None:
        wal_exists, shm_exists = _sidecar_state()
        if wal_exists and shm_exists:
            return _read_projects(resolved_db_path, immutable=False)
        if wal_exists:
            return _read_partial_snapshot()
        if shm_exists:
            return None
        return _read_projects(resolved_db_path, immutable=True)

    initial_state = _sidecar_state()
    try:
        return _read_matrix()
    except Exception:
        if _sidecar_state() != initial_state:
            try:
                return _read_matrix()
            except Exception:
                pass
        return None
