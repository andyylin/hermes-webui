#!/usr/bin/env python3
"""Migrate WebUI user Projects into Hermes Agent's projects.db (dry-run by default)."""

from __future__ import annotations
import argparse, json, shutil, sqlite3, sys, time
from pathlib import Path

SYSTEM_PROJECTS = {"Cron Jobs", "Webhooks"}


def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def build_plan(projects_db, db_path, rows):
    conn = projects_db.connect(db_path=db_path)
    try:
        canonical = projects_db.list_projects(conn, include_archived=True)
    finally:
        conn.close()
    by_name = {}
    for p in canonical:
        by_name.setdefault(p.name.strip().casefold(), []).append(p)
    actions, conflicts, mapping = [], [], {}
    for row in rows:
        old_id, name = (
            str(row.get("project_id") or "").strip(),
            str(row.get("name") or "").strip(),
        )
        if not old_id or not name:
            conflicts.append({"project": row, "reason": "missing project_id or name"})
            continue
        if name in SYSTEM_PROJECTS:
            actions.append(
                {"action": "keep_system_json", "old_id": old_id, "name": name}
            )
            continue
        matches = by_name.get(name.casefold(), [])
        if len(matches) > 1:
            conflicts.append(
                {
                    "project": row,
                    "reason": "multiple canonical projects have the same normalized name",
                    "matches": [p.slug for p in matches],
                }
            )
            continue
        if matches:
            target = matches[0]
            mapping[old_id] = target.slug
            actions.append(
                {
                    "action": "merge",
                    "old_id": old_id,
                    "target_slug": target.slug,
                    "target_id": target.id,
                    "name": name,
                }
            )
        else:
            mapping[old_id] = old_id
            actions.append(
                {
                    "action": "create",
                    "old_id": old_id,
                    "target_slug": old_id,
                    "name": name,
                    "color": row.get("color"),
                    "primary_path": row.get("default_workspace"),
                }
            )
    return {"actions": actions, "conflicts": conflicts, "mapping": mapping}


def apply_plan(projects_db, db_path, projects_file, sessions_dir, plan):
    if plan["conflicts"]:
        raise RuntimeError("migration has conflicts; refusing to apply")
    backup = (
        projects_file.parent
        / f"projects-migration-backup-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    backup.mkdir(parents=True)
    if projects_file.exists():
        shutil.copy2(projects_file, backup / projects_file.name)
    index_file = sessions_dir / "_index.json"
    if index_file.exists():
        shutil.copy2(index_file, backup / index_file.name)
    if db_path.exists():
        src, dst = sqlite3.connect(db_path), sqlite3.connect(backup / "projects.db")
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
    conn = projects_db.connect(db_path=db_path)
    try:
        for a in plan["actions"]:
            if a["action"] == "create":
                projects_db.create_project(
                    conn,
                    name=a["name"],
                    slug=a["target_slug"],
                    color=a.get("color"),
                    primary_path=a.get("primary_path"),
                    folders=[a["primary_path"]] if a.get("primary_path") else [],
                )
    finally:
        conn.close()
    mapping, affected = plan["mapping"], set()
    index = load_json(index_file, [])
    for row in index if isinstance(index, list) else []:
        old = row.get("project_id")
        if old in mapping and mapping[old] != old:
            row["project_id"] = mapping[old]
            if row.get("session_id"):
                affected.add(str(row["session_id"]))
    if index_file.exists():
        atomic_json(index_file, index)
    for sid in sorted(affected):
        path = sessions_dir / f"{sid}.json"
        if not path.exists():
            continue
        shutil.copy2(path, backup / path.name)
        payload = load_json(path, {})
        if payload.get("project_id") in mapping:
            payload["project_id"] = mapping[payload["project_id"]]
            atomic_json(path, payload)
    original = load_json(projects_file, [])
    atomic_json(
        projects_file,
        [r for r in original if str(r.get("name") or "") in SYSTEM_PROJECTS],
    )
    atomic_json(backup / "migration-plan.json", plan)
    return backup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hermes-home", type=Path, default=Path.home() / ".hermes")
    p.add_argument("--webui-state", type=Path)
    p.add_argument("--agent-root", type=Path, required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    home = a.hermes_home.expanduser().resolve()
    state = (a.webui_state or home / "webui").expanduser().resolve()
    sys.path.insert(0, str(a.agent_root.resolve()))
    from hermes_cli import projects_db

    plan = build_plan(
        projects_db, home / "projects.db", load_json(state / "projects.json", [])
    )
    print(json.dumps(plan, indent=2))
    if not a.apply:
        print("DRY RUN: no files changed")
        return 2 if plan["conflicts"] else 0
    print(
        f"APPLIED: backup={apply_plan(projects_db, home / 'projects.db', state / 'projects.json', state / 'sessions', plan)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
