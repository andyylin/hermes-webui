import pathlib
import subprocess
import sys
from unittest.mock import patch

import bootstrap


def _allow_runtime_pip(monkeypatch) -> None:
    """The suite sets HERMES_DISABLE_LAZY_INSTALLS so tests never install.

    These cases exercise the install path itself, so they opt back in.
    """
    monkeypatch.delenv("HERMES_WEBUI_SKIP_PIP_INSTALL", raising=False)
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)


def test_probe_accepts_agent_that_activates_webui_dependencies_on_import(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "run_agent.py").write_text(
        "import builtins\nbuiltins.agent_dependencies_activated = True\nclass AIAgent: pass\n",
        encoding="utf-8",
    )
    (agent_dir / "yaml.py").write_text(
        "import builtins\n"
        "if not getattr(builtins, 'agent_dependencies_activated', False):\n"
        "    raise ImportError('agent dependencies not activated')\n",
        encoding="utf-8",
    )

    assert bootstrap._python_can_run_webui_and_agent(sys.executable, agent_dir)


def _repo_venv_python(repo_root: pathlib.Path) -> pathlib.Path:
    rel = pathlib.Path(
        "Scripts/python.exe" if bootstrap.platform.system() == "Windows" else "bin/python"
    )
    return repo_root / ".venv" / rel


def test_ensure_python_prefers_agent_venv_when_launcher_cannot_import_agent(monkeypatch, tmp_path):
    """Avoid starting WebUI with a local venv that later cannot import AIAgent."""
    local_python = tmp_path / "webui" / ".venv" / "bin" / "python"
    agent_python = tmp_path / "agent" / "venv" / "bin" / "python"
    agent_python.parent.mkdir(parents=True)
    agent_python.write_text("", encoding="utf-8")

    probes = []

    def fake_can_run(python_exe: str, agent_dir: pathlib.Path | None = None) -> bool:
        probes.append(pathlib.Path(python_exe))
        return pathlib.Path(python_exe) == agent_python

    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", fake_can_run)
    _allow_runtime_pip(monkeypatch)

    selected = bootstrap.ensure_python_has_webui_deps(str(local_python), tmp_path / "agent")

    assert selected == str(agent_python)
    assert probes == [local_python, agent_python]


def test_ensure_python_fails_loudly_when_no_interpreter_can_import_agent(monkeypatch, tmp_path):
    """Do not report health OK when chat would fail with missing AIAgent."""
    local_python = tmp_path / "webui" / ".venv" / "bin" / "python"
    agent_python = tmp_path / "agent" / "venv" / "bin" / "python"
    agent_python.parent.mkdir(parents=True)
    agent_python.write_text("", encoding="utf-8")

    # Pretend REPO_ROOT/.venv already exists with a python binary so the function
    # skips venv.EnvBuilder.create() entirely. Without this, CI runners that
    # don't have a .venv try to build one and the monkey-patched subprocess
    # stub (which only covers subprocess.run, not the venv module's internal
    # subprocess.check_output) fails with AttributeError on .stdout. The
    # behavior under test is "what happens when no interpreter can import
    # both WebUI deps and the agent", not the venv-creation path itself.
    fake_venv_python = tmp_path / "fake-repo-venv-python"
    fake_venv_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)
    # Ensure the platform-native repo-local venv python already exists so
    # EnvBuilder.create() is skipped and the test stays focused on interpreter
    # selection rather than venv creation internals.
    venv_python = _repo_venv_python(tmp_path)
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")
    if (tmp_path / ".venv").exists():  # platform-independent guard
        pass

    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", lambda *a, **k: False)
    _allow_runtime_pip(monkeypatch)
    # Cover both subprocess.run (used for pip install) and any other subprocess
    # entry points the venv module might invoke. Returning None is fine because
    # we never inspect the result on this code path.
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: None)

    try:
        bootstrap.ensure_python_has_webui_deps(str(local_python), tmp_path / "agent")
    except RuntimeError as exc:
        assert "cannot import both WebUI dependencies and Hermes Agent" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_packaged_launch_disables_repo_local_venv_creation(monkeypatch, tmp_path):
    local_python = tmp_path / "webui" / ".venv" / "bin" / "python"
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("HERMES_WEBUI_DISABLE_LOCAL_VENV", "1")
    _allow_runtime_pip(monkeypatch)
    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", lambda *a, **k: False)

    with patch.object(bootstrap.venv, "EnvBuilder") as mock_builder:
        try:
            bootstrap.ensure_python_has_webui_deps(str(local_python), tmp_path / "agent")
        except RuntimeError as exc:
            assert "local .venv creation is disabled" in str(exc)
            assert "HERMES_WEBUI_PYTHON" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    mock_builder.assert_not_called()


def test_local_venv_is_created_with_symlinks(monkeypatch, tmp_path):
    """Regression: mise/asdf macOS Pythons need symlinks=True to avoid SIGABRT.

    Their copy-mode venv produces a python binary referencing
    @executable_path/../lib/libpython3.X.dylib that never gets copied into the
    new .venv. Symlinking keeps @executable_path resolving back to the original
    install. CPython's venv falls back to copy mode if symlink creation fails,
    so this is safe to set unconditionally.
    """
    local_python = tmp_path / "webui" / ".venv" / "bin" / "python"
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)
    _allow_runtime_pip(monkeypatch)
    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", lambda *a, **k: False)
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: None)

    with patch.object(bootstrap.venv, "EnvBuilder") as mock_builder:
        # Make EnvBuilder().create() materialize the venv python so the post-create
        # `_python_can_run_webui_and_agent` retry path doesn't trip on a missing file.
        venv_python = _repo_venv_python(tmp_path)

        def fake_create(target):
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")

        mock_builder.return_value.create.side_effect = fake_create

        try:
            bootstrap.ensure_python_has_webui_deps(str(local_python), None)
        except RuntimeError:
            pass  # expected — fake _python_can_run_webui_and_agent always returns False

        mock_builder.assert_called_once_with(with_pip=True, symlinks=True)


def _pip_commands(calls: list) -> list:
    return [cmd for cmd in calls if isinstance(cmd, (list, tuple)) and "pip" in cmd]


def test_skip_pip_install_env_does_not_invoke_pip(monkeypatch, tmp_path, capsys):
    """HERMES_WEBUI_SKIP_PIP_INSTALL fully skips the install, even if imports fail."""
    local_python = tmp_path / "webui" / ".venv" / "bin" / "python"
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("HERMES_WEBUI_SKIP_PIP_INSTALL", "1")
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", lambda *a, **k: False)
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "missing")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    with patch.object(bootstrap.venv, "EnvBuilder") as mock_builder:
        selected = bootstrap.ensure_python_has_webui_deps(str(local_python), None)

    assert selected == str(local_python)
    assert _pip_commands(calls) == []
    mock_builder.assert_not_called()
    assert "Skipping runtime pip install" in capsys.readouterr().out


def test_disable_lazy_installs_skips_pip(monkeypatch, tmp_path):
    """The Agent's existing opt-out skips the WebUI install step too."""
    local_python = tmp_path / "webui" / ".venv" / "bin" / "python"
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("HERMES_WEBUI_SKIP_PIP_INSTALL", raising=False)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", lambda *a, **k: False)
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "missing")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    selected = bootstrap.ensure_python_has_webui_deps(str(local_python), None)

    assert selected == str(local_python)
    assert _pip_commands(calls) == []


def test_satisfied_interpreter_skips_pip_without_an_installing_probe(monkeypatch, tmp_path):
    """A no-install import success must not fall through into pip or a syncing probe."""
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)
    _allow_runtime_pip(monkeypatch)
    probes = []

    def fake_run(cmd, *args, **kwargs):
        script = ""
        cmd_list = list(cmd)
        if "-c" in cmd_list:
            script = cmd_list[cmd_list.index("-c") + 1]
        env = kwargs.get("env") or {}
        disabled = str(env.get("HERMES_DISABLE_LAZY_INSTALLS", "")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if "from run_agent import AIAgent" in script or "import hermes_bootstrap" in script:
            if not disabled:
                raise AssertionError("satisfaction probe attempted an install")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "pip" in cmd_list:
            raise AssertionError(f"pip invoked for a satisfied interpreter: {cmd_list}")
        return subprocess.CompletedProcess(cmd, 1, "", "")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    def fake_can_run(*_args, **_kwargs):
        probes.append(True)
        raise AssertionError("installing probe ran after dependencies were satisfied")

    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", fake_can_run)
    # imports_without_install lives in managed_agent_startup and uses that module's subprocess.
    monkeypatch.setattr("managed_agent_startup.subprocess.run", fake_run)

    selected = bootstrap.ensure_python_has_webui_deps(str(python), None)

    assert selected == str(python)
    assert probes == []


def test_readonly_pip_failure_warns_once_and_continues(monkeypatch, tmp_path, capsys):
    """A read-only install failure is one warning and does not raise or retry."""
    local_python = tmp_path / "webui" / ".venv" / "bin" / "python"
    monkeypatch.setattr(bootstrap, "REPO_ROOT", tmp_path)
    _allow_runtime_pip(monkeypatch)
    venv_python = _repo_venv_python(tmp_path)
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(bootstrap, "_python_can_run_webui_and_agent", lambda *a, **k: False)
    pip_calls = []

    def fake_run(cmd, *args, **kwargs):
        cmd_list = list(cmd)
        if "pip" in cmd_list:
            pip_calls.append(cmd_list)
            raise subprocess.CalledProcessError(
                1,
                cmd_list,
                stderr="ERROR: Could not install packages due to an OSError: [Errno 30] Read-only file system",
            )
        return subprocess.CompletedProcess(cmd, 1, "", "not executable")

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setattr("managed_agent_startup.subprocess.run", fake_run)

    selected = bootstrap.ensure_python_has_webui_deps(str(local_python), None)

    assert selected == str(local_python)
    assert len(pip_calls) == 1
    err = capsys.readouterr().err
    assert err.count("Continuing without retrying") == 1
    assert "read-only" in err.lower()
