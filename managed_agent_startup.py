"""Initialize a discovered Agent source checkout before importing WebUI modules."""

import errno
import importlib
import os
from pathlib import Path
import subprocess
import sys

# Truthy spellings shared with bootstrap.py and the Agent's HERMES_DISABLE_LAZY_INSTALLS.
_TRUTHY = {"1", "true", "yes", "on"}
_READONLY_ERRNOS = {errno.EROFS, errno.EACCES, errno.EPERM}
_READONLY_MARKERS = (
    "read-only file system",
    "read-only filesystem",
    "erofs",
    "permission denied",
    "operation not permitted",
)
_UNSET = object()

# Import check that must not trigger `uv sync` ("Installing Python dependencies…").
# hermes_bootstrap's prepare_launch honors HERMES_DISABLE_LAZY_INSTALLS and returns
# before sync or relaunch. pydantic_core is the compiled-extension stand-in: a bare
# run_agent import can succeed on the wrong interpreter ABI.
_SATISFIED_IMPORT_SCRIPT = (
    "import hermes_bootstrap\n"
    "import pydantic_core\n"
    "from run_agent import AIAgent\n"
    "try:\n"
    "    import yaml\n"
    "except ImportError:\n"
    "    import ruamel.yaml\n"
)

_LEGACY_IMPORT_SCRIPT = (
    "from run_agent import AIAgent\n"
    "try:\n"
    "    import yaml\n"
    "except ImportError:\n"
    "    import ruamel.yaml\n"
)

# Exit 0 when this interpreter can create a file in site-packages, 1 when every
# candidate directory failed with a read-only or permission error, 2 otherwise.
# os.access is not enough: systemd ProtectSystem=full leaves mode bits writable
# and fails the open with EROFS.
_WRITE_PROBE_SCRIPT = r"""
import errno, os, sys
candidates = []
try:
    import site
    candidates.extend(site.getsitepackages())
except Exception:
    pass
prefix = getattr(sys, "prefix", "") or ""
if prefix:
    candidates.append(prefix)
seen = False
blocked = False
for directory in candidates:
    if not directory or not os.path.isdir(directory):
        continue
    seen = True
    probe = os.path.join(directory, ".hermes-webui-write-probe")
    fd = None
    try:
        fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, b"ok")
    except FileExistsError:
        try:
            os.unlink(probe)
        except OSError:
            pass
        raise SystemExit(0)
    except OSError as exc:
        if exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            blocked = True
            continue
        raise SystemExit(2)
    finally:
        if fd is not None:
            os.close(fd)
            try:
                os.unlink(probe)
            except OSError:
                pass
    raise SystemExit(0)
if blocked and seen:
    raise SystemExit(1)
raise SystemExit(2)
"""


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def runtime_pip_install_disabled() -> bool:
    """True when the operator opted out of startup pip/uv installs.

    ``HERMES_WEBUI_SKIP_PIP_INSTALL`` is the WebUI switch. ``HERMES_DISABLE_LAZY_INSTALLS``
    is the Agent's existing switch (managed images, systemd units, tests) and means
    the same thing here: do not install into the selected interpreter at startup.
    """
    return _truthy(os.environ.get("HERMES_WEBUI_SKIP_PIP_INSTALL")) or _truthy(
        os.environ.get("HERMES_DISABLE_LAZY_INSTALLS")
    )


def install_failure_is_readonly(text: str, errno_value: int | None = None) -> bool:
    """True when an install failure is a read-only mount or a permission error."""
    if errno_value in _READONLY_ERRNOS:
        return True
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _READONLY_MARKERS)


def _subprocess_code(result: object) -> int | None:
    return getattr(result, "returncode", None)


def interpreter_install_target_writable(python_exe: str | None = None) -> bool | None:
    """Whether ``python_exe`` can write into its own site-packages.

    True means a write succeeded, False means every candidate directory raised
    EROFS/EACCES/EPERM, and None means the probe itself did not run.
    """
    exe = python_exe or sys.executable
    if not exe or not os.access(exe, os.X_OK):
        return None
    try:
        result = subprocess.run(
            [exe, "-c", _WRITE_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    code = _subprocess_code(result)
    if code == 0:
        return True
    if code == 1:
        return False
    return None


def _pythonpath_with(agent_dir: str, env: dict[str, str]) -> None:
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = agent_dir if not current else f"{agent_dir}{os.pathsep}{current}"


def imports_without_install(python_exe: str, agent_dir: str | None) -> bool:
    """True when the interpreter can already import the Agent and YAML.

    The child always sets ``HERMES_DISABLE_LAZY_INSTALLS`` so a managed Agent's
    ``hermes_bootstrap`` import does not run ``uv sync``. When the checkout has
    ``hermes_bootstrap.py``, the child also imports ``pydantic_core`` so a
    pure-Python ``run_agent`` success on the wrong ABI does not count.
    """
    if not python_exe or not os.access(python_exe, os.X_OK):
        return False
    has_bootstrap = bool(agent_dir) and (Path(agent_dir) / "hermes_bootstrap.py").is_file()
    script = _SATISFIED_IMPORT_SCRIPT if has_bootstrap else _LEGACY_IMPORT_SCRIPT
    env = os.environ.copy()
    env["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    if agent_dir:
        _pythonpath_with(agent_dir, env)
    try:
        result = subprocess.run(
            [python_exe, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return _subprocess_code(result) == 0


def _begin_suppress(suppress: bool) -> object:
    if not suppress:
        return _UNSET
    previous = os.environ.get("HERMES_DISABLE_LAZY_INSTALLS")
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
    return previous


def _end_suppress(token: object) -> None:
    if token is _UNSET:
        return
    if token is None:
        os.environ.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
        return
    os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = str(token)


def _startup_install_decision(agent_dir: str) -> tuple[bool, str]:
    """Return (suppress_sync, reason).

    reason is ``skip-env``, ``satisfied``, ``readonly``, or ``""`` when the
    Agent's own launch preparation should run (fresh dev setups).
    """
    if runtime_pip_install_disabled():
        return True, "skip-env"
    if imports_without_install(sys.executable, agent_dir):
        return True, "satisfied"
    writable = interpreter_install_target_writable(sys.executable)
    if writable is False:
        return True, "readonly"
    return False, ""


def activate_managed_agent() -> None:
    agent_dir = os.environ.get("HERMES_WEBUI_AGENT_DIR")
    if not agent_dir:
        return
    # Browser-only setups may deliberately point at an empty Agent directory.
    if not (Path(agent_dir) / "run_agent.py").is_file():
        return

    webui_root = str(Path(__file__).resolve().parent)
    if webui_root not in sys.path:
        sys.path.insert(0, webui_root)
    # Activate dependencies without importing the application: api.config must
    # select the active profile before Agent modules cache profile-sensitive paths.
    # Older Agents and browser-only shims have no bootstrap layer to activate; for
    # them leave sys.path alone so api.config appends the Agent dir at the END as
    # before (a front position lets `pip install -t .` packages in the checkout
    # shadow site-packages).
    if (Path(agent_dir) / "hermes_bootstrap.py").is_file():
        # Bootstrap's probe adds the checkout to its own PYTHONPATH, not ours.
        if agent_dir not in sys.path:
            sys.path.insert(1, agent_dir)
        suppress, reason = _startup_install_decision(agent_dir)
        if reason == "skip-env":
            print(
                "[bootstrap] Skipping startup dependency install "
                "(HERMES_WEBUI_SKIP_PIP_INSTALL or HERMES_DISABLE_LAZY_INSTALLS).",
                flush=True,
            )
        elif reason == "satisfied":
            print(
                "[bootstrap] Agent dependencies already importable; skipping startup install.",
                flush=True,
            )
        elif reason == "readonly":
            print(
                "[!!] Startup dependency install skipped: the Python environment is "
                "read-only or not writable. Continuing without installing or retrying.",
                file=sys.stderr,
                flush=True,
            )
        # Setting the Agent flag for this import makes prepare_launch return before
        # `uv sync` (the "Installing Python dependencies…" line) and before relaunch.
        # Restore it afterwards so a satisfied check does not change later lazy-install
        # policy. An operator-set value is written back unchanged.
        token = _begin_suppress(suppress)
        try:
            importlib.import_module("hermes_bootstrap")
        except SystemExit:
            if reason == "readonly":
                print(
                    "[!!] Startup dependency install hit a read-only or permission "
                    "error. Continuing without retrying.",
                    file=sys.stderr,
                    flush=True,
                )
                return
            raise
        except Exception as exc:  # noqa: BLE001 - SystemExit (relaunch/repair exit) still propagates
            # A broken Agent must not stop WebUI from starting: before this hook the
            # Agent import was lazy and an ImportError only disabled chat, leaving the
            # UI, diagnostics and updater reachable. Keep that behavior.
            detail = f"{type(exc).__name__}: {exc}"
            errno_value = exc.errno if isinstance(exc, OSError) else None
            if reason == "readonly" or install_failure_is_readonly(detail, errno_value):
                print(
                    "[!!] Startup dependency install failed because the environment is "
                    "read-only or not writable. Continuing without retrying.",
                    file=sys.stderr,
                    flush=True,
                )
                return
            print(
                f"[!!] Hermes Agent dependency activation failed: {detail}; "
                "continuing startup. If WebUI then fails to import a dependency, run "
                "`hermes pm repair` or set HERMES_WEBUI_PYTHON.",
                file=sys.stderr,
                flush=True,
            )
        finally:
            _end_suppress(token)
