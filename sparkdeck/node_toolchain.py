"""Resolve the frontend build toolchain outside an interactive login shell."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


NODE_ENGINE = "^20.19.0 || >=22.12.0"
SUPPORTED_NODE = f"Node.js {NODE_ENGINE}"


@dataclass(frozen=True)
class NodeToolchain:
    npm: Path
    node: Path
    version: str
    path_entries: tuple[Path, ...]


def _supported(version: tuple[int, int, int]) -> bool:
    major, minor, patch = version
    return (
        (major == 20 and (minor, patch) >= (19, 0))
        or (major, minor, patch) >= (22, 12, 0)
    )


def _version_key(path: Path) -> tuple[int, int, int]:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", path.name)
    return tuple(map(int, match.groups())) if match else (0, 0, 0)


def _prepend_path(
    environment: Mapping[str, str], directories: tuple[Path, ...],
) -> dict[str, str]:
    result = dict(environment)
    current = result.get("PATH", "")
    prefix = os.pathsep.join(str(directory) for directory in directories)
    result["PATH"] = prefix + (os.pathsep + current if current else "")
    return result


def frontend_build_environment(
    toolchain: NodeToolchain,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment in which npm can find its matching Node binary."""
    return _prepend_path(
        os.environ if environment is None else environment, toolchain.path_entries,
    )


def _candidate_directories(
    environment: Mapping[str, str],
    home: Path,
    system: str,
) -> list[tuple[Path, ...]]:
    npm_name = "npm.cmd" if system == "Windows" else "npm"
    candidates: list[tuple[Path, ...]] = []
    explicit = environment.get("SPARKDECK_NODE_BIN")
    if explicit:
        configured = Path(explicit).expanduser()
        if not configured.is_absolute() or not configured.is_dir():
            raise RuntimeError("SPARKDECK_NODE_BIN must name an existing absolute directory")
        candidates.append((configured,))

    on_path = shutil.which(npm_name, path=environment.get("PATH"))
    if on_path:
        candidates.append((Path(on_path).parent,))

    if system != "Windows":
        volta_home = Path(environment.get("VOLTA_HOME", home / ".volta")).expanduser()
        asdf_home = Path(environment.get("ASDF_DATA_DIR", home / ".asdf")).expanduser()
        nvm_home = Path(environment.get("NVM_DIR", home / ".nvm")).expanduser()
        fnm_home = Path(
            environment.get("FNM_DIR", home / ".local" / "share" / "fnm")
        ).expanduser()
        candidates.extend([
            (volta_home / "bin",),
            (asdf_home / "shims", asdf_home / "bin"),
        ])
        candidates.extend((path.parent,) for path in sorted(
            (nvm_home / "versions" / "node").glob("*/bin/npm"),
            key=lambda path: _version_key(path.parent.parent),
            reverse=True,
        ))
        candidates.extend((path.parent,) for path in sorted(
            (fnm_home / "node-versions").glob("*/installation/bin/npm"),
            key=lambda path: _version_key(path.parent.parent.parent),
            reverse=True,
        ))
        candidates.extend((path.parent,) for path in sorted(
            (asdf_home / "installs" / "nodejs").glob("*/bin/npm"),
            key=lambda path: _version_key(path.parent.parent),
            reverse=True,
        ))
        candidates.extend([
            (home / ".local" / "bin",),
            (Path("/usr/local/bin"),),
            (Path("/usr/bin"),),
            (Path("/bin"),),
        ])

    unique: list[tuple[Path, ...]] = []
    seen: set[str] = set()
    for candidate in candidates:
        absolute = tuple(Path(os.path.abspath(directory)) for directory in candidate)
        key = os.path.normcase(str(absolute[0]))
        if key not in seen:
            seen.add(key)
            unique.append(absolute)
    return unique


def resolve_node_toolchain(
    environment: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
    system: str | None = None,
    discovery_timeout: float = 30.0,
) -> NodeToolchain:
    """Find a supported Node/npm pair visible to the SparkDeck service.

    systemd user services do not run an interactive shell, so tools installed
    by NVM, Volta, asdf, or fnm are commonly absent from their inherited PATH.
    """
    environment = os.environ if environment is None else environment
    system = platform.system() if system is None else system
    home = Path(environment.get("HOME") or Path.home()) if home is None else Path(home)
    node_name = "node.exe" if system == "Windows" else "node"
    unsupported: list[str] = []

    npm_name = "npm.cmd" if system == "Windows" else "npm"
    deadline = time.monotonic() + discovery_timeout

    def probe_timeout() -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Node.js toolchain discovery timed out after {discovery_timeout:g} seconds"
            )
        return min(10.0, remaining)

    for path_entries in _candidate_directories(environment, home, system):
        npm = path_entries[0] / npm_name
        node = path_entries[0] / node_name
        if not npm.is_file() or not node.is_file():
            continue
        candidate_environment = _prepend_path(environment, path_entries)
        try:
            result = subprocess.run(
                [str(node), "--version"],
                capture_output=True,
                text=True,
                timeout=probe_timeout(),
                check=False,
                env=candidate_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            unsupported.append(f"Node.js at {node} could not run: {exc}")
            continue
        version_text = (result.stdout or result.stderr).strip()
        match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", version_text)
        if result.returncode or not match:
            unsupported.append(f"Node.js at {node} returned an invalid version")
            continue
        version = tuple(map(int, match.groups()))
        if not _supported(version):
            unsupported.append(f"Node.js {version_text} at {node} is unsupported")
            continue
        try:
            npm_result = subprocess.run(
                [str(npm), "--version"],
                capture_output=True,
                text=True,
                timeout=probe_timeout(),
                check=False,
                env=candidate_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            unsupported.append(f"npm at {npm} could not run: {exc}")
            continue
        if npm_result.returncode or not npm_result.stdout.strip():
            unsupported.append(f"npm at {npm} could not report its version")
            continue
        return NodeToolchain(
            npm=npm,
            node=node,
            version=version_text.lstrip("v"),
            path_entries=path_entries,
        )

    if time.monotonic() >= deadline:
        raise RuntimeError(
            f"Node.js toolchain discovery timed out after {discovery_timeout:g} seconds"
        )
    if unsupported:
        raise RuntimeError(f"{unsupported[0]}; SparkDeck requires {SUPPORTED_NODE}")
    raise RuntimeError(
        "Node.js and npm are not available to the SparkDeck service. "
        f"Install {SUPPORTED_NODE} for this user, or set SPARKDECK_NODE_BIN "
        "to their absolute bin directory"
    )


def main() -> None:
    try:
        toolchain = resolve_node_toolchain()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    if len(sys.argv) == 1:
        print(toolchain.npm)
        return
    result = subprocess.run(
        [str(toolchain.npm), *sys.argv[1:]],
        check=False,
        env=frontend_build_environment(toolchain),
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
