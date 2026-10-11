"""Bandwidth-capped Docker image pulls.

``docker pull`` fetches registry bytes inside the Docker daemon, outside any
process SparkDeck owns, so a user-space rate limit cannot be applied to it.
When a max download bandwidth is configured, the pull instead runs through
two helper containers on an isolated Docker network:

* a small token-bucket HTTPS CONNECT proxy (python slim image) that also
  counts the bytes it moves, and
* ``skopeo`` copying the image into a docker-archive tarball through it.

The tarball is then loaded with ``docker load``. Only tagged references are
supported: docker-archive archives register their identity as RepoTags, and
docker load never registers digest references, so a digest-pinned pull cannot
become resolvable through this path and is rejected up front. After loading,
the registered reference is verified before success is reported.

Registry credentials follow ``docker login``: an inline ``auths`` entry from
the daemon-side ``.docker/config.json`` is mounted read-only; otherwise the
configured credential helper is consulted, and when neither yields a
credential the copy proceeds anonymously (a global ``credsStore`` says where
Docker keeps credentials, not that every registry requires them).

When no cap is configured the normal daemon-side ``docker pull`` is used and
nothing here runs. The first capped pull fetches the two helper images with
an uncapped daemon pull; they are small and fetched once per node.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

PROXY_IMAGE = "python:3.12-slim"
SKOPEO_IMAGE = "quay.io/skopeo/stable:latest"
_PROXY_PORT = 8899
# Written by the proxy into the shared staging volume; the poll loop reads it
# from the host. It counts bytes the token bucket actually moved.
PROGRESS_NAME = "proxy_bytes"

# The token bucket mirrors sparkdeck.download_throttle's virtual-time
# schedule, but it runs inside the proxy container where no project module is
# importable. Only the download direction (registry → client) is charged,
# and the running total is reported for progress.
_PROXY_SCRIPT = """
import os, socket, threading, time

RATE = float(os.environ["SPARKDECK_DOWNLOAD_BPS"])
PROGRESS = os.environ.get("SPARKDECK_BANDWIDTH_PROGRESS") or ""
DUE = [0.0]
LOCK = threading.Lock()
TOTAL = [0]
LAST_REPORT = [0.0]

def report(force=False):
    if not PROGRESS:
        return
    now = time.monotonic()
    if not force and now - LAST_REPORT[0] < 0.25:
        return
    LAST_REPORT[0] = now
    try:
        tmp = PROGRESS + ".tmp"
        with open(tmp, "w") as handle:
            handle.write(str(TOTAL[0]))
        os.replace(tmp, PROGRESS)
    except OSError:
        pass

def spend(n):
    if n <= 0:
        return
    with LOCK:
        TOTAL[0] += n
        now = time.monotonic()
        due = max(now, DUE[0]) + n / RATE
        DUE[0] = due
        wait = due - now
    if wait > 0:
        time.sleep(wait)
    report()

def pump(src, dst, charge):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            if charge:
                spend(len(data))
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def handle(client):
    upstream = None
    try:
        request = b""
        while b"\\r\\n\\r\\n" not in request and len(request) < 65536:
            chunk = client.recv(4096)
            if not chunk:
                return
            request += chunk
        head, _, rest = request.partition(b"\\r\\n\\r\\n")
        parts = head.split(b"\\r\\n")[0].decode("latin-1").split()
        if len(parts) < 3 or parts[0].upper() != "CONNECT":
            client.sendall(b"HTTP/1.1 405 Method Not Allowed\\r\\n\\r\\n")
            return
        host, _, port = parts[1].partition(":")
        upstream = socket.create_connection((host, int(port or "443")), 10)
        client.sendall(b"HTTP/1.1 200 Connection established\\r\\n\\r\\n")
        if rest:
            upstream.sendall(rest)
        threading.Thread(target=pump, args=(upstream, client, True), daemon=True).start()
        pump(client, upstream, False)
    except OSError:
        pass
    finally:
        report(force=True)
        for sock in (client, upstream):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass

listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("0.0.0.0", %d))
listener.listen(128)
while True:
    connection, _ = listener.accept()
    threading.Thread(target=handle, args=(connection,), daemon=True).start()
""" % _PROXY_PORT


class ThrottledPullError(RuntimeError):
    pass


def normalize_image_reference(image: str) -> tuple[str, str]:
    """Return (docker:// source ref, RepoTag the load must register).

    Unqualified references get the Docker Hub expansion that ``docker pull``
    performs, decided from the first path segment: a colon there is a port
    only when a slash follows it (``nginx:1.27`` has a tag, not a registry).
    A missing tag gets ``latest``. Digest references are rejected: docker
    load registers only RepoTags, so the requested digest would not resolve
    after the copy.
    """
    reference = str(image or "").strip()
    if not reference or any(char.isspace() for char in reference):
        raise ThrottledPullError("image reference is not valid")
    if reference.startswith("docker://"):
        raise ThrottledPullError("image reference must not include a transport prefix")
    if "@" in reference:
        raise ThrottledPullError(
            "bandwidth-limited pulls cannot register a digest-pinned image; "
            "use a tagged reference or set the download limit to 0"
        )
    head, slash, _ = reference.partition("/")
    qualified = bool(slash) and ("." in head or ":" in head or head == "localhost")
    if not qualified:
        reference = f"docker.io/{reference}" if slash else f"docker.io/library/{reference}"
    segment = reference.rsplit("/", 1)[-1]
    if ":" in segment:
        if not segment.split(":", 1)[1]:
            raise ThrottledPullError("image reference is not valid")
    else:
        reference = f"{reference}:latest"
    return f"docker://{reference}", reference


def skopeo_copy_args(
    source_ref: str, archive_ref: str, arch: str, *, has_authfile: bool,
) -> list[str]:
    """Build skopeo argv (without the entrypoint) for one capped copy."""
    args = ["copy", "--src-tls-verify=true", "--override-os", "linux"]
    if arch:
        args += ["--override-arch", arch]
    if has_authfile:
        args += ["--authfile", "/auth/config.json"]
    args += [source_ref, archive_ref]
    return args


def select_docker_config(auth_paths: list[str]) -> str | None:
    """Pick the daemon-side docker config.json the SDK pull would use."""
    for candidate in auth_paths:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _registry_aliases(registry_host: str) -> list[str]:
    if registry_host == "docker.io":
        return ["docker.io", "index.docker.io"]
    return [registry_host]


def _auth_entry(docker_config: dict, registry_host: str) -> dict | None:
    """Return the config's inline auth entry for the registry, if any."""
    for key, entry in (docker_config.get("auths") or {}).items():
        key_host = str(key).removeprefix("https://").removeprefix("http://")
        key_host = key_host.partition("/")[0].lower()
        if key_host in _registry_aliases(registry_host) and isinstance(entry, dict):
            if entry.get("auth"):
                return entry
    return None


_HELPER_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def helper_credentials(docker_config: dict, registry_host: str) -> dict | None:
    """Resolve registry credentials through the configured docker helper.

    A ``credsStore`` says where Docker keeps credentials, not that a registry
    needs them. When the helper binary is missing, has nothing stored, or
    fails, this returns None and the copy proceeds anonymously — matching
    what an unthrottled ``docker pull`` does for public images.
    """
    cred_helpers = docker_config.get("credHelpers") or {}
    store = ""
    for alias in _registry_aliases(registry_host):
        if alias in cred_helpers:
            store = str(cred_helpers[alias] or "")
            break
    if not store:
        store = str(docker_config.get("credsStore") or "")
    if not store or not _HELPER_NAME.fullmatch(store):
        return None
    binary = shutil.which(f"docker-credential-{store}")
    if binary is None:
        return None
    servers = (
        ["https://index.docker.io/v1/", "docker.io", "index.docker.io"]
        if registry_host == "docker.io" else [registry_host]
    )
    for server in servers:
        try:
            result = subprocess.run(
                [binary, "get"], input=server.encode("utf-8"),
                capture_output=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            continue
        try:
            payload = json.loads(result.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        username, secret = payload.get("Username"), payload.get("Secret")
        if isinstance(username, str) and isinstance(secret, str) and secret:
            auth = base64.b64encode(
                f"{username}:{secret}".encode("utf-8"),
            ).decode("ascii")
            return {"auth": auth}
    return None


def _read_counter(path: Path) -> int:
    try:
        return max(0, int(path.read_text().strip() or 0))
    except (OSError, ValueError):
        return 0


async def _run_docker(client: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(getattr(client.api, method), *args, **kwargs)


async def _ensure_image(client: Any, image: str) -> None:
    """Pull a helper image with the daemon if it is not cached locally."""
    try:
        await _run_docker(client, "inspect_image", image)
    except Exception:
        await _run_docker(client, "pull", image)


async def _exec_ok(client: Any, container: str, command: list[str]) -> bool:
    exec_id = await _run_docker(client, "exec_create", container, cmd=command)
    await _run_docker(client, "exec_start", exec_id["Id"], demux=False)
    result = await _run_docker(client, "exec_inspect", exec_id["Id"])
    return isinstance(result, dict) and result.get("ExitCode") == 0


async def _exec_output(client: Any, container: str, command: list[str]) -> bytes:
    exec_id = await _run_docker(client, "exec_create", container, cmd=command)
    output = await _run_docker(client, "exec_start", exec_id["Id"], demux=True)
    result = await _run_docker(client, "exec_inspect", exec_id["Id"])
    if not isinstance(result, dict) or result.get("ExitCode") != 0:
        raise ThrottledPullError(f"'{command[1] if len(command) > 1 else command[0]}' failed")
    if isinstance(output, tuple):
        output = output[0] or b""
    return bytes(output or b"")


async def _image_total_bytes(
    client: Any, container: str, source_ref: str,
    arch: str, has_authfile: bool,
) -> int:
    """Fetch the image's total size with ``skopeo inspect`` inside ``container``.

    Progress totals are a convenience: every failure path here falls back to
    0 (unknown total) and lets the copy itself surface real errors. The exec
    races container start-up slightly, so a few retries are allowed.
    """
    command = ["skopeo", "inspect", "--tls-verify=true"]
    if arch:
        command += ["--override-os", "linux", "--override-arch", arch]
    if has_authfile:
        command += ["--authfile", "/auth/config.json"]
    command.append(source_ref)
    for attempt in range(10):
        try:
            output = await _exec_output(client, container, command)
        except Exception:
            await asyncio.sleep(0.3 if attempt < 9 else 0.0)
            continue
        try:
            payload = json.loads(output.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return 0
        if isinstance(payload, dict) and isinstance(payload.get("Size"), int):
            return max(0, payload["Size"])
        return 0
    return 0


async def throttled_image_pull_events(
    client: Any, image: str, bytes_per_second: float,
) -> AsyncIterator[dict]:
    """Yield docker-pull-shaped progress dicts for one bandwidth-capped pull.

    Raises nothing: a failure is emitted as ``{"error": ...}``, matching the
    event stream contract of ``manager.pull_image_stream``.
    """
    loop = asyncio.get_running_loop()
    try:
        source_ref, repo_tag = normalize_image_reference(image)
    except ThrottledPullError as exc:
        yield {"error": str(exc)}
        return
    network_id: str | None = None
    suffix = uuid.uuid4().hex[:8]
    network_name = f"sparkdeck-pull-{suffix}"
    proxy_name = f"sparkdeck-pull-{suffix}-proxy"
    proxy_container: str | None = None
    copy_container: str | None = None
    auth_dir: Path | None = None
    staging = Path(tempfile.mkdtemp(prefix="sparkdeck-pull-"))
    try:
        yield {"status": f"Preparing bandwidth-limited pull of {image}"}
        version = await _run_docker(client, "version")
        arch = str((version or {}).get("Arch") or "")
        await _ensure_image(client, PROXY_IMAGE)
        await _ensure_image(client, SKOPEO_IMAGE)
        network = await _run_docker(
            client, "create_network", name=network_name, driver="bridge",
        )
        network_id = str(network["Id"])
        registry_host = source_ref.removeprefix("docker://").partition("/")[0]
        authfile = select_docker_config([
            "/root/.docker/config.json",
            os.path.expanduser("~/.docker/config.json"),
        ])
        docker_config: dict = {}
        if authfile:
            try:
                with open(authfile, encoding="utf-8") as handle:
                    loaded = json.load(handle)
                docker_config = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                docker_config = {}
        copy_binds = [f"{staging}:/archive"]
        has_authfile = False
        entry = _auth_entry(docker_config, registry_host)
        if entry is not None:
            has_authfile = True
            copy_binds.append(f"{authfile}:/auth/config.json:ro")
        else:
            helper_auth = helper_credentials(docker_config, registry_host)
            if helper_auth is not None:
                auth_dir = Path(tempfile.mkdtemp(prefix="sparkdeck-pull-auth-"))
                written = auth_dir / "config.json"
                written.write_text(
                    json.dumps({"auths": {registry_host: helper_auth}}),
                    encoding="utf-8",
                )
                try:
                    os.chmod(written, 0o600)
                except OSError:
                    pass
                has_authfile = True
                copy_binds.append(f"{written}:/auth/config.json:ro")
        proxy = await _run_docker(
            client, "create_container",
            image=PROXY_IMAGE,
            name=proxy_name,
            command=["python", "-u", "-c", _PROXY_SCRIPT],
            environment={
                "SPARKDECK_DOWNLOAD_BPS": repr(float(bytes_per_second)),
                "SPARKDECK_BANDWIDTH_PROGRESS": f"/archive/{PROGRESS_NAME}",
                "PYTHONUNBUFFERED": "1",
            },
            host_config={
                "Binds": [f"{staging}:/archive"],
                "NetworkMode": network_id,
            },
        )
        proxy_container = proxy["Id"]
        await _run_docker(client, "start_container", proxy_container)
        proxy_ready = False
        for _ in range(60):
            if await _exec_ok(client, proxy_container, [
                "python", "-c",
                "import socket,sys; sys.exit(0 if socket.create_connection(("
                f"'127.0.0.1',{_PROXY_PORT}),1) else 1)",
            ]):
                proxy_ready = True
                break
            await asyncio.sleep(0.5)
        if not proxy_ready:
            raise ThrottledPullError("the bandwidth limiter did not start")
        user = None if os.name == "nt" else f"{os.getuid()}:{os.getgid()}"
        copy = await _run_docker(
            client, "create_container",
            image=SKOPEO_IMAGE,
            command=skopeo_copy_args(
                source_ref,
                f"docker-archive:/archive/image.tar:{repo_tag}",
                arch, has_authfile=has_authfile,
            ),
            environment={
                "HTTPS_PROXY": f"http://{proxy_name}:{_PROXY_PORT}",
                "HTTP_PROXY": f"http://{proxy_name}:{_PROXY_PORT}",
                "NO_PROXY": "localhost,127.0.0.1",
                "HOME": "/tmp",
            },
            host_config={
                "Binds": copy_binds,
                "NetworkMode": network_id,
            },
            user=user,
        )
        copy_container = copy["Id"]
        await _run_docker(client, "start_container", copy_container)
        total_bytes = await _image_total_bytes(
            client, copy_container, source_ref, arch, has_authfile,
        )
        progress_path = staging / PROGRESS_NAME
        current = -1
        info: dict = {}
        while True:
            counter = await asyncio.to_thread(_read_counter, progress_path)
            if counter != current:
                current = counter
                detail: dict = {"current": counter}
                if total_bytes > 0:
                    detail["total"] = min(counter, total_bytes)
                yield {
                    "id": repo_tag.rsplit("/", 1)[-1],
                    "status": "Downloading",
                    "progressDetail": detail,
                }
            try:
                info = await _run_docker(client, "inspect_container", copy_container)
            except Exception:
                info = {}
            if isinstance(info, dict) and info and not (
                (info.get("State") or {}).get("Running", False)
            ):
                break
            proxy_info = await _run_docker(client, "inspect_container", proxy_container)
            if not ((proxy_info or {}).get("State") or {}).get("Running", False):
                raise ThrottledPullError(
                    "the bandwidth limiter exited during the pull"
                )
            await asyncio.sleep(0.5)
        exit_code = (info.get("State") or {}).get("ExitCode") if isinstance(info, dict) else None
        if exit_code != 0:
            detail = ""
            try:
                logs = await _run_docker(
                    client, "container_logs", copy_container,
                    stderr=True, stdout=False, tail=12,
                )
                if isinstance(logs, bytes):
                    detail = logs.decode("utf-8", errors="replace")
                else:
                    detail = str(logs or "")
                detail = " | ".join(
                    line.strip() for line in detail.strip().splitlines()[-6:] if line.strip()
                )
            except Exception:
                pass
            raise ThrottledPullError(
                "the bandwidth-limited image copy failed"
                + (f": {detail}" if detail else "")
            )
        archive = staging / "image.tar"
        if not archive.is_file():
            raise ThrottledPullError("the bandwidth-limited pull produced no image archive")
        yield {"status": "Loading image into Docker"}

        def load() -> None:
            errors: list[str] = []
            with open(archive, "rb") as handle:
                # docker-py returns the daemon's streamed progress/error
                # events; the archive may still fail to load despite the
                # HTTP call answering, so the stream must be consumed.
                for event in client.api.load_image(handle):
                    if isinstance(event, dict) and event.get("error"):
                        errors.append(str(event["error"]))
            if errors:
                raise ThrottledPullError(
                    "the image archive failed to load: " + "; ".join(errors[:3])
                )

        await loop.run_in_executor(None, load)
        try:
            await _run_docker(client, "inspect_image", repo_tag)
        except Exception as exc:
            raise ThrottledPullError(
                f"the loaded image does not register as '{repo_tag}'"
            ) from exc
        yield {"status": f"Successfully pulled {image}"}
    except Exception as exc:
        yield {"error": str(exc) or type(exc).__name__}
    finally:
        for container in (copy_container, proxy_container):
            if container is not None:
                try:
                    await _run_docker(
                        client, "remove_container", container, v=True, force=True,
                    )
                except Exception:
                    pass
        if network_id is not None:
            try:
                await _run_docker(client, "remove_network", network_id)
            except Exception:
                pass

        def cleanup_staging() -> None:
            shutil.rmtree(staging, ignore_errors=True)
            if auth_dir is not None:
                shutil.rmtree(auth_dir, ignore_errors=True)

        await loop.run_in_executor(None, cleanup_staging)
