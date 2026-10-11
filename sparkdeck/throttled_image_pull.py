"""Bandwidth-capped Docker image pulls.

``docker pull`` fetches registry bytes inside the Docker daemon, outside any
process SparkDeck owns, so a user-space rate limit cannot be applied to it.
When a max download bandwidth is configured, the pull instead runs through
two helper containers on an isolated Docker network:

* a small token-bucket HTTPS CONNECT proxy (python slim image), and
* ``skopeo`` copying the image into a docker-archive tarball through it.

The tarball is then loaded with ``docker load``. Authentication mirrors what
an unauthenticated SDK pull already does: the daemon-side user's
``.docker/config.json`` is mounted read-only for skopeo. Digest-verified
copying, archive assembly, and layer compression handling are skopeo's.

When no cap is configured the normal daemon-side ``docker pull`` is used and
nothing here runs. The first capped pull fetches the two helper images with
an uncapped daemon pull; they are small and fetched once per node.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

PROXY_IMAGE = "python:3.12-slim"
SKOPEO_IMAGE = "quay.io/skopeo/stable:latest"
_PROXY_PORT = 8899

# The token bucket mirrors sparkdeck.download_throttle's virtual-time
# schedule, but it runs inside the proxy container where no project module is
# importable. Only the download direction (registry → client) is charged.
_PROXY_SCRIPT = """
import os, socket, threading, time

RATE = float(os.environ["SPARKDECK_DOWNLOAD_BPS"])
DUE = [0.0]
LOCK = threading.Lock()

def spend(n):
    if n <= 0:
        return
    with LOCK:
        now = time.monotonic()
        due = max(now, DUE[0]) + n / RATE
        DUE[0] = due
        wait = due - now
    if wait > 0:
        time.sleep(wait)

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
    """Return (docker:// source ref, docker-archive ref) for one image.

    Unqualified references get the Docker Hub expansion that ``docker pull``
    performs, and a missing tag gets ``latest``. A digest reference carries no
    repository tag into the loaded image, exactly like ``docker pull``.
    """
    reference = str(image or "").strip()
    if not reference or any(char.isspace() for char in reference):
        raise ThrottledPullError("image reference is not valid")
    if reference.startswith("docker://"):
        raise ThrottledPullError("image reference must not include a transport prefix")
    path = reference
    digest = ""
    if "@" in path:
        path, digest = path.split("@", 1)
        if not digest:
            raise ThrottledPullError("image reference is not valid")
    first, _, remainder = path.partition("/")
    if not remainder:
        first, remainder = path, ""
    if first and "." not in first and ":" not in first and first != "localhost":
        path = (
            f"docker.io/library/{path}" if "/" not in path
            else f"docker.io/{path}"
        )
    if digest:
        return f"docker://{path}@{digest}", "docker-archive:/archive/image.tar"
    if ":" not in path.rsplit("/", 1)[-1]:
        path = f"{path}:latest"
    return f"docker://{path}", f"docker-archive:/archive/image.tar:{path}"


def skopeo_copy_args(
    source_ref: str, archive_ref: str, arch: str, *, has_authfile: bool,
) -> list[str]:
    """Build skopeo argv (without the entrypoint) for one capped copy."""
    args = ["copy", "--src-tls-verify=true", "--format-event-data",
            "--event-file-path", "/archive/events.jsonl",
            "--override-os", "linux"]
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


def _read_event_lines(path: Path, offset: int) -> tuple[list[dict], int]:
    """Read newline-delimited skopeo events appended since ``offset``."""
    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size <= offset:
        return [], offset
    with open(path, "rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    last_newline = chunk.rfind(b"\n")
    if last_newline == -1:
        # No complete line yet; keep the offset for the next read.
        return [], offset
    events: list[dict] = []
    for line in chunk[:last_newline + 1].split(b"\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, offset + last_newline + 1


def _progress_event(event: dict) -> dict | None:
    """Map one skopeo progress event to a docker-pull-style progress dict."""
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    kind = str(event.get("type") or "")
    layer = str(data.get("id") or "")[:19]
    if kind == "full":
        total = data.get("bytes")
        if isinstance(total, (int, float)) and total > 0:
            return {"id": layer, "status": "Downloading",
                    "progressDetail": {"current": 0, "total": int(total)}}
        return None
    if kind in ("progress", "start"):
        offset = data.get("offset")
        if isinstance(offset, (int, float)) and offset > 0:
            return {"id": layer, "status": "Downloading",
                    "progressDetail": {"current": int(offset)}}
    return None


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


async def throttled_image_pull_events(
    client: Any, image: str, bytes_per_second: float,
) -> AsyncIterator[dict]:
    """Yield docker-pull-shaped progress dicts for one bandwidth-capped pull.

    Raises nothing: a failure is emitted as ``{"error": ...}``, matching the
    event stream contract of ``manager.pull_image_stream``.
    """
    loop = asyncio.get_running_loop()
    try:
        source_ref, archive_ref = normalize_image_reference(image)
    except ThrottledPullError as exc:
        yield {"error": str(exc)}
        return
    network_id: str | None = None
    suffix = uuid.uuid4().hex[:8]
    network_name = f"sparkdeck-pull-{suffix}"
    proxy_name = f"sparkdeck-pull-{suffix}-proxy"
    proxy_container: str | None = None
    copy_container: str | None = None
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
        registry_host = (
            source_ref.removeprefix("docker://").partition("/")[0]
        )
        authfile = select_docker_config([
            "/root/.docker/config.json",
            os.path.expanduser("~/.docker/config.json"),
        ])
        binds = [f"{staging}:/archive"]
        if authfile:
            try:
                with open(authfile, encoding="utf-8") as handle:
                    docker_config = json.load(handle)
            except (OSError, json.JSONDecodeError):
                docker_config = {}
            creds_store = str(docker_config.get("credsStore") or "")
            auth_entry = (docker_config.get("auths") or {}).get(registry_host) or {}
            if creds_store and not auth_entry.get("auth"):
                raise ThrottledPullError(
                    f"registry credentials for {registry_host} come from the"
                    " credential helper, which cannot run inside the"
                    " bandwidth-limited pull helper; store a docker login"
                    " entry in config.json or disable the bandwidth cap"
                )
            binds.append(f"{authfile}:/auth/config.json:ro")
        proxy = await _run_docker(
            client, "create_container",
            image=PROXY_IMAGE,
            name=proxy_name,
            command=["python", "-u", "-c", _PROXY_SCRIPT],
            environment={
                "SPARKDECK_DOWNLOAD_BPS": repr(float(bytes_per_second)),
                "PYTHONUNBUFFERED": "1",
            },
            host_config={"NetworkMode": network_id},
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
                source_ref, archive_ref, arch, has_authfile=authfile is not None,
            ),
            environment={
                "HTTPS_PROXY": f"http://{proxy_name}:{_PROXY_PORT}",
                "HTTP_PROXY": f"http://{proxy_name}:{_PROXY_PORT}",
                "NO_PROXY": "localhost,127.0.0.1",
                "HOME": "/tmp",
            },
            host_config={
                "Binds": binds,
                "NetworkMode": network_id,
            },
            user=user,
        )
        copy_container = copy["Id"]
        await _run_docker(client, "start_container", copy_container)
        events_path = staging / "events.jsonl"
        offset = 0
        info: dict = {}
        while True:
            events, offset = await asyncio.to_thread(
                _read_event_lines, events_path, offset,
            )
            for event in events:
                mapped = _progress_event(event)
                if mapped is not None:
                    yield mapped
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
            with open(archive, "rb") as handle:
                client.api.load_image(handle)

        await loop.run_in_executor(None, load)
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
            try:
                shutil.rmtree(staging, ignore_errors=True)
            except OSError:
                pass

        await loop.run_in_executor(None, cleanup_staging)
