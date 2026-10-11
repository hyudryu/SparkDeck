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

The archive is staged on the host filesystem. ``SPARKDECK_IMAGE_STAGING_DIR``
selects the parent directory (the system temp dir by default), and before any
bytes move the image's compressed size is fetched with ``skopeo inspect`` so
the staging filesystem's free space can be checked against it.

When no cap is configured the normal daemon-side ``docker pull`` is used and
nothing here runs. Loopback registries are never routed through here either
(see :func:`is_loopback_registry_image`). The first capped pull fetches the
two helper images with an uncapped daemon pull; they are small and fetched
once per node.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
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
import base64, os, socket, threading, time

RATE = float(os.environ["SPARKDECK_DOWNLOAD_BPS"])
PROGRESS = os.environ.get("SPARKDECK_BANDWIDTH_PROGRESS") or ""
UPSTREAM = os.environ.get("SPARKDECK_UPSTREAM_PROXY") or ""
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

def split_authority(value):
    # Bracketed IPv6 authorities like [2001:db8::1]:5000.
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        return host, rest.lstrip(":")
    host, _, port = value.partition(":")
    return host, port

def parse_upstream(url):
    if "://" in url:
        url = url.split("://", 1)[1]
    auth = ""
    if "@" in url:
        auth, _, url = url.rpartition("@")
    host, port = split_authority(url)
    return auth, host, int(port or "80")

def connect_target(authority):
    # Without an upstream, dial the registry directly; with one (corporate
    # proxy-only hosts), send the CONNECT to the proxy and chain through.
    if not UPSTREAM:
        host, port = split_authority(authority)
        return socket.create_connection((host, int(port or "443")), 10)
    auth, host, port = parse_upstream(UPSTREAM)
    sock = socket.create_connection((host, port), 10)
    request = "CONNECT %%s HTTP/1.1\\r\\nHost: %%s\\r\\n" %% (authority, authority)
    if auth:
        token = base64.b64encode(auth.encode("latin-1")).decode("ascii")
        request += "Proxy-Authorization: Basic %%s\\r\\n" %% token
    sock.sendall((request + "\\r\\n").encode("latin-1"))
    response = b""
    while b"\\r\\n\\r\\n" not in response and len(response) < 65536:
        chunk = sock.recv(4096)
        if not chunk:
            sock.close()
            raise OSError("upstream proxy closed the connection")
        response += chunk
    status = response.partition(b"\\r\\n")[0]
    if b" 200" not in status:
        sock.close()
        raise OSError("upstream proxy refused CONNECT: " + status.decode("latin-1"))
    return sock

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
        upstream = connect_target(parts[1])
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
listener.bind(("0.0.0.0", int(os.environ.get("SPARKDECK_PROXY_PORT") or %d)))
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


_LOOPBACK_HOSTS = frozenset({"localhost", "::1", "ip6-localhost", "ip6-loopback"})


def registry_head(image: str) -> str | None:
    """Lower-cased ``host[:port]`` of a qualified reference, else None."""
    reference = str(image or "").strip().removeprefix("docker://")
    head, slash, _rest = reference.partition("/")
    if not slash:
        return None
    head = head.lower()
    if "@" in head:  # userinfo is never a registry name
        return None
    if not ("." in head or ":" in head or head == "localhost"):
        return None
    return head


def split_authority(authority: str) -> tuple[str, str]:
    """Split ``host:port``, including bracketed IPv6 forms."""
    value = str(authority or "")
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        return host, rest.lstrip(":")
    host, _, port = value.partition(":")
    return host, port


def matches_insecure_registry(image: str, insecure: list[str]) -> bool:
    """True when the image's registry matches a daemon insecure entry.

    Docker's ``InsecureRegistryCIDRs`` mixes CIDR networks with plain
    ``host[:port]`` entries; a hit means only the daemon (which holds the
    trust configuration and locally installed CA) can reach the registry,
    so the capped helper must not take over the pull.
    """
    head = registry_head(image)
    if not head:
        return False
    host, port = split_authority(head)
    if head.startswith("["):
        bare = "[" + host + "]"
    else:
        bare = head.partition(":")[0]
    for entry in insecure:
        candidate = str(entry or "").strip().lower()
        if not candidate:
            continue
        if "/" in candidate:
            try:
                network = ipaddress.ip_network(candidate, strict=False)
                if ipaddress.ip_address(host) in network:
                    return True
            except ValueError:
                pass
            continue
        if candidate == head or candidate == bare:
            return True
        entry_host, entry_port = split_authority(candidate)
        if entry_port and not port and entry_host == host:
            return True
    return False


def is_loopback_registry_image(image: str) -> bool:
    """True when the reference's registry host is this machine's loopback.

    A local registry is not an internet download, and the capped path runs
    inside a container network where loopback reaches the container itself,
    not the host. Callers pull such references with the daemon instead.
    """
    head = registry_head(image)
    if not head:
        return False
    host, _ = split_authority(head)
    host = host.rstrip(".")
    if host in _LOOPBACK_HOSTS or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def staging_parent() -> str | None:
    """Parent directory for pull staging; ``SPARKDECK_IMAGE_STAGING_DIR``."""
    configured = str(os.environ.get("SPARKDECK_IMAGE_STAGING_DIR") or "").strip()
    return configured or None


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


async def ensure_helper_images(client: Any) -> None:
    """Pre-install the two helper images the capped pull flow needs.

    The throttle itself runs inside these containers, so their bootstrap
    daemon pull can never be rate limited. Enabling a download cap
    pre-installs them immediately, turning that uncapped step into a
    one-time action at settings-save time instead of the first user pull.
    """
    for image in (PROXY_IMAGE, SKOPEO_IMAGE):
        await _ensure_image(client, image)


async def _exec_ok(client: Any, container: str, command: list[str]) -> bool:
    exec_id = await _run_docker(client, "exec_create", container, cmd=command)
    await _run_docker(client, "exec_start", exec_id["Id"], demux=False)
    result = await _run_docker(client, "exec_inspect", exec_id["Id"])
    return isinstance(result, dict) and result.get("ExitCode") == 0


async def _image_total_bytes(
    client: Any, network_id: str, source_ref: str,
    arch: str, copy_binds: list[str],
) -> int:
    """Fetch the image's compressed size via a ``skopeo inspect`` preflight.

    Runs one short-lived skopeo container before the copy starts so the
    caller can capacity-check the staging filesystem before any bytes move.
    The progress total and the capacity check are both a convenience: every
    failure path returns 0 (unknown size) and lets the copy itself surface
    real errors.
    """
    command = ["inspect", "--tls-verify=true"]
    if arch:
        command += ["--override-os", "linux", "--override-arch", arch]
    has_authfile = any(b.endswith(":/auth/config.json:ro") for b in copy_binds)
    if has_authfile:
        command += ["--authfile", "/auth/config.json"]
    command.append(source_ref)
    container: str | None = None
    try:
        created = await _run_docker(
            client, "create_container",
            image=SKOPEO_IMAGE,
            command=command,
            environment={"HOME": "/tmp"},
            host_config={
                "Binds": [b for b in copy_binds if ":/auth/config.json" in b],
                "NetworkMode": network_id,
            },
        )
        container = created["Id"]
        await _run_docker(client, "start_container", container)
        info: dict = {}
        for _ in range(120):
            info = await _run_docker(client, "inspect_container", container)
            if not (info.get("State") or {}).get("Running", False):
                break
            await asyncio.sleep(0.5)
        if (info.get("State") or {}).get("ExitCode") != 0:
            return 0
        output = await _run_docker(
            client, "container_logs", container, stdout=True, stderr=False,
        )
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        payload = json.loads(str(output or ""))
        if not isinstance(payload, dict):
            return 0
        # Registry manifests report the archive size per layer; the
        # top-level Size only exists for schema 2 / docker-archive
        # sources, so LayersData is the primary source.
        layers = payload.get("LayersData")
        if isinstance(layers, list):
            total = sum(
                entry["Size"] for entry in layers
                if isinstance(entry, dict) and isinstance(entry.get("Size"), int)
                and not isinstance(entry.get("Size"), bool)
            )
            if total > 0:
                return total
        size = payload.get("Size")
        if isinstance(size, int) and not isinstance(size, bool):
            return max(0, size)
        return 0
    except Exception:
        return 0
    finally:
        if container is not None:
            try:
                await _run_docker(
                    client, "remove_container", container, v=True, force=True,
                )
            except Exception:
                pass
    return 0


def staging_capacity_error(staging: Path, total_bytes: int) -> str | None:
    """Error text when the staging filesystem cannot hold ``total_bytes``.

    Unknown totals (0) cannot be checked and are let through: the copy then
    fails on its own if the filesystem actually runs out.
    """
    if total_bytes <= 0:
        return None
    try:
        free = shutil.disk_usage(staging).free
    except OSError:
        return None
    required = total_bytes + total_bytes // 8 + 512 * 1024 * 1024
    if free >= required:
        return None
    return (
        f"the pull staging directory ({staging.parent}) has "
        f"{free / 1e9:.1f} GB free but this image needs about "
        f"{total_bytes / 1e9:.1f} GB; set SPARKDECK_IMAGE_STAGING_DIR to a "
        "larger filesystem, or set the download limit to 0 to pull without "
        "staging"
    )


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
    staging = Path(tempfile.mkdtemp(
        prefix="sparkdeck-pull-", dir=staging_parent(),
    ))
    try:
        yield {"status": f"Preparing bandwidth-limited pull of {image}"}
        version = await _run_docker(client, "version")
        arch = str((version or {}).get("Arch") or "")
        for helper in (PROXY_IMAGE, SKOPEO_IMAGE):
            try:
                await _run_docker(client, "inspect_image", helper)
            except Exception:
                # Chicken-and-egg: the limiter lives inside these helpers, so
                # their own daemon pull cannot be capped. Settings-save
                # pre-installs them; surfacing the event keeps the exception
                # visible instead of silently overrunning the cap.
                yield {"status": (
                    f"Bootstrapping pull helper {helper}; this one-time "
                    "daemon pull is not covered by the bandwidth limit"
                )}
                await _run_docker(client, "pull", helper)
        network = await _run_docker(
            client, "create_network", name=network_name, driver="bridge",
        )
        network_id = str(network["Id"])
        registry_host = source_ref.removeprefix("docker://").partition("/")[0]
        # Same candidate order as docker-py: $DOCKER_CONFIG first, then the
        # root and user home defaults the SDK falls back to.
        docker_config_dir = os.environ.get("DOCKER_CONFIG") or ""
        authfile = select_docker_config([
            os.path.join(docker_config_dir, "config.json") if docker_config_dir else "",
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
                # Proxy-only hosts reach registries via the corporate proxy;
                # the limiter chains CONNECT through it when configured.
                "SPARKDECK_UPSTREAM_PROXY": os.environ.get(
                    "HTTPS_PROXY", os.environ.get("https_proxy", ""),
                ),
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
        # Preflight the compressed size before the archive can start filling
        # the staging filesystem, then capacity-check before bytes move.
        total_bytes = await _image_total_bytes(
            client, network_id, source_ref, arch, copy_binds,
        )
        capacity = staging_capacity_error(staging, total_bytes)
        if capacity:
            raise ThrottledPullError(capacity)
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
        progress_path = staging / PROGRESS_NAME
        current = -1
        info: dict = {}
        while True:
            counter = await asyncio.to_thread(_read_counter, progress_path)
            if counter != current:
                current = counter
                # The total is fixed from the preflight inspect; the counter
                # is what clamps, since manifest requests make it overshoot.
                if total_bytes > 0:
                    detail = {
                        "current": min(counter, total_bytes),
                        "total": total_bytes,
                    }
                else:
                    detail = {"current": counter}
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
