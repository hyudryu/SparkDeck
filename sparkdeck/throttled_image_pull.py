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
# Cap file the proxy re-reads while it runs (bytes per second, 0 = unlimited).
# It lives in staging (bind-mounted at /archive) so the host can rewrite it
# when the bandwidth setting changes mid-pull.
RATE_NAME = "rate.txt"

# Rate files of the capped pulls running in this process, mapped to the
# SHARE of the cluster budget each pull was dispatched with (1.0 for a
# single transfer; a multi-node fan-out gets cap/width per member). A
# settings change rewrites each file at new_cluster_cap * factor, so a
# fan-out of N members re-rates to N * new_cluster_cap/N = new_cluster_cap
# in aggregate instead of N times the full cluster value. Plain strings,
# not Path objects: staging paths are built under the caller's os.name
# (the uid handling patches it to exercise the Linux paths), and cleanup
# runs in an executor thread where a captured PosixPath could no longer be
# combined. Iterated and mutated only from the owning event loop plus the
# settings handler, both under the GIL.
_ACTIVE_RATE_FILES: dict[str, float] = {}


def apply_bandwidth_cap(bytes_per_second: float) -> None:
    """Re-rate the capped image pulls running in this process.

    The value is the NEW CLUSTER cap in bytes per second; each active pull
    takes its allocated share of it (factor 1.0 for single transfers).
    Pulls that started unlimited used the plain daemon-side pull, which no
    user-space limiter can touch; the newly enabled cap applies to them
    nothing and to every subsequent transfer entirely.
    """
    cluster = float(bytes_per_second)
    for path, factor in list(_ACTIVE_RATE_FILES.items()):
        value = repr(cluster * float(factor))
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(value)
            os.replace(tmp, path)
        except OSError:
            pass

# The token bucket mirrors sparkdeck.download_throttle's virtual-time
# schedule, but it runs inside the proxy container where no project module is
# importable. Only the download direction (registry → client) is charged,
# and the running total is reported for progress.
_PROXY_SCRIPT = """
import base64, ipaddress, os, socket, ssl, threading, time
from urllib.parse import unquote, urlparse

RATE = float(os.environ["SPARKDECK_DOWNLOAD_BPS"])
PROGRESS = os.environ.get("SPARKDECK_BANDWIDTH_PROGRESS") or ""
# A proxy set only in daemon.json never reaches this process's environment;
# the owner passes it as the SPARKDECK_DAEMON_* fallback. The process env
# wins so a host-level proxy policy keeps precedence over the daemon's.
UPSTREAM = (os.environ.get("SPARKDECK_UPSTREAM_PROXY")
            or os.environ.get("SPARKDECK_DAEMON_PROXY") or "")
UPSTREAM_CA = os.environ.get("SPARKDECK_UPSTREAM_PROXY_CA") or ""
NO_PROXY = (os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
            or os.environ.get("SPARKDECK_DAEMON_NO_PROXY") or "")
RATE_FILE = os.environ.get("SPARKDECK_BANDWIDTH_RATE_FILE") or ""
DUE = [0.0]
LOCK = threading.Lock()
TOTAL = [0]
LAST_REPORT = [0.0]
LAST_RATE_POLL = [0.0]

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

def poll_rate():
    # The owner rewrites the staged rate file when the bandwidth setting
    # changes, so a pull already in flight re-rates instead of finishing at
    # its dispatched rate; 0 or less in the file means unlimited.
    global RATE
    if not RATE_FILE:
        return
    now = time.monotonic()
    if now - LAST_RATE_POLL[0] < 0.5:
        return
    LAST_RATE_POLL[0] = now
    try:
        with open(RATE_FILE) as handle:
            value = float(handle.read().strip() or 0)
    except (OSError, ValueError):
        return
    with LOCK:
        RATE = value if value > 0 else float("inf")

def spend(n):
    if n <= 0:
        return
    poll_rate()
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
    # Parse the proxy setting as a URL, not by splitting on colons: valid
    # settings carry a path or trailing slash (http://proxy.corp:8080/), and
    # a naive authority split turns the port into "8080/".
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    secure = (parsed.scheme or "http").strip().lower() == "https"
    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    auth = ""
    if parsed.username:
        # Proxy URLs percent-encode userinfo delimiters (@ : %%); the Basic
        # credential must carry the decoded username and password.
        auth = unquote(parsed.username)
        if parsed.password:
            auth += ":" + unquote(parsed.password)
    return auth, host, int(port or ("443" if secure else "80")), secure

def no_proxy_direct(authority):
    # NO_PROXY is proxy policy, not an uplink exemption: the destination is
    # still capped, the limiter just dials it directly instead of sending
    # the CONNECT through the corporate proxy that excluded it.
    if not NO_PROXY:
        return False
    host = split_authority(authority)[0].strip().lower().rstrip(".")
    host_ip = None
    for raw in NO_PROXY.replace(",", " ").split():
        entry = (
            raw.strip().removeprefix("http://").removeprefix("https://")
            .rstrip("/")
        )
        if entry == "*":
            return True
        entry = split_authority(entry)[0].strip().lower().lstrip(".").rstrip(".")
        if "/" in entry:
            # A CIDR entry matches by subnet containment only. Suffix logic
            # cannot see a network ("10.0.0.0/8" matches 10.1.2.3, not any
            # "*.10.0.0.0/8" host), and a hostname never carries a slash, so
            # skipping the suffix branch for CIDR spelling is deliberate.
            if host_ip is None:
                try:
                    host_ip = ipaddress.ip_address(host)
                except ValueError:
                    continue
            try:
                network = ipaddress.ip_network(entry, strict=False)
            except ValueError:
                continue
            if host_ip.version == network.version and host_ip in network:
                return True
            continue
        if entry and (host == entry or host.endswith("." + entry)):
            return True
    return False

def upstream_context(host):
    # A TLS-protected corporate endpoint needs a real TLS handshake before
    # CONNECT; a plaintext handshake against a TLS listener just hangs. The
    # CA-file override covers proxies certified by a private CA, mirroring
    # SSL_CERT_FILE for the host's own proxy-aware clients.
    if UPSTREAM_CA:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(UPSTREAM_CA)
        return context
    return ssl.create_default_context()

def connect_target(authority):
    # Without an upstream, dial the registry directly; with one (corporate
    # proxy-only hosts), send the CONNECT to the proxy and chain through —
    # except for NO_PROXY hosts, which the proxy explicitly does not serve.
    if not UPSTREAM or no_proxy_direct(authority):
        host, port = split_authority(authority)
        return socket.create_connection((host, int(port or "443")), 10)
    auth, host, port, secure = parse_upstream(UPSTREAM)
    sock = socket.create_connection((host, port), 10)
    if secure:
        sock = upstream_context(host).wrap_socket(sock, server_hostname=host)
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
    ``host[:port]`` entries. A hit is a TRANSPORT TRUST setting (talk to
    this registry without TLS), not a statement about uplink locality, so
    the capped helper keeps the pull and mirrors the setting by turning
    TLS verification off; only loopback registries are provably off the
    uplink and exempt from throttling.
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
    tls_verify: bool = True,
) -> list[str]:
    """Build skopeo argv (without the entrypoint) for one capped copy.

    ``tls_verify`` follows the daemon's insecure-registries policy for this
    registry: an insecure designation is transport trust, not an uplink
    exemption, so the capped path mirrors the daemon instead of falling
    back to the unthrottled ``docker pull``.
    """
    verify = "true" if tls_verify else "false"
    args = ["copy", "--src-tls-verify=" + verify, "--override-os", "linux"]
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


# Registry certificate directories, in Docker's own lookup order. A registry
# whose CA lives here (a private secure registry) verifies fine through the
# daemon but fails inside the isolated skopeo helper unless the same
# directory is mounted into it.
DOCKER_CERTS_D_BASES = ("/etc/docker/certs.d", "~/.docker/certs.d")
CONTAINERS_CERTS_D = "/etc/containers/certs.d"


def registry_ca_binds(
    registry_host: str, certs_staging: Path | str,
    bases: list[str] | None = None,
) -> list[str]:
    """Colon-safe exposure of the daemon's per-registry CA to the helpers.

    ``containers/image`` (skopeo) reads ``/etc/containers/certs.d/<registry>``
    exactly like dockerd reads ``/etc/docker/certs.d/<registry>``, so the
    certificates are copied into ``<certs_staging>/<registry key>`` and one
    bind mounts the staging parent at ``/etc/containers/certs.d``. The copy
    is not a nicety: Docker bind specs split on colons, so a port-qualified
    registry (``reg.corp:5443``) in the source or target path would be parsed
    as extra fields and fail container creation. Copying also lets the
    port-less directory spelling the operator may have created land at the
    key skopeo actually looks up. Only this registry's certificates are
    copied.
    """
    if not registry_host or not str(certs_staging):
        return []
    keys = [registry_host]
    host, _, port = registry_host.partition(":")
    if port:
        keys.append(host)
    for base in bases if bases is not None else DOCKER_CERTS_D_BASES:
        base = os.path.expanduser(str(base))
        for key in keys:
            directory = os.path.join(base, key)
            try:
                if not os.path.isdir(directory):
                    continue
                # containers/image reads the same per-registry directory the
                # daemon does: CA .crt files plus the mutual-TLS identity
                # under the exact names client.cert/client.key. Copying only
                # the CA would let a TLS-authenticated registry reject the
                # helper's handshake with no client certificate.
                names = [
                    name for name in os.listdir(directory)
                    if name.lower().endswith(".crt")
                    or name in ("client.cert", "client.key")
                ]
                if not names:
                    continue
                destination = Path(certs_staging) / registry_host
                destination.mkdir(parents=True, exist_ok=True)
                for name in names:
                    shutil.copy2(
                        os.path.join(directory, name), destination / name,
                    )
            except OSError:
                continue
            return [f"{certs_staging}:{CONTAINERS_CERTS_D}:ro,z"]
    return []


_DOCKER_HUB_HOSTS = {"docker.io", "index.docker.io", "registry-1.docker.io"}


def registry_mirrors_config(daemon_info: dict, registry_host: str) -> str:
    """Registries.conf.d drop-in text replaying the daemon's Hub mirrors.

    An isolated skopeo helper talks to the upstream registry directly and
    ignores dockerd's ``registry-mirrors``. On a mirror-only LAN that is a
    hard failure, and elsewhere it silently skips the internal pull-through
    cache the daemon would have used, so the daemon's Hub mirrors are
    re-expressed in the containers/image syntax the helper reads. Returns
    "" when no mirrors apply (not a Hub pull, or the daemon has none).
    """
    if str(registry_host or "").strip().lower() not in _DOCKER_HUB_HOSTS:
        return ""
    registry_config = (daemon_info or {}).get("RegistryConfig") or {}
    lines = ['[[registry]]', 'prefix = "docker.io"', 'location = "docker.io"', ""]
    for mirror in registry_config.get("Mirrors") or []:
        value = str(mirror or "").strip().lower()
        for scheme in ("https://", "http://"):
            if value.startswith(scheme):
                value = value[len(scheme):]
        value = value.rstrip("/")
        if not value:
            continue
        lines += ["[[registry.mirror]]", 'location = "' + value + '"', ""]
    if len(lines) == 4:
        return ""
    return "\n".join(lines)


def daemon_proxy_config(daemon_info: dict) -> tuple[str, str]:
    """The daemon's own (proxy, no_proxy) settings, as host-env spellings.

    Proxy-only hosts often configure the proxy only in daemon.json, where
    no host process can see it: the SparkDeck process env is empty, and a
    limiter built from process env alone would try to dial registries the
    network only reaches through the proxy. Docker reports the daemon's
    proxy under RegistryConfig (older APIs) or a top-level Proxies object,
    with the same key spellings, so read both. HTTPS proxy wins where both
    are set, matching how the proxy-aware host clients treat registry
    (HTTPS) traffic.
    """
    sources = []
    info = daemon_info or {}
    registry_config = info.get("RegistryConfig")
    if isinstance(registry_config, dict):
        sources.append(registry_config)
    proxies = info.get("Proxies")
    if isinstance(proxies, dict):
        sources.append(proxies)
    proxy = ""
    for key in ("HTTPSProxy", "HttpsProxy", "HTTPProxy", "HttpProxy"):
        for source in sources:
            value = str(source.get(key) or "").strip()
            if value:
                proxy = value
                break
        if proxy:
            break
    entries: list[str] = []
    for source in sources:
        value = source.get("NoProxy")
        if isinstance(value, str):
            value = value.split(",")
        if isinstance(value, list):
            entries.extend(str(item).strip() for item in value if str(item).strip())
        if entries:
            break
    return proxy, ",".join(entries)


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
            # containers/image's Docker-config parser honors both ``auth``
            # and an identity-token login; requiring ``auth`` would run
            # skopeo anonymously for identitytoken-only logins that the
            # daemon itself can use.
            if entry.get("auth") or entry.get("identitytoken"):
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


def _preflight_binds(copy_binds: list[str]) -> list[str]:
    """The binds the inspect preflight needs: credentials and registry CAs.

    Same isolation as the copy container (no staging archive), but the
    private-CA directories must be present or ``--tls-verify=true`` fails.
    """
    return [
        bind for bind in copy_binds
        # Match the certs.d target at either spelling: the current
        # parent-mount bind (``...:/etc/containers/certs.d:ro,z``) and the
        # per-registry directory bind skopeo also honors. The
        # registries.conf.d drop-ins (Hub mirrors) are needed too: without
        # them the preflight inspects the Hub directly while the copy would
        # go through the mirror, disagreeing on reachability.
        if (
            ":/auth/config.json" in bind
            or f":{CONTAINERS_CERTS_D}" in bind
            or ":/etc/containers/registries.conf.d" in bind
        )
    ]


async def _image_total_bytes(
    client: Any, network_id: str, proxy_url: str, source_ref: str,
    arch: str, copy_binds: list[str], tls_verify: bool = True,
) -> int:
    """Fetch the image's compressed size via a ``skopeo inspect`` preflight.

    Runs one short-lived skopeo container before the copy starts so the
    caller can capacity-check the staging filesystem before any bytes move.
    It talks through the same limiter proxy as the copy: on proxy-only hosts
    a direct connection cannot reach the registry at all, and a direct one
    would also dodge the token bucket for the manifest traffic.
    The progress total and the capacity check are both a convenience: every
    failure path returns 0 (unknown size) and lets the copy itself surface
    real errors.
    """
    command = ["inspect", "--tls-verify=" + ("true" if tls_verify else "false")]
    if arch:
        command += ["--override-os", "linux", "--override-arch", arch]
    has_authfile = any(":/auth/config.json:ro" in b for b in copy_binds)
    if has_authfile:
        command += ["--authfile", "/auth/config.json"]
    command.append(source_ref)
    container: str | None = None
    try:
        created = await _run_docker(
            client, "create_container",
            image=SKOPEO_IMAGE,
            command=command,
            environment={
                "HOME": "/tmp",
                "HTTPS_PROXY": proxy_url,
                "HTTP_PROXY": proxy_url,
                "NO_PROXY": "localhost,127.0.0.1",
            },
            host_config={
                "Binds": _preflight_binds(copy_binds),
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
    # The docker-archive destination writes the layers DEcompressed
    # (containers/image's tarfile writer never recompresses), but the
    # registry only reports compressed layer sizes. Compiled GPU layers
    # commonly expand 2-3x under gzip, so the requirement is 3x the
    # compressed total plus 1 GB of headroom rather than a near-1.0 guess.
    required = total_bytes * 3 + 1024 * 1024 * 1024
    if free >= required:
        return None
    return (
        f"the pull staging directory ({staging.parent}) has "
        f"{free / 1e9:.1f} GB free but this image's decompressed archive "
        f"needs up to about {required / 1e9:.1f} GB; set "
        "SPARKDECK_IMAGE_STAGING_DIR to a larger filesystem, or set the "
        "download limit to 0 to pull without staging"
    )


async def throttled_image_pull_events(
    client: Any, image: str, bytes_per_second: float,
    cluster_bytes_per_second: float | None = None,
) -> AsyncIterator[dict]:
    """Yield docker-pull-shaped progress dicts for one bandwidth-capped pull.

    ``bytes_per_second`` is this pull's allocated rate. A cluster fan-out
    divides the settings cap across members, so it also passes the FULL
    cluster cap: a live re-rate then scales each member by its share
    (allocated/cluster) instead of writing the whole new cluster value to
    every member, which would multiply the aggregate by the member count.
    Without it (or 0) the pull is the only holder of the budget (factor 1).

    Raises nothing: a failure is emitted as ``{"error": ...}``, matching the
    event stream contract of ``manager.pull_image_stream``.
    """
    loop = asyncio.get_running_loop()
    share_factor = 1.0
    try:
        cluster_bps = float(cluster_bytes_per_second or 0)
    except (TypeError, ValueError):
        cluster_bps = 0.0
    if cluster_bps > 0 and float(bytes_per_second) > 0:
        share_factor = max(
            min(float(bytes_per_second) / cluster_bps, 1.0), 1e-9,
        )
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
    ca_dir: Path | None = None
    mirror_dir: Path | None = None
    staging = Path(tempfile.mkdtemp(
        prefix="sparkdeck-pull-", dir=staging_parent(),
    ))
    try:
        yield {"status": f"Preparing bandwidth-limited pull of {image}"}
        version = await _run_docker(client, "version")
        arch = str((version or {}).get("Arch") or "")
        # One info call drives the whole transport picture: rootless/userns
        # uid handling (below), the daemon's insecure-registries trust
        # setting, and its Hub registry-mirrors — the isolated helper
        # containers must reproduce all three or the capped path diverges
        # from the daemon's.
        daemon_info: dict = {}
        try:
            daemon_info = (await _run_docker(client, "info")) or {}
        except Exception:
            daemon_info = {}
        registry_config = daemon_info.get("RegistryConfig") or {}
        daemon_proxy, daemon_no_proxy = daemon_proxy_config(daemon_info)
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
        # An insecure-registries designation is transport trust, not uplink
        # locality, so this registry stays inside the limiter; the helpers
        # simply skip TLS verification exactly like the daemon would.
        tls_verify = not matches_insecure_registry(
            image,
            [str(e) for e in (registry_config.get("InsecureRegistryCIDRs") or [])],
        )
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
        # ":z" (shared) because the proxy and copy containers are different
        # containers that both write into staging; a private ":Z" label would
        # relabel away the peer's access on SELinux hosts and is ignored
        # everywhere else.
        copy_binds = [f"{staging}:/archive:z"]
        has_authfile = False
        # Grant the helpers only the one registry this pull needs: mounting
        # the original config.json would hand the skopeo containers every
        # unrelated registry's secret, unlike an ordinary daemon pull. The
        # materialized single-entry file is 0600 and removed with staging.
        registry_auth = _auth_entry(docker_config, registry_host)
        if registry_auth is None:
            registry_auth = helper_credentials(docker_config, registry_host)
        if registry_auth is not None:
            auth_dir = Path(tempfile.mkdtemp(prefix="sparkdeck-pull-auth-"))
            written = auth_dir / "config.json"
            written.write_text(
                json.dumps({"auths": {registry_host: registry_auth}}),
                encoding="utf-8",
            )
            try:
                os.chmod(written, 0o600)
            except OSError:
                pass
            has_authfile = True
            copy_binds.append(f"{written}:/auth/config.json:ro,z")
        # Private secure registries keep their CA in the daemon's certs.d,
        # which the isolated skopeo containers cannot see without a bind;
        # inherit it so the capped path verifies like the daemon does. The
        # copy target keeps the colon-named registry key OFF the bind spec.
        ca_dir = Path(tempfile.mkdtemp(prefix="sparkdeck-pull-cas-"))
        copy_binds += registry_ca_binds(registry_host, ca_dir)
        # An isolated skopeo ignores dockerd's registry-mirrors: on a
        # mirror-only LAN that is a hard failure, and elsewhere it skips the
        # internal pull-through cache. Replay the daemon's Hub mirrors as a
        # registries.conf.d drop-in; the parent-mount bind stays colon-free.
        mirror_text = registry_mirrors_config(daemon_info, registry_host)
        if mirror_text:
            mirror_dir = Path(tempfile.mkdtemp(prefix="sparkdeck-pull-mir-"))
            (mirror_dir / "sparkdeck.conf").write_text(
                mirror_text, encoding="utf-8",
            )
            copy_binds.append(
                f"{mirror_dir}:/etc/containers/registries.conf.d:ro,z",
            )
        # The proxy re-reads this file (at /archive/rate.txt) while it runs,
        # so a settings change re-rates the pull already in flight — at its
        # share of the cluster cap, recorded as the registration factor.
        # Plain string path: staging may be a PosixPath-shaped object when
        # the flow emulates a Linux host, and string joins stay valid
        # anywhere.
        rate_path = os.path.join(str(staging), RATE_NAME)
        with open(rate_path, "w", encoding="utf-8") as handle:
            handle.write(repr(float(bytes_per_second)))
        _ACTIVE_RATE_FILES[rate_path] = share_factor
        proxy_environment = {
            "SPARKDECK_DOWNLOAD_BPS": repr(float(bytes_per_second)),
            "SPARKDECK_BANDWIDTH_RATE_FILE": f"/archive/{RATE_NAME}",
            "SPARKDECK_BANDWIDTH_PROGRESS": f"/archive/{PROGRESS_NAME}",
            "PYTHONUNBUFFERED": "1",
            # Proxy-only hosts reach registries via the corporate proxy;
            # the limiter chains CONNECT through it when configured, and
            # honors NO_PROXY by dialing those hosts directly (still capped).
            "SPARKDECK_UPSTREAM_PROXY": os.environ.get(
                "HTTPS_PROXY", os.environ.get("https_proxy", ""),
            ),
            "NO_PROXY": os.environ.get(
                "NO_PROXY", os.environ.get("no_proxy", ""),
            ),
            # Hosts that configure the proxy only in daemon.json have an
            # empty process env above; the daemon's own settings are the
            # fallback inside the script (process env still wins).
            "SPARKDECK_DAEMON_PROXY": daemon_proxy,
            "SPARKDECK_DAEMON_NO_PROXY": daemon_no_proxy,
            "SPARKDECK_UPSTREAM_PROXY_CA": "",
        }
        proxy_binds = [f"{staging}:/archive:z"]
        proxy_ca = str(os.environ.get("SSL_CERT_FILE") or "")
        if proxy_ca.startswith("/") and os.path.isfile(proxy_ca):
            # The proxy image has no copy of the host trust store, so a
            # corporate HTTPS proxy on a private CA needs the host's
            # CA-file override, mounted at its own path inside.
            proxy_binds.append(f"{proxy_ca}:{proxy_ca}:ro,z")
            proxy_environment["SPARKDECK_UPSTREAM_PROXY_CA"] = proxy_ca
        proxy = await _run_docker(
            client, "create_container",
            image=PROXY_IMAGE,
            name=proxy_name,
            command=["python", "-u", "-c", _PROXY_SCRIPT],
            environment=proxy_environment,
            host_config={
                "Binds": proxy_binds,
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
            client, network_id, f"http://{proxy_name}:{_PROXY_PORT}",
            source_ref, arch, copy_binds, tls_verify=tls_verify,
        )
        capacity = staging_capacity_error(staging, total_bytes)
        if capacity:
            raise ThrottledPullError(capacity)
        user = None
        if os.name != "nt":
            # Running the copy as the SparkDeck host user keeps the archive
            # and progress files owned by it. Under rootless Docker the daemon
            # already runs as that user, so container root maps straight to it
            # while an explicit host UID maps to an unprivileged subuid that
            # cannot write the 0700 staging directory or read the 0600 auth
            # file; there the override must be omitted.
            security = daemon_info.get("SecurityOptions") or []
            options = [str(item) for item in security]
            # Daemon.Info reports remapping as the literal SecurityOptions
            # entries "name=rootless" and "name=name=userns" (the doubled
            # prefix is how moby emits the userns marker), so match exactly.
            if "name=rootless" in options:
                pass
            elif "name=name=userns" in options:
                # Rootful userns-remap: bind mounts keep host permissions the
                # remapped container UID cannot use, so the 0700 staging
                # directory and 0600 auth file are unusable no matter which
                # user the container runs as. Omitting the override would
                # merely replace a confusing denial, and making it work would
                # need world-writable staging and a world-readable credential
                # copy; refuse with the actionable knob instead.
                raise ThrottledPullError(
                    "bandwidth-limited pulls cannot run on a Docker daemon "
                    "with userns-remap; set the download bandwidth limit to "
                    "0 on this node"
                )
            else:
                user = f"{os.getuid()}:{os.getgid()}"
        copy = await _run_docker(
            client, "create_container",
            image=SKOPEO_IMAGE,
            command=skopeo_copy_args(
                source_ref,
                f"docker-archive:/archive/image.tar:{repo_tag}",
                arch, has_authfile=has_authfile, tls_verify=tls_verify,
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
        inspect_failures = 0
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
                inspect_failures = 0
            except Exception as exc:
                # A container removed externally (or a daemon that has
                # stopped answering for it) must end the wait, not spin:
                # this loop holds the node and cluster download budgets, so
                # every later capped transfer would block behind a phantom.
                info = {}
                inspect_failures += 1
                if inspect_failures >= 20:
                    raise ThrottledPullError(
                        "the bandwidth-limited copy container could no "
                        f"longer be inspected: {exc}"
                    ) from exc
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
            _ACTIVE_RATE_FILES.pop(os.path.join(str(staging), RATE_NAME), None)
            shutil.rmtree(staging, ignore_errors=True)
            if auth_dir is not None:
                shutil.rmtree(auth_dir, ignore_errors=True)
            if ca_dir is not None:
                shutil.rmtree(ca_dir, ignore_errors=True)
            if mirror_dir is not None:
                shutil.rmtree(mirror_dir, ignore_errors=True)

        await loop.run_in_executor(None, cleanup_staging)
