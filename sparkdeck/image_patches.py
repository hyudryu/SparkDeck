"""Build and inspect file-only derived images without executing patch code."""

import hashlib
import io
import json
import re
import tarfile
import threading
import uuid
from pathlib import PurePosixPath

from docker.errors import ImageNotFound, NotFound

MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 4 * MAX_FILE_BYTES
_BUILD_LOCK = threading.Lock()
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
_BASE_IDENTITY = re.compile(r"runtime-v1:sha256:[0-9a-f]{64}")


def valid_base_image_identity(value):
    return isinstance(value, str) and _BASE_IDENTITY.fullmatch(value) is not None


def base_image_identity(image):
    """Fingerprint runnable contents, independent of Docker's storage backend.

    Classic Docker IDs hash image configuration; containerd IDs may instead
    identify a manifest or index. Both inspect APIs expose the selected
    platform's runtime configuration and ordered uncompressed layer digests.
    Deliberately exclude tags, creation timestamps, history, and store IDs.
    """
    attrs = image.attrs
    config = attrs.get("Config")
    rootfs = attrs.get("RootFS")
    if not isinstance(config, dict) or not isinstance(rootfs, dict):
        raise ValueError("Docker did not provide the base image configuration and filesystem identity")
    layers = rootfs.get("Layers")
    if (rootfs.get("Type") != "layers" or not isinstance(layers, list)
            or any(not isinstance(layer, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer) for layer in layers)):
        raise ValueError("Docker did not provide valid base image layer identities")
    architecture, os_name = attrs.get("Architecture"), attrs.get("Os")
    if not isinstance(architecture, str) or not architecture or os_name != "linux":
        raise ValueError("Docker did not provide a supported Linux base image platform")
    # Different API/store versions serialize absent default fields as null,
    # empty, or false. Normalize only top-level defaults: nested empty values
    # in Volumes or ExposedPorts are meaningful declarations and stay intact.
    config = {key: value for key, value in config.items()
              if value is not None and value is not False and value != "" and value != [] and value != {}}
    identity = {
        "config": config, "rootfs": {"type": "layers", "layers": layers},
        "platform": {"os": os_name, "architecture": architecture,
                     "variant": attrs.get("Variant") or "", "os_version": attrs.get("OsVersion") or "",
                     "os_features": attrs.get("OsFeatures") or []},
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return "runtime-v1:sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_patch_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("Patch request must be an object")
    if set(payload) - {"base_image", "image", "files", "expected_base_id", "expected_base_identity"}:
        raise ValueError("Unsupported patch request fields")
    base = payload.get("base_image")
    output = payload.get("image")
    for name, value in (("Base image", base), ("Output image", output)):
        if not isinstance(value, str) or len(value) > 255 or not _IMAGE.fullmatch(value):
            raise ValueError(f"{name} must be a valid image reference")
    if "@" in output or ":" not in output.rsplit("/", 1)[-1]:
        raise ValueError("Output image must include an explicit tag")
    repository, tag = output.rsplit(":", 1)
    if not repository or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", tag):
        raise ValueError("Output image must include a valid tag")
    if output == base or output == base + ":latest":
        raise ValueError("Output image must differ from the base image")
    files = payload.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 16:
        raise ValueError("Provide between 1 and 16 patch files")
    normalized = []
    total = 0
    targets = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"target", "content"}:
            raise ValueError("Each patch file needs target and content")
        target, content = item["target"], item["content"]
        if (not isinstance(target, str) or len(target) > 4096
                or not target.startswith("/") or target.startswith("//")
                or target == "/" or str(PurePosixPath(target)) != target
                or ".." in target.split("/") or "\\" in target
                or any(ord(char) < 32 or ord(char) == 127 for char in target)):
            raise ValueError("Patch targets must be normalized absolute Linux file paths")
        if target.split("/")[1] in {"proc", "sys", "dev"}:
            raise ValueError("Patch targets cannot use /proc, /sys, or /dev")
        if any(target == old or target.startswith(old + "/") or old.startswith(target + "/") for old in targets):
            raise ValueError("Patch file targets cannot duplicate or overlap")
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("Patch content must be UTF-8 text without NUL characters")
        try:
            size = len(content.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("Patch content must be valid UTF-8 text") from exc
        if size > MAX_FILE_BYTES:
            raise ValueError("Each patch file must be at most 1 MiB")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Patch files must total at most 4 MiB")
        normalized.append({"target": target, "content": content})
        targets.append(target)
    result = {"base_image": base, "image": output, "files": normalized}
    if "expected_base_id" in payload:
        expected = payload["expected_base_id"]
        if not isinstance(expected, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
            raise ValueError("Expected base image ID must be a SHA-256 image ID")
        result["expected_base_id"] = expected
    if "expected_base_identity" in payload:
        if not valid_base_image_identity(payload["expected_base_identity"]):
            raise ValueError("Expected base image identity must be a versioned runtime fingerprint")
        result["expected_base_identity"] = payload["expected_base_identity"]
    return result


def _exists(client, reference):
    try:
        client.images.get(reference)
        return True
    except ImageNotFound:
        return False


def _context(base_id, files):
    context = io.BytesIO()
    lines = [f"FROM {base_id}"]
    with tarfile.open(fileobj=context, mode="w") as archive:
        for index, item in enumerate(files):
            source = f"patch-{index}"
            data = item["content"].encode("utf-8")
            entry = tarfile.TarInfo(source)
            entry.mode = 0o644
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
            lines.append("COPY " + json.dumps([source, item["target"]], ensure_ascii=True))
        dockerfile = ("\n".join(lines) + "\n").encode("utf-8")
        entry = tarfile.TarInfo("Dockerfile")
        entry.size = len(dockerfile)
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(dockerfile))
    context.seek(0)
    return context


def _verify(client, image_id, files, on_log):
    container = client.containers.create(image_id, command=["/bin/true"], network_disabled=True)
    hashes = []
    try:
        for item in files:
            chunks, _ = container.get_archive(item["target"])
            buffer = io.BytesIO()
            for chunk in chunks:
                if buffer.tell() + len(chunk) > MAX_FILE_BYTES + 128 * 1024:
                    raise ValueError("Built patch file exceeds verification size limit")
                buffer.write(chunk)
            buffer.seek(0)
            with tarfile.open(fileobj=buffer, mode="r:*") as archive:
                members = archive.getmembers()
                if len(members) != 1 or not members[0].isfile() or members[0].size > MAX_FILE_BYTES:
                    raise ValueError("Built patch target is not a regular file")
                data = archive.extractfile(members[0]).read(MAX_FILE_BYTES + 1)
            expected = item["content"].encode("utf-8")
            if data != expected:
                raise ValueError(f"Built patch content did not match: {item['target']}")
            hashes.append({"target": item["target"], "sha256": hashlib.sha256(data).hexdigest()})
    finally:
        try:
            container.remove(v=True, force=True)
        except NotFound:
            pass
        except Exception:
            on_log("Warning: Docker could not remove the stopped patch verification container; cleanup may be needed.")
    return hashes


def build_patched_image(client, payload, on_log=lambda message: None):
    """Build locally, verify in a stopped container, then publish a new tag.

    The lock serializes this process's builds, including the final tag check.
    Docker has no atomic create-if-absent tag API; external Docker writers must
    not concurrently publish the same output tag.
    """
    request = normalize_patch_request(payload)
    with _BUILD_LOCK:
        output = request["image"]
        if _exists(client, output):
            raise ValueError("Output image already exists; choose a new tag")
        try:
            base = client.images.get(request["base_image"])
        except ImageNotFound:
            on_log("Pulling base image…")
            base = client.images.pull(request["base_image"])
        if base.attrs.get("Config", {}).get("OnBuild"):
            raise ValueError("Base images with ONBUILD commands are not supported")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", base.id):
            raise ValueError("Docker returned an invalid base image ID")
        if request.get("expected_base_id", base.id) != base.id:
            raise ValueError("Base image differs from the first node; use the same base image on every node")
        identity = base_image_identity(base)
        if request.get("expected_base_identity", identity) != identity:
            raise ValueError("Base image runtime configuration, filesystem, or platform differs from the first node")
        temporary_tag = "sparkdeck-patch-build:" + uuid.uuid4().hex
        context = _context(base.id, request["files"])
        try:
            on_log("Building file-only image…")
            for event in client.api.build(fileobj=context, custom_context=True,
                                          tag=temporary_tag, rm=True, forcerm=True,
                                          pull=False, decode=True, network_mode="none"):
                if event.get("error") or event.get("errorDetail"):
                    raise ValueError(event.get("error") or event["errorDetail"].get("message", "Image build failed"))
                if event.get("stream"):
                    on_log(event["stream"])
            built = client.images.get(temporary_tag)
            on_log("Verifying patch file contents without starting the container…")
            hashes = _verify(client, built.id, request["files"], on_log)
            if _exists(client, output):
                raise ValueError("Output image already exists; choose a new tag")
            repository, tag = output.rsplit(":", 1)
            if not built.tag(repository, tag=tag, force=False):
                raise ValueError("Docker could not tag the patched image")
            return {"image_id": built.id, "base_id": base.id, "base_identity": identity,
                    "image": output, "files": hashes}
        finally:
            context.close()
            try:
                client.images.remove(temporary_tag, noprune=True)
            except ImageNotFound:
                pass
            except Exception:
                on_log(f"Warning: Docker could not remove temporary image tag {temporary_tag}; cleanup may be needed.")
