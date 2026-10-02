"""Private supervised Hugging Face writer. Credentials arrive through stdin."""

import json
import os
import signal
import sys
import threading

from sparkdeck.virtual_nas import _enable_hf_xet_high_performance, validate_model_id, validate_revision, _is_commit_sha


def _guard_parent_lifetime(parent_pid: int) -> None:
    """An abrupt agent exit must not leave an unowned cache writer."""
    import ctypes
    if sys.platform.startswith("linux"):
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise RuntimeError("could not establish download worker parent-death guard")
        # Close the race between the parent exit and installing PDEATHSIG.
        if os.getppid() != parent_pid:
            raise RuntimeError("download worker parent already exited")
    elif os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        handle = kernel.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE
        if not handle:
            raise RuntimeError("download worker parent is unavailable")

        def watch_parent():
            kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            os._exit(1)

        threading.Thread(target=watch_parent, daemon=True).start()
    else:
        raise RuntimeError("supervised model downloads require Linux or Windows")


def main() -> None:
    payload = json.load(sys.stdin)
    _guard_parent_lifetime(payload["parent_pid"])
    model_id = validate_model_id(payload["model_id"])
    revision = validate_revision(payload["revision"])
    if not _is_commit_sha(revision):
        raise ValueError("download revision must be immutable")
    _enable_hf_xet_high_performance()
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, revision=revision,
                      cache_dir=payload["cache_dir"], token=payload["token"] or None)


if __name__ == "__main__":
    main()
