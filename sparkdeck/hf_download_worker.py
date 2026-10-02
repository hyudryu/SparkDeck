"""Private supervised Hugging Face writer. Credentials arrive through stdin."""

import json
import sys

from sparkdeck.virtual_nas import _enable_hf_xet_high_performance, validate_model_id, validate_revision, _is_commit_sha


def main() -> None:
    payload = json.load(sys.stdin)
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
