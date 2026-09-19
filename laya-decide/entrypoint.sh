#!/usr/bin/env bash
# Translate SparkDeck's launch arguments into the decision server's environment.
#
# SparkDeck passes an argv list to every managed runtime. Laya is configured
# entirely through the environment, so this shim accepts the small, documented
# flag set below and ignores nothing silently: an unknown flag fails the launch
# instead of starting a server that quietly ignores an operator's intent.
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: entrypoint.sh [--model ID] [--host ADDR] [--port N] [--device DEV]
                     [--max-concurrency N] [--served-model-name NAME]
                     [-- <extra args for uvicorn>]

  --model ID             Hugging Face repo id or local path to load
                         (env: LAYA_MODEL, default convaiinnovations/laya)
  --host ADDR            bind address (default 0.0.0.0)
  --port N               bind port (env: LAYA_PORT, default 8080)
  --device DEV           cuda, cuda:1, mps, or cpu (env: LAYA_DEVICE, auto)
  --max-concurrency N    parallel decision requests (env: LAYA_MAX_CONCURRENCY, 1)
  --served-model-name N  alias reported by /v1/models in place of the repo id
                         (env: LAYA_SERVED_MODEL_NAME)
EOF
    exit 2
}

host="${LAYA_HOST:-0.0.0.0}"
port="${LAYA_PORT:-8080}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            [ "$#" -ge 2 ] || usage
            export LAYA_MODEL="$2"; shift 2 ;;
        --host)
            [ "$#" -ge 2 ] || usage
            host="$2"; shift 2 ;;
        --port)
            [ "$#" -ge 2 ] || usage
            port="$2"; shift 2 ;;
        --device)
            [ "$#" -ge 2 ] || usage
            export LAYA_DEVICE="$2"; shift 2 ;;
        --max-concurrency)
            [ "$#" -ge 2 ] || usage
            export LAYA_MAX_CONCURRENCY="$2"; shift 2 ;;
        --served-model-name)
            [ "$#" -ge 2 ] || usage
            export LAYA_SERVED_MODEL_NAME="$2"; shift 2 ;;
        --)
            shift; break ;;
        -h|--help)
            usage ;;
        *)
            echo "laya-decide: unsupported argument '$1'" >&2
            usage ;;
    esac
done

exec python -m uvicorn server:app --host "$host" --port "$port" "$@"
