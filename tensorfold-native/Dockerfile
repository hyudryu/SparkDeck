# TensorFold's native (Zig) CUDA server as a SparkDeck runtime image.
#
# Upstream ships release binaries (github.com/ashhart/TensorFold/releases); the
# archive carries the server, its built-in CUDA kernels and libtfgrammar, and
# needs no Python package. A tensor-parallel pair needs NCCL compatible with
# CUDA 13 on both ranks, and the engine's RoCE all-gather dlopens libibverbs,
# so the NVIDIA PyTorch container is the base: it ships both, and it is the
# base the Python-engine TensorFold image already uses.
#
# The SparkDeck launcher contract is unchanged: the entrypoint is the
# `tensorfold` CLI (`serve MODEL --host 0.0.0.0 --port 8080 [flags]`), which
# tensorfold-native implements — including the cross-host `--tp/--rank/
# --master` placement flags the sharded planner injects.
FROM nvcr.io/nvidia/pytorch:26.07-py3

ARG TENSORFOLD_VERSION=1.3.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libibverbs1 ibverbs-providers rdma-core \
    && rm -rf /var/lib/apt/lists/* \
    && case "$(uname -m)" in \
         aarch64|arm64) PLATFORM=linux-aarch64 ;; \
         x86_64) PLATFORM=linux-x86_64 ;; \
         *) echo 'No TensorFold release archive for this CPU' >&2; exit 1 ;; \
       esac \
    && curl -fsLO "https://github.com/ashhart/TensorFold/releases/download/v${TENSORFOLD_VERSION}/tensorfold-${TENSORFOLD_VERSION}-${PLATFORM}.tar.gz" \
    && curl -fsLO "https://github.com/ashhart/TensorFold/releases/download/v${TENSORFOLD_VERSION}/tensorfold-${TENSORFOLD_VERSION}-${PLATFORM}.tar.gz.sha256" \
    && sha256sum -c "tensorfold-${TENSORFOLD_VERSION}-${PLATFORM}.tar.gz.sha256" \
    && mkdir -p /opt/tensorfold \
    && tar -xzf "tensorfold-${TENSORFOLD_VERSION}-${PLATFORM}.tar.gz" -C /opt/tensorfold --strip-components=1 \
    && rm -f "tensorfold-${TENSORFOLD_VERSION}-${PLATFORM}.tar.gz"* \
    && ln -s /opt/tensorfold/bin/tensorfold-native /usr/local/bin/tensorfold

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8080

# SparkDeck's launcher sends `serve MODEL --host 0.0.0.0 --port 8080 [flags]`,
# so the image entrypoint must be the tensorfold CLI itself.
ENTRYPOINT ["tensorfold"]
