# TensorFold native runtime image

[TensorFold](https://github.com/ashhart/TensorFold) ships its native (Zig) CUDA
server, `tensorfold-native`, as release binaries. This image packages the
Linux release for SparkDeck's managed TensorFold runtime with the same
launcher contract as the Python-engine image: the `tensorfold` CLI as the
entrypoint (`serve MODEL --host 0.0.0.0 --port 8080 [flags]`).

The native engine is the one to use for sharded (tensor-parallel) TensorFold
deployments: it serves GLM-5.3-Flash's NVIDIA FP4-Local and NVFP4 checkpoints
as stored, and it places cross-host ranks with the same `--tp`, `--rank` and
`--master` flags the Python engine takes. The Python engine (TensorFold
0.6.6, packaged in `../tensorfold/`) reads only EXL3 and MLX checkpoints and
cannot load FP4-Local weights at all.

## Building the image

```bash
docker build -t sparkdeck/tensorfold-native:1.3.4 .
```

The build downloads the pinned release archive from upstream and verifies its
sha256. Override the version with `--build-arg TENSORFOLD_VERSION=…`. The
image is per-CPU-architecture: build it on an aarch64 host (a DGX Spark) for
the Sparks.

## Sharded deployments

A TensorFold sharded deployment needs this image (or any image keeping the
`tensorfold` CLI as its entrypoint) on every selected node. Set it as the
deployment's image override; SparkDeck's default `sparkdeck/tensorfold:latest`
stays the Python engine for single-node deployments. The launcher mounts the
node's Hugging Face cache, runs both ranks on the host network with the
node's RoCE devices, and injects `--tp/--rank/--master` per rank — rank 0
serves the OpenAI-compatible API, the other ranks follow its steps.

## Notes

- The NVIDIA driver must be R580 or newer (CUDA 13) on every node.
- NCCL and the verbs stack come from the base image plus `rdma-core`; the
  engine's RoCE all-gather dlopens `libibverbs` at run time.
- Weights resolve through the node's shared Hugging Face cache, mounted at
  `HF_HOME=/root/.cache/huggingface`; a local checkpoint directory path is
  served as stored.
