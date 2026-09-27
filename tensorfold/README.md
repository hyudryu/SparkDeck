# TensorFold runtime image

[TensorFold](https://github.com/ashhart/TensorFold) is an OpenAI-compatible
serving engine with exact speculative decoding for NVIDIA GPUs (and Apple
Silicon via MLX, which this Linux image does not target). SparkDeck launches it
as a managed runtime with the engine name `tensorfold`.

## Building the image

SparkDeck's default image tag is `sparkdeck/tensorfold:latest`. TensorFold has
no upstream container or PyPI package, so build the tag once per cluster (or
push it to your registry and set the image override on the deployment):

```bash
docker build -t sparkdeck/tensorfold:latest .
```

The image installs TensorFold from its repository default branch and exposes
the `tensorfold` CLI as the entrypoint, which is the contract SparkDeck's
launcher relies on (`serve MODEL --host 0.0.0.0 --port 8080 [flags]`).

## Notes

- The CUDA backend requires an NVIDIA GPU and driver on every node the
  deployment targets; launches are rejected before eviction when a node
  reports none.
- Weights resolve through the node's shared Hugging Face cache, which the
  image mounts at `HF_HOME=/root/.cache/huggingface`.
- Single and replicated layouts are supported; TensorFold has no tensor or
  pipeline parallelism, and model revisions cannot be pinned.
- A custom registry tag can be passed as the deployment's image override; it
  must keep the `tensorfold` CLI as its entrypoint.
