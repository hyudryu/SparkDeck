# TensorFold (https://github.com/ashhart/TensorFold) serves OpenAI-compatible
# exact-decoding endpoints on NVIDIA GPUs. Its Linux backend is CUDA-only and
# upstream recommends NVIDIA's PyTorch container, so that is the base here.
FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# SparkDeck mounts each node's Hugging Face cache at the HF_HOME path above, so
# weights already staged by Virtual NAS are reused instead of re-downloaded.
# TensorFold ships no PyPI package, so the install follows upstream's own
# instruction and tracks the repository default branch.
RUN pip install --no-cache-dir "git+https://github.com/ashhart/TensorFold.git"

EXPOSE 8080

# SparkDeck's launcher sends `serve MODEL --host 0.0.0.0 --port 8080 [flags]`,
# so the image entrypoint must be the tensorfold CLI itself.
ENTRYPOINT ["tensorfold"]
