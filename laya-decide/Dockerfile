FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Laya is a 421M-parameter ModernBERT-large decision model. The CUDA base image
# keeps the same torch build on every SparkDeck node, so a deployment needs no
# driver-side tuning beyond the usual NVIDIA container runtime.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LAYA_MODEL=convaiinnovations/laya \
    LAYA_PORT=8080

# SparkDeck mounts each node's Hugging Face cache at the HF_HOME path above, so
# a weight set already staged by Virtual NAS is reused instead of re-downloaded.
RUN pip install --no-cache-dir \
        "laya==0.1.6" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27" \
        "pydantic>=2.5"

COPY server.py /opt/laya-decide/server.py
COPY entrypoint.sh /opt/laya-decide/entrypoint.sh
RUN chmod +x /opt/laya-decide/entrypoint.sh

WORKDIR /opt/laya-decide
EXPOSE 8080

# The decision server exposes /v1/models and /v1/chat/completions plus the
# Laya-native /laya/predict route.
ENTRYPOINT ["/opt/laya-decide/entrypoint.sh"]
