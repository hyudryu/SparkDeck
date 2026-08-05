# VLLMController

Web UI for managing NVIDIA vLLM containers — pull LLMs, queue inference, and monitor CPU/GPU/RAM/temp/power in real time. Single-process FastAPI + vanilla JS, Tailscale-friendly.

Each LLM runs in its own container. The controller orchestrates lifecycle, queues inference requests, retries on failure, and auto-stops idle models. Built for a single-GPU host (developed on NVIDIA GB10 / Grace + Blackwell) and intended to be reachable over a private network.

## Quick start

```bash
./run.sh
# then open http://<host>:7878
```

## Two-node cluster setup

Run the same VLLMController checkout on both nodes. Node 1 remains the web UI
and coordinator; node 2 exposes an authenticated agent API from the same
FastAPI service.

1. Connect and configure the ConnectX interfaces on both machines. Keep this
   inference fabric separate from the normal LAN/Tailscale management address.
2. Start VLLMController on both nodes.
3. On node 2, read the one-time pairing code:

   ```bash
   jq -r .pairing_code data/agent.json
   ```

4. On node 1, open **Cluster → Pair another node**. Enter node 2's management
   URL (for example `http://spark-2:7878`) and the pairing code.
5. Confirm that both node cards show **online**, Docker **Ready**, and the
   expected ConnectX IP/interface. The fabric values can be overridden under
   Settings when automatic interface detection chooses the wrong NIC.
6. Launch a model and choose one of:

   - **Single node** — one complete model on one selected node.
   - **Sharded across nodes** — one model spanning the selected nodes.
   - **Replicated** — an independent complete model on every selected node.

Sharded vLLM launches use native multi-node multiprocessing with coordinator-
owned rank/master arguments, one tensor-parallel rank per DGX Spark, pipeline
parallelism across nodes, and headless worker ranks. Sharded SGLang launches use coordinated
`--nnodes`, `--node-rank`, `--dist-init-addr`, and TP arguments. Cluster
containers use host networking, `/dev/infiniband` when present, unlimited
memlock, and node-specific NCCL/Gloo interface settings. The public model API
is the rank-0 endpoint on node 1.

Agent credentials and paired-node tokens are stored in `data/agent.json` and
`data/nodes.json` with mode `0600`. Do not expose either controller to the
public internet; use a trusted LAN, Tailscale, or WireGuard management network.

## Calling a model

There are two ways to query a running model. The **API** tab in the UI shows the same examples below with one-click copy buttons.

### ① Direct → vLLM container

Each running model exposes vLLM's full OpenAI-compatible API on its own host port (visible in the Models tab). Lowest overhead, supports streaming, works with any OpenAI client. The model must already be running.

```bash
curl -s http://<host>:<port>/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role":"user","content":"Hello"}],
    "max_tokens": 64
  }'
```

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<host>:<port>/v1",
    api_key="sk-noauth",   # vLLM ignores it
)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

Streaming works the same way — pass `stream=True`.

### ② Through the controller (queued, auto-start, retries)

Submits to the controller's queue. If the model isn't running the controller will start it (subject to `max_concurrent_models`), retry on failure (up to `max_retries`), and refresh the idle clock. Returns a job id immediately — poll for the result.

```bash
# submit
curl -s http://<controller>:7878/api/inference \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role":"user","content":"Hello"}],
    "params": {"max_tokens": 64, "temperature": 0.7}
  }'
# → {"id":"abc12345", "status":"pending", ...}

# poll
curl -s http://<controller>:7878/api/inference/abc12345
# status one of: pending | dispatching | running | done | error | canceled
```

```python
import requests, time

CTRL = "http://<controller>:7878"

def ask(model, prompt, **params):
    job = requests.post(f"{CTRL}/api/inference", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "params": params,
    }).json()
    while True:
        time.sleep(0.5)
        s = requests.get(f"{CTRL}/api/inference/{job['id']}").json()
        if s["status"] == "done":
            return s["result_text"]
        if s["status"] in ("error", "canceled"):
            raise RuntimeError(s.get("error") or s["status"])

print(ask("Qwen/Qwen2.5-Coder-7B-Instruct", "Hello", max_tokens=64))
```

#### Override the concurrency limit on a queued job

```bash
curl -s -X POST http://<controller>:7878/api/queue/<job_id>/run
```

## Endpoint reference

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/api/state`                | Aggregate state: containers, queue, settings, stats |
| `GET`    | `/api/stats`                | Host telemetry (CPU/GPU/RAM/temps/power) |
| `POST`   | `/api/inference`            | Submit a chat-completion job to the queue |
| `GET`    | `/api/inference/{id}`       | Poll job status / result |
| `POST`   | `/api/queue/{id}/run`       | Override max-concurrent and run now |
| `DELETE` | `/api/queue/{id}`           | Cancel a pending/dispatching job |
| `POST`   | `/api/queue/clear`          | Drop finished jobs from the queue |
| `GET`/`POST` | `/api/settings`         | Read/update controller settings |
| `GET`/`POST`/`DELETE` | `/api/containers[/...]` | List/create/start/stop/remove containers, fetch logs |
| `POST`/`PATCH`/`DELETE` | `/api/nodes[/...]` | Pair, configure, probe, or remove cluster nodes |
| `POST`/`GET` | `/api/deployments/{id}/...` | Start/stop/remove a deployment or collect member logs |
| `GET`/`POST`/`DELETE` | `/api/agent/...` | Authenticated node status, pairing, and member lifecycle API |
| `POST`   | `/api/images/pull`          | Pull a docker image (SSE progress stream) |

## Auto-stop on idle

Each managed container is auto-stopped after `idle_timeout_seconds` (default **30s**) of inactivity. The idle monitor:

- watches each container's vLLM `/metrics` request counters and resets the clock whenever the counter advances,
- bumps the clock whenever the controller dispatches a queued job to that container,
- only acts on **managed** containers (those launched through the controller — externals are left alone),
- transparently restarts a stopped container the next time an inference request targets it.

Set the value to **0** in Settings to disable auto-stop entirely.

## Settings

Persisted in `data/settings.json`. All fields are editable from the Settings tab.

| Setting | Default | Notes |
|---|---|---|
| `max_concurrent_models` | `1` | How many model containers may run at once |
| `max_retries` | `2` | Retries on inference failure (0 = none) |
| `idle_timeout_seconds` | `30` | Auto-stop after this many seconds of inactivity (0 disables) |
| `vllm_image` | `nvcr.io/nvidia/vllm:26.03.post1-py3` | Image used for new model containers |
| `hf_cache` | `~/.cache/huggingface` | Bind-mounted into each container at `/root/.cache/huggingface` |
| `shm_size` | `16g` | `--shm-size` for new containers |
| `default_gpu_memory_utilization` | `0.9` | Pre-fills the launch form |
| `port_range_start` / `port_range_end` | `8000` / `8099` | Auto-allocated host ports |
| `cluster_node_name` | Hostname | Name displayed on the Cluster tab |
| `cluster_fabric_ip` | Auto-detect | ConnectX/data-plane address for distributed inference |
| `cluster_fabric_interface` | Auto-detect | Interface used for NCCL and Gloo collectives |

## Stack

- **Backend** — FastAPI, Docker SDK for Python, `httpx`, `nvidia-smi`, `/proc`, thermal zones.
- **Frontend** — vanilla JS + Alpine.js + hand-written CSS. No build step. Three files served by FastAPI.

## Requirements

- Linux host with Docker and NVIDIA Container Toolkit
- An NVIDIA GPU (tested on GB10) with `nvidia-smi`
- Python 3.10+
- A Hugging Face cache (`~/.cache/huggingface`) — `huggingface-cli login` first if you plan to pull gated models

## Notes

- **No auth.** Designed to live behind a private network (Tailscale, WireGuard, SSH tunnel). Don't expose it to the public internet as-is.
- **Unified-memory caveat.** On Grace+Blackwell SoCs, `nvidia-smi` reports `N/A` for VRAM because GPU and CPU share memory; the RAM gauge in the UI is the true unified pool.
- **Single-GPU concurrency.** Running >1 model on one GPU requires reducing `--gpu-memory-utilization` per container; the queue's ▶ Run-now button lets you override `max_concurrent_models` on demand.
