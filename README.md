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
owned rank/master arguments and headless worker ranks. By default, pipeline
parallelism spans the selected nodes; recipes can explicitly select a valid
tensor/pipeline layout whose product equals the node count. Sharded SGLang launches use coordinated
`--nnodes`, `--node-rank`, `--dist-init-addr`, and TP arguments. Cluster
containers use host networking, `/dev/infiniband` when present, unlimited
memlock, and node-specific NCCL/Gloo interface settings. The public model API
is the rank-0 endpoint on node 1.

Stopped cluster deployment cards expose **Launch settings**. The editor shows
the saved common arguments and each rank's generated runtime flags. Common
vLLM/SGLang controls are parsed into dedicated fields, including context,
concurrency, KV-cache dtype, thinking, speculative-token count, CUDA-graph
capture size, and batched-token limits. Saving
marks the deployment for rebuild, and the next Start recreates every member
with the edited model, image, memory, topology, and command-line settings.
The same panel exposes cache-miss input, cache-hit input, and output pricing
per million tokens. Pricing is accounting metadata, can be saved while a
cluster is running, and does not trigger a rebuild.
Cluster deployments and standalone Docker model cards also have a **Rename**
action. These aliases are display-only, persist across controller restarts,
and never change Docker identities, API model names, routing, or running state.

The **Temps** tab records one-second CPU and GPU temperature runs for any
online cluster node. A run arms at a configurable target plus trigger margin,
starts when the hotter sensor reaches that threshold, and stops automatically
after the node cools below the target. Saved runs can be renamed, overlaid in
one graph, and exported as PNG or CSV.

The **Usage** table has separate display aliases and merge groups. Assigning
the same merge group to multiple model IDs combines their input, cached,
output, request, cost, and speed columns while preserving the underlying raw
counters. Average speed is calculated over the newest 1 million output tokens
and unions overlapping decode intervals, so concurrent streams report
their aggregate throughput rather than a per-stream average.
Managed vLLM launches enable prompt-token details so prefix-cache hits are
recorded separately and charged at the configured cached-input rate.

The **Rules** section at the bottom of Usage configures directional accounting
routes. A rule such as `xxx/abcdefg → xxx/1234567` rolls the source model's
usage into the destination row and applies the destination model's pricing to
all routed tokens, while preserving the original raw counters and without
changing inference API routing.

Enable **Sync token usage** in Settings on the coordinator and each
participating node to replicate lifetime and hourly per-model counters every
five seconds. The coordinator pulls each node's independently versioned
component and pushes the union back to all participating peers, so every node
shows the same cluster aggregate without counting a replicated snapshot more
than once. The sync endpoints use the existing paired-node authentication.

Agent credentials and paired-node tokens are stored in `data/agent.json` and
`data/nodes.json` with mode `0600`. Do not expose either controller to the
public internet; use a trusted LAN, Tailscale, or WireGuard management network.

### llama.cpp tensor parallelism

GGUF models in the **llama-server** section can use paired Spark GPUs through
llama.cpp's RPC backend. Build the matching worker binary on every Spark from
the same llama.cpp source revision used by `llama-server`:

```bash
cmake -S ~/.unsloth/llama.cpp -B ~/.unsloth/llama.cpp/build-rpc \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_CUDA=ON -DGGML_RPC=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ~/.unsloth/llama.cpp/build-rpc \
  --target ggml-rpc-server -j 8
```

In a model's **Load settings**, enable **Multi-GPU cluster**, choose the GPU
count, and select a split mode. A count of 2 uses the coordinator GPU plus one
paired Spark. The
coordinator starts the remote worker through its authenticated agent API,
binds it only to the selected ConnectX address, and launches `llama-server`
with `--rpc`, the selected `--split-mode`, and an equal `--tensor-split`.
`tensor` is experimental and is unavailable for some architectures, including
DeepSeek 4; use `layer` for those models. Workers are stopped when the model
unloads or its launch fails. The llama.cpp RPC protocol
itself is unauthenticated, so keep the ConnectX fabric isolated and never bind
its port to a public or general-purpose network.

### llama.cpp DSPARK speculative decoding

For a GGUF repository that publishes a separate DSPARK draft model, download
the drafter into the same Hugging Face cache as the target model. For example:

```bash
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf
```

Refresh the Models page, open the target model's **Load settings**, enable
**DSPARK**, and set **Speculative draft tokens**. The controller passes the
downloaded draft with `--spec-draft-model`, selects `draft-dspark`, and fully
offloads the drafter with `--spec-draft-ngl 99`. Q8_0 is preferred when both
Q8_0 and BF16 draft files are cached. DSPARK and MTP are mutually exclusive.

## MCP cluster automation and A/B testing

The controller includes an MCP control plane for recipes, cluster lifecycle,
and reproducible benchmarks. It uses Streamable HTTP on the controller's
existing port at `http://127.0.0.1:7878/mcp`; no second process or port is
required.

For clients that accept an HTTP MCP URL, configure:

```text
http://127.0.0.1:7878/mcp
```

For example, Codex can use this project configuration:

```toml
[mcp_servers.vllm-controller]
url = "http://127.0.0.1:7878/mcp"
```

The server publishes tools to list recipes/deployments, create/update/clone a
recipe, deploy with overrides, wait for readiness, benchmark a deployment,
stop or delete by deployment ID, and run a complete sequential A/B comparison.
`run_cluster_ab_test` launches A, waits, warms up, measures it, frees its GPUs,
then repeats for B using identical prompts. Results include aggregate output
tokens/second, request throughput, mean/p50/p95 latency, the winning variant,
and percentage change. The last 100 outcomes are retained in
`data/mcp_ab_results.json`.

MCP deployments are tagged with `managed_by=vllm-controller-mcp` and an
`automation_run_id`. Stop/delete tools reject deployment IDs created outside
MCP unless the caller explicitly sets `allow_unowned=true`. The A/B tool is
idle-cluster-safe by default and only removes deployment IDs from its own run.

The MCP route follows the same network exposure as VLLMController. To require a
bearer token for MCP requests, set these variables on the controller service:

```bash
VLLM_MCP_TOKEN='replace-with-a-long-random-token' \
VLLM_MCP_PUBLIC_URL='http://your-private-host:7878/mcp' \
./run.sh
```

Send that value as `Authorization: Bearer <token>` from the MCP client. Keep
the endpoint on a trusted private network; bearer authentication does not add
TLS.

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

Requests sent through the controller's OpenAI-compatible endpoint
(`http://<controller>:7878/v1`) use a per-deployment FIFO admission queue. The
deployment's vLLM `--max-num-seqs` setting is the proxy limit: up to that many
requests are forwarded to vLLM, while additional chat, completion, or
controller inference-job requests wait at the controller until a slot opens. A
client disconnect removes its queued request. Inspect live admission state with
`GET /api/inference-queue` or the `inference_admission` field in
`GET /api/state`. Direct container-port calls bypass this controller queue.

The controller also includes an optional adaptive vLLM stream nudger, disabled
by default. When enabled and two or more
forwarded streams remain below 5 aggregate tokens/second for three seconds, it
temporarily lowers that deployment's effective admission limit to one. Newer
streams that have emitted no SSE data are closed upstream, kept connected at
the controller, and transparently replayed through the FIFO queue. A stream
that has emitted any data is never replayed, because doing so could duplicate
or change its output; the nudger only prevents additional concurrent work in
that case. After one stall interval at a limit of one, the controller restores
one slot per interval until it reaches the configured limit. Throughput remains
monitored during recovery; another sustained low-rate interval returns the
effective limit to one. The enable flag, rate threshold, and stall interval are
controller settings:
`vllm_nudger_enabled`, `vllm_nudger_rate_threshold`, and
`vllm_nudger_stall_seconds`.

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
| `GET`    | `/api/temperature-history`  | Selected-node CPU/GPU temperatures (30-second samples, two-hour window) |
| `GET`    | `/api/active-request-rates` | Live thinking/output token rates for the two-second UI poll |
| `POST`   | `/api/inference`            | Submit a chat-completion job to the queue |
| `GET`    | `/api/inference/{id}`       | Poll job status / result |
| `POST`   | `/api/queue/{id}/run`       | Override max-concurrent and run now |
| `DELETE` | `/api/queue/{id}`           | Cancel a pending/dispatching job |
| `POST`   | `/api/queue/clear`          | Drop finished jobs from the queue |
| `GET`/`POST` | `/api/settings`         | Read/update controller settings |
| `GET`/`POST`/`DELETE` | `/api/containers[/...]` | List/create/start/stop/remove containers, fetch logs |
| `POST`/`PATCH`/`DELETE` | `/api/nodes[/...]` | Pair, configure, probe, or remove cluster nodes |
| `POST`/`GET`/`PUT` | `/api/deployments/{id}/...` | Start/stop/remove a deployment, edit launch settings, or collect member logs |
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
| `hf_cache` | `~/.cache/huggingface` | Bind-mounted at the image's `HF_HOME` (or `/root/.cache/huggingface` when unset) |
| `shm_size` | `16g` | `--shm-size` for new containers |
| `default_gpu_memory_utilization` | `0.9` | Pre-fills the launch form |
| `port_range_start` / `port_range_end` | `8000` / `8099` | Auto-allocated host ports |
| `cluster_node_name` | Hostname | Name displayed on the Cluster tab |
| `vllm_auto_adjust_concurrency` | `true` | Floor vLLM's startup full-context KV concurrency and redeploy all ranks when `--max-num-seqs` is higher |
| `sync_token_usage` | `false` | Idempotently aggregate lifetime/hourly usage across participating paired nodes |
| `cluster_fabric_ip` | Auto-detect | ConnectX/data-plane address for distributed inference |
| `cluster_fabric_interface` | Auto-detect | Interface used for NCCL and Gloo collectives |

For each active vLLM deployment, the controller watches startup logs for
`GPU KV cache size` and `Maximum concurrency for … tokens per request`. When
the reported context length matches `--max-model-len` and the floored maximum
is below `--max-num-seqs`, it saves the lower limit, stops every rank, and
recreates the deployment with one coherent configuration. The observed KV
capacity, per-rank reports, and adjustment are retained on the deployment for
diagnostics. Disable **Auto-fit vLLM concurrency** to opt out.

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
