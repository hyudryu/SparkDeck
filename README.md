# SparkDeck

SparkDeck is a local-first control plane for discovering, running, and comparing open models. It brings vLLM, llama.cpp `llama-server`, and SGLang deployments into one responsive interface and records comparable performance results without storing prompt or response content.

The long-term goal is a public community catalog where people can find a model that fits their hardware, compare measured throughput and latency, and optionally contribute their own benchmark results. Cloud sync is being designed around explicit consent and an authenticated upload queue; sharing is off by default.

## What SparkDeck does

- Search Hugging Face models and inspect locally available artifacts.
- Launch and manage models with vLLM, llama.cpp, or SGLang.
- Chat with one model or compare model responses side by side.
- Track host health, model logs, images, queues, and deployment settings.
- Capture local benchmark measurements such as time to first token, output throughput, token counts, and latency.
- Prepare eligible, redacted benchmark samples for future community sync.
- Coordinate compatible SparkDeck nodes for distributed or replicated inference.
- Copy Hugging Face cache weights between paired nodes with the opt-in virtual NAS.

SparkDeck is under active development. The management API is not hardened for direct public-internet exposure. Run it on a trusted network or behind an authenticated reverse proxy.

## Quick start

### Requirements

- Linux with Docker and the NVIDIA Container Toolkit
- Python 3.11 or newer
- Node.js 20 or newer with npm (for the web app build)
- An NVIDIA GPU for GPU-backed runtimes
- A Hugging Face account for gated models

Create a virtual environment, install the dependencies, and start the application:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

Open `http://localhost:7878`. Application state is written beneath `data/`, which is intentionally excluded from version control.

## Supported runtimes

| Runtime | Model format | Typical use |
| --- | --- | --- |
| vLLM | Hugging Face model repositories | High-throughput OpenAI-compatible serving and multi-GPU deployments |
| llama.cpp `llama-server` | GGUF | Efficient local inference and flexible CPU/GPU offload |
| SGLang | Hugging Face model repositories | High-performance serving, structured generation, and distributed deployments |

Runtime-specific settings remain explicit. For example, SparkDeck reports tensor parallelism for vLLM and SGLang, while llama.cpp deployments expose their actual parallel-slot, GPU-layer, RPC, and split configuration.

## OpenAI-compatible API

SparkDeck exposes the familiar model and completion routes:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
```

For example:

```bash
curl http://localhost:7878/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

Requests made through SparkDeck can be queued, measured, and used to refresh the deployment idle timer. Calls made directly to a runtime's own port bypass SparkDeck and cannot be included in its benchmark history.

## Benchmark privacy and community sync

SparkDeck records successful proxied inference measurements locally. Benchmark records may include the model revision and artifact, quantization, runtime and version, hardware class, deployment settings, token counts, latency, time to first token, and generation throughput.

Community upload payloads contain exactly three fields: `model_id`, `context_window_size`, and `inference_tokens_per_second`. Richer benchmark details remain local; SparkDeck does **not** upload prompts, generated text, API keys, endpoint URLs, hostnames, IP addresses, hardware details, runtime settings, token counts, revisions, quantization, latency, or local filesystem paths. Community sharing is off by default. When cloud support becomes available, enabling it will require explicit consent, and queued samples will remain reviewable and deletable locally.

Community results should be treated as evidence, not a guarantee. Hardware, runtime versions, quantization, context length, concurrency, and parallelism all materially affect performance.

## Cluster and MCP automation

SparkDeck nodes can be paired over a trusted management network for distributed and replicated deployments. Pairing credentials are stored locally beneath `data/`; do not commit or share that directory.

### Virtual NAS model transfers

Virtual NAS is an opt-in cluster feature for copying model weights that are already in the Hugging Face cache on one SparkDeck node to another. Open **Storage** in the app, enable virtual NAS, choose a source model, and select one or more online target nodes. A joined worker forwards its normal Storage view to the controller, so every node shows the same cluster-wide inventory and transfer jobs.

Keep every node and transfer on a cluster-private network, preferably Tailscale. The transfer endpoints use the paired node-agent credentials, but SparkDeck is not a public file server and these routes should never be exposed directly to the internet. Pair the nodes first, confirm that their Tailscale addresses report online, and allow enough free space for a complete copy before starting a transfer.

Only complete Hugging Face cache model weights are shown and transferable. Virtual NAS does not browse arbitrary directories, expose cache paths or Hugging Face tokens, or copy unrelated files. Deletion is similarly limited to an exact cached model ID and is refused while the model is serving or participating in an active transfer.

The optional MCP endpoint is served on the application port:

```text
http://127.0.0.1:7878/mcp
```

Example Codex configuration:

```toml
[mcp_servers.sparkdeck]
url = "http://127.0.0.1:7878/mcp"
```

Set `SPARKDECK_MCP_TOKEN` to require a bearer token and `SPARKDECK_MCP_PUBLIC_URL` when the externally visible MCP URL differs from the local default. Older `VLLM_MCP_TOKEN` and `VLLM_MCP_PUBLIC_URL` values are accepted only as migration fallbacks. Bearer authentication does not provide TLS; keep the endpoint private or place it behind a secure reverse proxy.

MCP-created deployments use the current `managed_by=sparkdeck-mcp` marker. SparkDeck continues to recognize the legacy `managed_by=vllm-controller-mcp` marker solely so deployments created by older releases can still be managed safely.

## Service installation

The included `sparkdeck.service` is a portable user-service template. It assumes the repository is checked out at `%h/SparkDeck` and reads optional environment values from `%h/.config/sparkdeck/sparkdeck.env`:

```bash
mkdir -p ~/.config/systemd/user
cp sparkdeck.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now sparkdeck.service
```

Adjust the template if your checkout lives elsewhere. Never place tokens directly in the committed unit file.

## Development

Run backend tests with:

```bash
python -m pytest -q
```

When the frontend workspace is present, its package scripts provide linting, type checking, unit tests, and a production build:

```bash
cd frontend
npm ci
npm run lint
npm run typecheck
npm test
npm run build
```

See [SECURITY.md](SECURITY.md) before reporting a vulnerability. Contributions should use the pull request template and include validation appropriate to the changed runtime or UI flow.

## License

SparkDeck is available under the [MIT License](LICENSE).
