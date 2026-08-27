# SparkDeck

**Built by a DGX Spark GB10 cluster owner, for other DGX Spark GB10 owners.**

Running one DGX Spark is straightforward. The moment I added more, the practical questions multiplied: Which models actually fit? What inference speed should I expect? Which Spark has the weights? Is every node healthy? Which runtime and configuration really wins?

I built SparkDeck for my own GB10 cluster to answer those questions in one local-first interface. It brings vLLM, llama.cpp `llama-server`, and SGLang together, coordinates paired machines, stages model weights where the work will run, and records comparable performance evidence. I am sharing it so other Spark owners can spend less time coordinating machines and more time running models.

The cluster stays yours. Management remains local, community sharing is opt-in, and SparkDeck never uploads prompts or generated responses.

## Example render

![SparkDeck dark-mode dashboard showing a four-node DGX Spark cluster](docs/screenshots/readme/sparkdeck-dashboard-dark.png)

_A four-node SparkDeck dashboard in dark mode. Every value shown in this README is illustrative demo data, not a measured hardware claim._

## Why I built it

### Know what performance to expect before pulling a model

The **Community Run Models** catalog gives me a directional inference tok/s estimate before spending time downloading and deploying a model. Today those estimates are aggregated from the measurements my own cluster records; measurements from other SparkDeck users appear only when SparkDeck is pointed at a separately configured community aggregates service, since hosted community sync is not available in this release. Evidence is matched by exact model ID and context window; it is an estimate, not a GB10-specific guarantee, because runtime, quantization, hardware, parallelism, and other settings still matter.

![Community Run Models with illustrative throughput and cluster-fit evidence](docs/screenshots/readme/community-performance-dark.png)

_Community throughput evidence and per-node fit. Illustrative demo data; estimates are evidence, not guarantees._

### Manage the whole GB10 cluster from one controller

SparkDeck pairs machines over a private network, gives every node a recognizable identity, and shows membership and reachability from one controller. Workers run assigned jobs and provide alternate private entry points, while the controller remains authoritative for cluster management.

![Cluster management with one controller and three DGX Spark workers](docs/screenshots/readme/cluster-management-dark.png)

_Controller and worker membership across a private cluster network. Illustrative demo data._

### Put model weights where the work will run

My Sparks are ConnectX-7-equipped for distributed inference, but a fast fabric is useful only when the right weights are already on the right machines. **Virtual NAS** inventories complete Hugging Face cache copies per node, queues authenticated transfers, and tracks progress without exposing arbitrary host paths.

When SparkDeck node addresses are configured on a private ConnectX-7 network, Virtual NAS traffic can take that path. The current transfer engine streams over SparkDeck's authenticated HTTP transport; it does not yet use RDMA.

![Virtual NAS model inventory across three DGX Spark nodes](docs/screenshots/readme/virtual-nas-inventory-dark.png)

![Virtual NAS transfer progress between paired DGX Spark nodes](docs/screenshots/readme/virtual-nas-transfer-dark.png)

_Complete cached model weights staged across paired nodes, with an active copy in the transfer queue. Illustrative demo data._

### Compare configurations, not hunches — work in progress

SparkDeck already records successful proxied runs with runtime, quantization, throughput, latency, and time to first token. The next step is a faster configuration-benchmark loop: vary context length, parallelism, memory settings, and launch arguments, then rank like-for-like runs side by side.

![Work-in-progress benchmark history with illustrative runtime results](docs/screenshots/readme/benchmark-history-dark.png)

_Work in progress: the shipped history view is shown with illustrative data; configuration-aware comparison and ranking are planned._

## What SparkDeck does

- Search Hugging Face models and inspect locally available artifacts.
- Launch and manage models with vLLM, llama.cpp, or SGLang.
- Chat with one model or compare model responses side by side.
- Track host health, model logs, images, queues, and deployment settings.
- Capture local benchmark measurements such as time to first token, output throughput, token counts, and latency, with configuration comparison still in progress.
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

Community upload payloads contain exactly three fields: `model_id`, `context_window_size`, and `inference_tokens_per_second`. Richer benchmark details remain local; SparkDeck does **not** upload prompts, generated text, API keys, endpoint URLs, hostnames, IP addresses, hardware details, runtime settings, token counts, revisions, quantization, latency, or local filesystem paths. Community sharing is off by default and requires explicit consent. Without a trusted ingestion service and node-scoped credential, samples stay queued locally; queued samples remain reviewable and deletable locally.

Community results should be treated as evidence, not a guarantee. Hardware, runtime versions, quantization, context length, concurrency, and parallelism all materially affect performance.

## Community sign-in (Cognito email + password)

Community features (shared benchmark telemetry and community estimates) require an account, created and used from **Settings → Community Features**. The app renders a native email + password form and calls the Amazon Cognito IDP HTTPS API (`https://cognito-idp.us-east-2.amazonaws.com/`) directly — there is no hosted UI, no redirect, and no third-party identity provider. Sign-up uses `SignUp`/`ConfirmSignUp` (Cognito emails a verification code; Cognito's default email sender is used, which is fine for low volume), sign-in uses `InitiateAuth` with the `USER_PASSWORD_AUTH` flow, and sessions refresh with `REFRESH_TOKEN_AUTH`. Password reset is also handled in-app via `ForgotPassword`/`ConfirmForgotPassword` from the same form. All authentication traffic goes straight from the browser to Cognito over HTTPS; passwords never touch the SparkDeck backend.

After a successful sign-in the frontend sends the Cognito ID token to the SparkDeck backend (`POST /api/v1/community/pair`), which verifies it against the pool's JWKS before marking the device paired. Signing out (`DELETE /api/v1/community/pair` via the app's Sign out button, plus a best-effort `RevokeToken` call) returns the device to `not_paired`.

**Sessions are effectively indefinite.** The pool issues refresh tokens valid for 3650 days (the Cognito maximum). Tokens are stored in the browser's localStorage, so sign-in survives browser restarts; an expired ID token is refreshed silently on load without user interaction.

**Sign-in propagates across the cluster.** After pairing locally, the controller pushes the sign-in (and later the sign-out) to every enabled peer node over the cluster-private agent channel (`PUT`/`DELETE /api/agent/community-pairing`). Propagation is best-effort and never blocks sign-in: a node that is already paired with a *different* account is never overridden — this applies to the controller itself too, which refuses the sign-in and tells you which account it holds. Conflicts and unreachable nodes are reported in **Settings → Community Features** and pick up the state on the next sign-in or sign-out. Pairing also promotes any locally queued "waiting for account" benchmark uploads. They are drained by a lifecycle-owned worker when that node has a trusted upload endpoint and node-scoped credential; otherwise the UI reports them honestly as queued locally.

Community upload workers are configured per node with `SPARKDECK_COMMUNITY_UPLOAD_URL` and `SPARKDECK_COMMUNITY_UPLOAD_TOKEN`. The URL is the trusted ingestion API base and the token must be a revocable, upload-only credential for that node. SparkDeck never forwards Cognito browser tokens or refresh tokens to this endpoint or across the cluster. Each request sends one sample containing exactly the three documented fields and places the local sample UUID only in the `Idempotency-Key` header. Redirects and non-public destinations are rejected.

The infrastructure lives in `infra/cognito-community.yml` (CloudFormation stack `sparkdeck-community-auth`): a user pool with email-as-username, self sign-up, and auto-verified email, plus a public app client (no secret; `ALLOW_USER_PASSWORD_AUTH` + `ALLOW_REFRESH_TOKEN_AUTH`; revocation enabled; ID/access tokens 60 minutes, refresh tokens 3650 days).

The backend defaults to the deployed pool and can be pointed at another one with environment variables (these identifiers are not secret):

- `SPARKDECK_COGNITO_USER_POOL_ID` (default `us-east-2_TjntedtdI`)
- `SPARKDECK_COGNITO_CLIENT_ID` (default `30ihrkeg4k1rn95d4mmkq00fvl`)
- `SPARKDECK_COGNITO_ISSUER` (default `https://cognito-idp.us-east-2.amazonaws.com/us-east-2_TjntedtdI`)

## Cluster and MCP automation

SparkDeck nodes can be paired over a trusted management network for distributed and replicated deployments. Pairing credentials are stored locally beneath `data/`; do not commit or share that directory. The controller is the authoritative cluster registry, but every online worker can also display the controller's current one-time pairing code and act as an onboarding entry point at its own Tailscale URL. The joining node verifies the worker's pinned controller referral, registers directly with that controller, and then uses the controller directly for normal management traffic.

### RouterOS switch integration

The **Switch** view supports MikroTik switches running RouterOS v7 with its [REST API](https://help.mikrotik.com/docs/spaces/ROS/pages/47579162/REST%2BAPI). Configure a certificate and enable the `www-ssl` HTTPS service as described in [RouterOS Services](https://help.mikrotik.com/docs/spaces/ROS/pages/103841820/Services); do not expose credentials through the unencrypted `www` service. Create a dedicated RouterOS user in a custom, least-privilege group with only the `rest-api` and read/write policies SparkDeck needs, restrict its allowed source addresses, and keep the management endpoint on a private LAN or Tailscale-routed network. See the [RouterOS user and policy guide](https://help.mikrotik.com/docs/spaces/ROS/pages/8978504/User).

![RouterOS switch monitoring and fan settings](docs/screenshots/routeros-switch.png)

_Representative RouterOS fixture showing switch identity, live health sensors, supported fan controls, and per-interface traffic statistics._

Automatic discovery uses MNDP and is limited to the local Layer-2 broadcast domain. Routed VLANs, remote cluster nodes, and many Tailscale layouts therefore require the switch's HTTPS URL to be entered manually. See [RouterOS neighbor discovery](https://help.mikrotik.com/docs/spaces/ROS/pages/24805517/Neighbor%2Bdiscovery).

Available telemetry and controls depend on the switch model, sensors, fan controller, and RouterOS version. SparkDeck shows only the temperature, voltage, power, fan-speed, and fan-setting fields reported by that device; unsupported controls remain unavailable. MikroTik documents the model and version limitations in [System Health](https://help.mikrotik.com/docs/spaces/ROS/pages/25690117/Health). Switches booted into SwOS are not supported because [SwOS has no API or other programmatic management interface](https://help.mikrotik.com/docs/spaces/SWOS/overview); a dual-boot model must be running RouterOS for this integration.

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

### Cluster-wide release updates

After installing the bundled user service, open **Settings → Software update** to choose from the official GitHub repository's published releases. Select the newest release to upgrade, or an older compatible release to roll back. An install started from any joined node is forwarded to the controller. The controller preflights the entire cluster, updates and verifies workers one at a time, and restarts itself last. Model data, local settings, credentials, untracked files, and running Docker workloads are not replaced.

Self-update is intentionally release-only: it never deploys an arbitrary branch or URL. Every node must use the official Git origin, have a clean tracked checkout, run the bundled `sparkdeck.service`, and already support the update protocol. A dirty, offline, divergent, or unsupported node blocks the rollout before any node changes. Because older installations do not expose this protocol, install the release containing this feature manually on every node once; later releases can update the whole cluster from the Settings button.

SparkDeck resolves the selected release tag to an immutable commit, verifies its updater/data compatibility manifest, and stages the Python and frontend builds in a temporary worktree. Upgrades fast-forward the installed checkout; downgrades use a detached checkout so the user's branch pointer is preserved. Only releases in the same linear history are accepted. After restarting through systemd, SparkDeck verifies the selected revision is healthy and automatically restores the previous revision if that health check fails. If no release exists, Settings reports that honestly and does not fall back to `main`.

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
