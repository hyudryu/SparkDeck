# Laya decisions on SparkDeck

[Laya](https://huggingface.co/convaiinnovations/laya) is a non-autoregressive
**System 1 decision model**. Give it a state (text, email, ticket, or JSON
document) and typed questions, and it returns typed answers with calibrated
probabilities and confidence scores:

| Question type | Returns |
| --- | --- |
| `choice` | The selected option, a probability per option, and a confidence score |
| `score` | The expected level on your ordinal rubric, the full distribution, and confidence |
| `noul` | A calibrated boolean probability `P(true)` from `0.0` to `1.0` |

It never generates text, so there is nothing to parse and no hallucinated prose.
That also means it does not fit a chat-completions contract on its own: it needs
`state` and `questions`, not `messages`. This directory supplies the missing
piece — a small HTTP server that wraps the decision engine in the same
OpenAI-compatible `/v1` surface SparkDeck already proxies, so a Laya deployment
is reachable through the normal cluster router at `POST <node>:7878/v1/chat/completions`
with no special-case code in the router.

## Files

| File | Purpose |
| --- | --- |
| `server.py` | FastAPI decision server: `/v1/models`, `/v1/chat/completions`, `/laya/predict`, `/healthz` |
| `entrypoint.sh` | Translates SparkDeck's launch argv into the server's environment |
| `Dockerfile` | PyTorch CUDA runtime + `laya` + FastAPI, weights read from the node's Hugging Face cache |

## Build and publish the image

SparkDeck launches managed runtimes from a container image. The default Laya
image is `sparkdeck/laya-decide:latest`; build and publish it once, then point
deployments at it (or override `image` per deployment for a private registry).

```bash
docker build -t sparkdeck/laya-decide:latest laya-decide
docker push sparkdeck/laya-decide:latest
```

Weights are **not** baked into the image. SparkDeck mounts each node's Hugging
Face cache at the `HF_HOME` declared by the image
(`/root/.cache/huggingface`), so the first launch downloads
`convaiinnovations/laya` once per node and later launches reuse it. A weight set
already staged by Virtual NAS is reused as-is.

## Request contract

Send the decision payload as top-level body fields. Standard OpenAI fields
(`model`, `stream`, `max_tokens`, `user`) are accepted alongside them.

```bash
curl http://100.x.x.x:7878/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "convaiinnovations/laya",
    "state": {
      "from": "customer@acme.com",
      "subject": "Duplicate billing on March invoice #4411",
      "body": "Hi team, we were billed twice for March. Please refund the duplicate before Friday or we will cancel our plan."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs, outages, integrations",
          "sales": "pricing, contracts, demos",
          "other": "everything else"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
      },
      "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or switch to a competitor?"
      }
    }
  }'
```

The response is a normal `chat.completion`. The assistant message carries the
decision as compact JSON, and the same answers are repeated as a typed `laya`
extension so callers do not have to parse the content string:

```json
{
  "object": "chat.completion",
  "model": "convaiinnovations/laya",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "{\"answers\":{\"department\":{\"type\":\"choice\",\"choice\":\"billing\",\"probabilities\":{\"billing\":0.9734,\"technical\":0.0098,\"sales\":0.0098,\"other\":0.007},\"confidence\":0.8907,\"action\":{\"act_probability\":1.0}}, ...}}",
      "laya": { "model": "laya-rl-agent", "answers": { "department": { "choice": "billing", "confidence": 0.8907 } } }
    },
    "finish_reason": "stop"
  }],
  "usage": { "prompt_tokens": 278, "completion_tokens": 0, "total_tokens": 278 }
}
```

`usage.completion_tokens` is always `0`: Laya answers in one forward pass and
generates no tokens. `usage.prompt_tokens` is Laya's real token count, so
SparkDeck's accounting and benchmark history stay meaningful.

### Alternate payload spellings

| Spelling | Use |
| --- | --- |
| Top-level `state` + `questions` | Canonical form |
| `system` message containing a JSON object with `state`/`questions` | Plain OpenAI clients that can only send messages |
| `presets` instead of `questions` | Built-in Laya question sets: `router`, `guard`, `moderation`, `triage` |

A top-level `state`/`questions` pair always wins over one embedded in a message.
Presets layer the built-in questions in and any explicit `questions` entries
override them by id; combining presets namespaces colliding ids as
`<preset>_<id>` rather than silently overwriting.

### Streaming

`"stream": true` returns `text/event-stream`. Because the decision is computed
in one pass, the server replays the finished answer as a role chunk, a content
chunk with `finish_reason: "stop"`, a usage-only chunk, and `data: [DONE]` — a
valid SSE sequence any OpenAI client can consume.

## Laya-native endpoint

`POST /laya/predict` accepts the same `state`/`questions` payload and returns
Laya's native result without the OpenAI envelope. Useful for direct calls and
for debugging what the model actually produced.

## Running the server outside a container

```bash
pip install laya fastapi 'uvicorn[standard]'
cd laya-decide
LAYA_MODEL=convaiinnovations/laya python -m uvicorn server:app --host 0.0.0.0 --port 8080
```

An already-running server can also be registered in SparkDeck as an **external**
deployment with a `base_url` instead of a managed one.

## Launch options

`entrypoint.sh` accepts the argv SparkDeck builds for a managed deployment:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model ID` | `convaiinnovations/laya` | Hugging Face repo id or local path to load |
| `--host ADDR` | `0.0.0.0` | Bind address |
| `--port N` | `8080` | Bind port |
| `--device DEV` | auto | `cuda`, `cuda:1`, `mps`, or `cpu` |
| `--max-concurrency N` | `1` | Parallel decision requests; the model is tiny, so one GPU request at a time is the safe default |
| `--served-model-name N` | repo id | Alias reported by `/v1/models` in place of the repo id |

An unrecognized flag fails the launch instead of starting a server that silently
ignores the operator's intent.

## Runtime characteristics in SparkDeck

- **Layouts:** `single` and `replicated`. Laya has no tensor or pipeline
  parallelism, so SparkDeck rejects `sharded` and `grouped_sharded` layouts
  rather than degrading them silently. Replicating across nodes is how you
  scale, and the controller load-balances across replicas normally.
- **Placement:** choose the nodes that already hold the weights; a deployment
  runs one complete decision server per selected node.
- **Co-existence:** unlike SGLang and llama.cpp, a Laya launch does not evict
  other engines. At 421M parameters it fits alongside a chat model on one node.
- **Planning:** the catalog marks Laya as deployable only for checkpoints tagged
  `laya`, so unrelated Transformers models are not offered the runtime.

## Model limitations

Laya is text-only English, with a 512-token budget per question; longer states
are truncated. Calibration is measured on the publisher's benchmark datasets, so
evaluate on your own distribution before automating on a confidence threshold.
Arithmetic, counting, date comparisons, and multi-hop index lookups belong in
deterministic code rather than in a decision question.

## Verifying an installation

`tools/laya_live_check.py` loads the published weights and drives the server's
HTTP surface end to end (completion, system-message payload, streaming,
presets, and validation errors):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install laya fastapi httpx
python tools/laya_live_check.py
```

Run `python -m pytest tests/test_laya_decision_server.py tests/test_laya_runtime_manager.py`
after changing either `server.py` or the runtime adapter.
